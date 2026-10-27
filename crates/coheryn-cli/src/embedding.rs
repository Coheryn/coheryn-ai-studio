use crate::{Cli, daemon};
use clap::Args;
use coheryn_api::{
    client::{Client, ClientEmbeddings},
    embeddings::{
        self, BoundedWriter, EmbeddingRequest, EmbeddingResponse, EmbeddingSpace, Encoding,
        ResponseItem, Usage, Vector,
    },
};
use coheryn_engine::{
    Cancellation, Error, ErrorCode, ResourceLimits, Result,
    embedding::{EmbeddingModel, EmbeddingOptions, EmbeddingResult},
    memory::{Budget, Reservation},
    planner::Profile,
};
use serde::Serialize;
use std::{
    fs::File,
    io::{self, BufRead, BufReader, Read, Write},
    path::PathBuf,
    time::Instant,
};

#[derive(Args)]
pub struct EmbedArgs {
    pub model: PathBuf,
    #[arg(long, required_unless_present = "file", conflicts_with = "file")]
    pub input: Option<String>,
    #[arg(long, required_unless_present = "input", conflicts_with = "input")]
    pub file: Option<PathBuf>,
    #[arg(long)]
    pub output: Option<PathBuf>,
    #[arg(long, requires = "output")]
    pub overwrite: bool,
    #[arg(long)]
    pub standalone: bool,
    #[arg(long, default_value_t = 8)]
    pub batch_size: usize,
    #[arg(long, default_value_t = 512)]
    pub max_input_tokens: usize,
    #[arg(long, default_value_t = 2048)]
    pub max_batch_tokens: usize,
    /// Bounded file window, independent of compute microbatch size.
    #[arg(long, default_value_t = 32)]
    pub request_size: usize,
    #[arg(long)]
    pub truncate: bool,
    #[arg(long, default_value = "balanced")]
    pub profile: String,
    #[arg(long)]
    pub threads: Option<usize>,
    #[arg(long, default_value_t = 1073741824)]
    pub max_memory: u64,
    #[arg(long, default_value = "float")]
    pub encoding_format: String,
    #[arg(long, default_value_t = 1073741824)]
    pub max_output_bytes: u64,
    #[arg(long, default_value_t = 1073741824)]
    pub max_file_bytes: u64,
}
impl EmbedArgs {
    fn options(&self) -> Result<EmbeddingOptions> {
        let o = EmbeddingOptions {
            max_input_tokens: self.max_input_tokens,
            truncate: self.truncate,
            batch_size: self.batch_size,
            max_batch_tokens: self.max_batch_tokens,
            profile: Profile::parse(&self.profile)?,
            threads: self.threads,
        };
        o.validate(512)?;
        if self
            .input
            .as_ref()
            .is_some_and(|s| s.is_empty() || s.len() > 65536)
        {
            return Err(Error::new(
                ErrorCode::EmbeddingInputLimitExceeded,
                "input must contain 1..=65536 UTF-8 bytes",
            ));
        }
        if !(1..=256).contains(&self.request_size)
            || self.max_memory == 0
            || !(1..=1 << 30).contains(&self.max_output_bytes)
            || !(1..=1 << 30).contains(&self.max_file_bytes)
        {
            return Err(Error::request(
                "invalid embedding file/window/memory quotas",
            ));
        }
        Ok(o)
    }
    fn standalone(&self) -> bool {
        self.standalone
            || crate::local_path(&self.model)
            || self.model.extension().is_some_and(|s| s == "gguf")
    }
    fn encoding(&self) -> Result<Encoding> {
        match self.encoding_format.as_str() {
            "float" => Ok(Encoding::Float),
            "base64" => Ok(Encoding::Base64),
            _ => Err(Error::request("encoding_format must be float or base64")),
        }
    }
}
enum Output {
    Stdout(BoundedWriter<Vec<u8>>),
    File(BoundedWriter<tempfile::NamedTempFile>),
}
impl Output {
    fn create(args: &EmbedArgs) -> Result<Self> {
        if let Some(path) = &args.output {
            if !args.overwrite && path.try_exists()? {
                return Err(Error::new(
                    ErrorCode::Conflict,
                    "output exists; use --overwrite explicitly",
                ));
            }
            let parent = path
                .parent()
                .filter(|p| !p.as_os_str().is_empty())
                .unwrap_or(std::path::Path::new("."));
            let file = tempfile::Builder::new()
                .prefix(".coheryn-embedding-")
                .suffix(".part")
                .tempfile_in(parent)?;
            Ok(Self::File(BoundedWriter::new(file, args.max_output_bytes)))
        } else {
            let limit = args.max_output_bytes.min(8 << 20);
            let mut bytes = Vec::new();
            bytes
                .try_reserve_exact(limit as usize)
                .map_err(|_| Error::resource("stdout buffer allocation failed"))?;
            Ok(Self::Stdout(BoundedWriter::new(bytes, limit)))
        }
    }
    fn exceeded(&self) -> bool {
        match self {
            Self::Stdout(w) => w.exceeded,
            Self::File(w) => w.exceeded,
        }
    }
    fn publish(mut self, args: &EmbedArgs, cancel: &Cancellation) -> Result<()> {
        cancel.check()?;
        self.flush()?;
        match self {
            Self::Stdout(w) => {
                io::stdout().lock().write_all(&w.inner)?;
            }
            Self::File(w) => {
                w.inner.as_file().sync_all()?;
                cancel.check()?;
                let path = args.output.as_ref().expect("file output");
                if args.overwrite {
                    w.inner.persist(path).map_err(|e| Error::from(e.error))?;
                } else {
                    w.inner.persist_noclobber(path).map_err(|e| {
                        if e.error.kind() == io::ErrorKind::AlreadyExists {
                            Error::new(
                                ErrorCode::Conflict,
                                "output appeared during export; not overwritten",
                            )
                        } else {
                            Error::from(e.error)
                        }
                    })?;
                }
            }
        }
        Ok(())
    }
}
impl Write for Output {
    fn write(&mut self, b: &[u8]) -> io::Result<usize> {
        match self {
            Self::Stdout(w) => w.write(b),
            Self::File(w) => w.write(b),
        }
    }
    fn flush(&mut self) -> io::Result<()> {
        match self {
            Self::Stdout(w) => w.flush(),
            Self::File(w) => w.flush(),
        }
    }
}
struct Lines {
    reader: BufReader<File>,
    line: usize,
    bytes: u64,
    limit: u64,
    pending: Option<String>,
}
impl Lines {
    fn new(path: &PathBuf, limit: u64) -> Result<Self> {
        let file = File::open(path)?;
        if file.metadata()?.len() > limit {
            return Err(Error::new(
                ErrorCode::EmbeddingInputLimitExceeded,
                "input file exceeds --max-file-bytes",
            ));
        }
        Ok(Self {
            reader: BufReader::new(file),
            line: 0,
            bytes: 0,
            limit,
            pending: None,
        })
    }
    fn next(&mut self, cancel: &Cancellation) -> Result<Option<String>> {
        cancel.check()?;
        if let Some(pending) = self.pending.take() {
            return Ok(Some(pending));
        }
        let mut bytes = Vec::new();
        let n = self
            .reader
            .by_ref()
            .take(65539)
            .read_until(b'\n', &mut bytes)?;
        if n == 0 {
            return Ok(None);
        }
        self.line += 1;
        self.bytes = self
            .bytes
            .checked_add(n as u64)
            .ok_or_else(|| Error::resource("file byte count overflow"))?;
        if self.bytes > self.limit || self.line > 1_000_000 {
            return Err(Error::new(
                ErrorCode::EmbeddingInputLimitExceeded,
                "embedding file exceeds byte/record quota",
            ));
        }
        if bytes.last() == Some(&b'\n') {
            bytes.pop();
            if bytes.last() == Some(&b'\r') {
                bytes.pop();
            }
        }
        if bytes.is_empty() || bytes.len() > 65536 {
            return Err(Error::new(
                ErrorCode::EmbeddingInputLimitExceeded,
                format!("record {} must contain 1..=65536 UTF-8 bytes", self.line),
            ));
        }
        let text = String::from_utf8(bytes)
            .map_err(|_| Error::request(format!("invalid UTF-8 in record {}", self.line)))?;
        Ok(Some(text))
    }
    fn window(
        &mut self,
        count: usize,
        byte_limit: usize,
        cancel: &Cancellation,
    ) -> Result<Vec<String>> {
        let mut values = Vec::new();
        let mut bytes = 0;
        while values.len() < count {
            let Some(text) = self.next(cancel)? else {
                break;
            };
            if bytes + text.len() > byte_limit {
                self.pending = Some(text);
                break;
            }
            bytes += text.len();
            values.push(text);
        }
        Ok(values)
    }
}
pub(crate) enum Backend {
    Local {
        model: Box<EmbeddingModel>,
        space: EmbeddingSpace,
        budget: Budget,
    },
    Remote {
        client: Client,
        id: String,
        budget: Budget,
    },
}
pub struct Batch {
    pub response: EmbeddingResponse,
    pub space_id: String,
    pub model_sha256: String,
    pub dimension: usize,
    pub token_counts: Vec<usize>,
    pub truncated: Vec<usize>,
    pub batch_size: usize,
    pub token_capacity: usize,
    _engine: Option<EmbeddingResult>,
    _serialization: Reservation,
}
impl Backend {
    pub(crate) fn open(cli: &Cli, args: &EmbedArgs, budget: Budget) -> Result<Self> {
        let local = args.standalone();
        if local {
            let hash = coheryn_engine::storage::sha256_file(&args.model)?;
            let model = EmbeddingModel::load_shared(
                &args.model,
                ResourceLimits {
                    context: args.max_input_tokens,
                    max_memory: args.max_memory,
                },
                budget.clone(),
                1,
            )?;
            let space = EmbeddingSpace::new(&hash, model.descriptor(), model.config())?;
            eprintln!(
                "{}",
                serde_json::json!({"event":"standalone_embedding_instance","model_sha256":hash,"embedding_space":space.id})
            );
            Ok(Self::Local {
                model: Box::new(model),
                space,
                budget,
            })
        } else {
            let id = args
                .model
                .to_str()
                .ok_or_else(|| Error::request("model ID must be UTF-8"))?
                .to_owned();
            coheryn_api::registry::validate_id(&id).map_err(|e| e.error)?;
            Ok(Self::Remote {
                client: daemon::client(cli)?,
                id,
                budget,
            })
        }
    }
    pub(crate) async fn embed(
        &self,
        inputs: Vec<String>,
        options: &EmbeddingOptions,
        encoding: Encoding,
        max_memory: u64,
        cancel: &Cancellation,
    ) -> Result<Batch> {
        let (budget, dim) = match self {
            Self::Local { budget, space, .. } => (budget, space.dimension),
            Self::Remote { budget, .. } => (budget, 4096),
        };
        // Includes wire bytes, typed client vectors and serializer overlap for a bounded window.
        let reserve =
            embeddings::response_allowance(inputs.len(), dim, 64 << 20).map_err(|e| e.error)? * 3;
        let serialization = budget.reserve(reserve)?;
        match self {
            Self::Local { model, space, .. } => {
                let texts: Vec<_> = inputs.iter().map(String::as_str).collect();
                let mut result = model.embed_with_hardware(
                    &texts,
                    options,
                    &coheryn_api::hardware::probe(),
                    cancel,
                )?;
                let token_counts = result.data.iter().map(|i| i.prompt_tokens).collect();
                let truncated = result
                    .data
                    .iter()
                    .filter(|i| i.truncated)
                    .map(|i| i.index)
                    .collect();
                let data = std::mem::take(&mut result.data)
                    .into_iter()
                    .map(|item| ResponseItem {
                        object: "embedding".into(),
                        index: item.index,
                        embedding: Vector::Float(item.embedding),
                    })
                    .collect();
                let response = EmbeddingResponse {
                    object: "list".into(),
                    model: space.model_sha256.clone(),
                    data,
                    usage: Usage {
                        prompt_tokens: result.usage.prompt_tokens,
                        total_tokens: result.usage.total_tokens,
                    },
                };
                Ok(Batch {
                    response,
                    space_id: space.id.clone(),
                    model_sha256: space.model_sha256.clone(),
                    dimension: space.dimension,
                    token_counts,
                    truncated,
                    batch_size: result.plan.batch_size,
                    token_capacity: result.plan.token_capacity,
                    _engine: Some(result),
                    _serialization: serialization,
                })
            }
            Self::Remote { client, id, .. } => {
                let mut request = EmbeddingRequest::new(id.clone(), inputs, options);
                request.encoding_format = encoding;
                request.max_memory = Some(max_memory);
                let ClientEmbeddings {
                    response,
                    space_id,
                    model_sha256,
                    dimension,
                    token_counts,
                    truncated,
                    batch_size,
                    token_capacity,
                } = client.embeddings(&request, cancel).await?;
                Ok(Batch {
                    response,
                    space_id,
                    model_sha256,
                    dimension,
                    token_counts,
                    truncated,
                    batch_size,
                    token_capacity,
                    _engine: None,
                    _serialization: serialization,
                })
            }
        }
    }
}
fn json(out: &mut impl Write, value: &impl Serialize) -> Result<()> {
    serde_json::to_writer(out, value).map_err(|e| Error::new(ErrorCode::Io, e.to_string()))
}
pub async fn run(cli: &Cli, args: &EmbedArgs, cancel: &Cancellation) -> Result<()> {
    let options = args.options()?;
    let encoding = args.encoding()?;
    let budget = Budget::new(args.max_memory);
    let _stdout = if args.output.is_none() {
        Some(budget.reserve(args.max_output_bytes.min(8 << 20))?)
    } else {
        None
    };
    let input_capacity = if let Some(text) = &args.input {
        text.len() as u64
    } else {
        (args.request_size as u64 * 65536).min(if args.standalone() {
            8 << 20
        } else {
            128 << 10
        })
    };
    let _input = budget.reserve(input_capacity * 3 + (1 << 20))?;
    let mut out = Output::create(args)?;
    let mut lines = args
        .file
        .as_ref()
        .map(|p| Lines::new(p, args.max_file_bytes))
        .transpose()?;
    let backend = Backend::open(cli, args, budget)?;
    let result = export(
        args, &options, encoding, &backend, &mut lines, &mut out, cancel,
    )
    .await;
    if out.exceeded() {
        return Err(Error::resource(
            "embedding output quota exceeded; use --output with a sufficient --max-output-bytes limit",
        ));
    }
    result?;
    out.publish(args, cancel)
}

async fn export(
    args: &EmbedArgs,
    options: &EmbeddingOptions,
    encoding: Encoding,
    backend: &Backend,
    lines: &mut Option<Lines>,
    out: &mut Output,
    cancel: &Cancellation,
) -> Result<()> {
    out.write_all(b"{\"object\":\"list\",\"data\":[")?;
    let mut count = 0usize;
    let mut tokens = 0usize;
    let mut truncated_count = 0usize;
    let mut identity = None;
    let mut selected = (0, 0);
    let mut hash = String::new();
    let mut dimension = 0;
    loop {
        cancel.check()?;
        let inputs = if let Some(lines) = lines.as_mut() {
            lines.window(
                args.request_size,
                if matches!(backend, Backend::Local { .. }) {
                    8 << 20
                } else {
                    128 << 10
                },
                cancel,
            )?
        } else if count == 0 {
            vec![
                args.input
                    .clone()
                    .ok_or_else(|| Error::request("supply --input or --file"))?,
            ]
        } else {
            vec![]
        };
        if inputs.is_empty() {
            break;
        }
        let batch = backend
            .embed(inputs, options, encoding, args.max_memory, cancel)
            .await?;
        if identity.as_ref().is_some_and(|(id, sha, dim)| {
            id != &batch.space_id || sha != &batch.model_sha256 || *dim != batch.dimension
        }) {
            return Err(Error::new(
                ErrorCode::ModelChanged,
                "embedding-space identity changed between file batches; export aborted",
            ));
        }
        identity = Some((
            batch.space_id.clone(),
            batch.model_sha256.clone(),
            batch.dimension,
        ));
        hash = batch.model_sha256.clone();
        dimension = batch.dimension;
        selected = (batch.batch_size, batch.token_capacity);
        for item in &batch.response.data {
            cancel.check()?;
            if count > 0 {
                out.write_all(b",")?;
            }
            let values = item.embedding.floats()?;
            #[derive(Serialize)]
            struct Row<'a> {
                #[serde(flatten)]
                item: embeddings::ItemView<'a>,
                prompt_tokens: usize,
                truncated: bool,
            }
            json(
                out,
                &Row {
                    item: embeddings::ItemView {
                        index: count,
                        values: &values,
                        encoding,
                    },
                    prompt_tokens: batch.token_counts[item.index],
                    truncated: batch.truncated.contains(&item.index),
                },
            )?;
            count = count
                .checked_add(1)
                .ok_or_else(|| Error::resource("record count overflow"))?;
        }
        tokens = tokens
            .checked_add(batch.response.usage.prompt_tokens)
            .ok_or_else(|| Error::resource("usage overflow"))?;
        truncated_count += batch.truncated.len();
    }
    if count == 0 {
        return Err(Error::request("embedding input file contains no records"));
    }
    out.write_all(b"],\"model\":")?;
    json(out, &args.model.to_string_lossy())?;
    out.write_all(b",\"usage\":")?;
    json(
        out,
        &Usage {
            prompt_tokens: tokens,
            total_tokens: tokens,
        },
    )?;
    out.write_all(b",\"coheryn\":")?;
    json(
        out,
        &serde_json::json!({"embedding_space":identity.map(|i|i.0),"model_sha256":hash,"dimension":dimension,"count":count,"truncated_inputs":truncated_count,"max_input_tokens":options.max_input_tokens,"last_batch_size":selected.0,"last_batch_tokens":selected.1}),
    )?;
    out.write_all(b"}\n")?;
    Ok(())
}

pub async fn bench(cli: &Cli, args: &daemon::BenchArgs, cancel: &Cancellation) -> Result<()> {
    if !(1..=10).contains(&args.iterations) || !(1..=256).contains(&args.batch_items) {
        return Err(Error::request(
            "invalid embedding benchmark iterations/items",
        ));
    }
    let embed = EmbedArgs {
        model: args.model.clone(),
        input: Some(args.prompt.clone()),
        file: None,
        output: None,
        overwrite: false,
        standalone: args.standalone,
        batch_size: args.batch_size,
        max_input_tokens: args.resources.context.unwrap_or(512),
        max_batch_tokens: 2048,
        request_size: 32,
        truncate: false,
        profile: args.profile.clone(),
        threads: args.threads,
        max_memory: args.resources.max_memory,
        encoding_format: "float".into(),
        max_output_bytes: 1 << 30,
        max_file_bytes: 1 << 30,
    };
    let options = embed.options()?;
    let start = Instant::now();
    let backend = Backend::open(cli, &embed, Budget::new(embed.max_memory))?;
    let load_ms = start.elapsed().as_secs_f64() * 1000.;
    let mut runs = Vec::new();
    let mut identity = None;
    for _ in 0..args.iterations {
        let start = Instant::now();
        let result = backend
            .embed(
                vec![args.prompt.clone(); args.batch_items],
                &options,
                Encoding::Float,
                embed.max_memory,
                cancel,
            )
            .await?;
        if identity.as_ref().is_some_and(|id| id != &result.space_id) {
            return Err(Error::new(
                ErrorCode::ModelChanged,
                "benchmark embedding space changed",
            ));
        }
        identity = Some(result.space_id.clone());
        let elapsed = start.elapsed().as_secs_f64();
        runs.push(serde_json::json!({"model_sha256":result.model_sha256,"dimension":result.dimension,"items":result.response.data.len(),"prompt_tokens":result.response.usage.prompt_tokens,"elapsed_ms":elapsed*1000.,"vectors_per_second":result.response.data.len() as f64/elapsed,"input_tokens_per_second":result.response.usage.prompt_tokens as f64/elapsed,"selected_batch_size":result.batch_size,"selected_batch_tokens":result.token_capacity,"engine_memory":result._engine.as_ref().map(|r|r.memory)}));
    }
    crate::print_json(
        &serde_json::json!({"task":"embedding","profile":args.profile,"requested_threads":args.threads,"max_input_tokens":options.max_input_tokens,"hardware":coheryn_api::hardware::probe(),"embedding_space":identity,"backend":"cpu-reference","load_and_identity_verification_ms":load_ms,"cache_condition":"uncontrolled","runs":runs}),
    )
}
