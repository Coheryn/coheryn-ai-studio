use crate::{Cli, Resources, Run, print_json};
use clap::Args;
use coheryn_api::{
    auth, client::Client, config::ServerConfig, service::Service, wire::GenerationRequest,
};
use coheryn_engine::{
    Cancellation, Error, ErrorCode, GenerationOptions, GenerationResult, Model, Result,
    tokenizer::Message,
};
use serde_json::json;
use std::{
    io::{self, BufRead, Read, Write},
    net::IpAddr,
    path::{Path, PathBuf},
    sync::Arc,
};

#[derive(Args)]
pub struct ServeArgs {
    #[arg(long, default_value_t = 536870912)]
    pub image_store_bytes: u64,
    #[arg(long, default_value_t = 134217728)]
    pub owner_image_bytes: u64,
    #[arg(long, default_value_t = 67108864)]
    pub session_image_bytes: u64,
    #[arg(long, default_value_t = 16)]
    pub session_max_images: usize,
    /// Permit public HTTPS image URLs (DNS pinned on every redirect). Disabled by default.
    #[arg(long)]
    pub allow_image_urls: bool,
    #[arg(long, default_value_t = 50331648)]
    pub image_max_body_bytes: usize,
    #[arg(long, default_value_t = 16777216)]
    pub image_max_bytes: u64,
    #[arg(long, default_value_t = 33554432)]
    pub image_max_total_bytes: u64,
    #[arg(long, default_value_t = 4)]
    pub image_max_count: usize,
    #[arg(long, default_value_t = 30)]
    pub image_acquisition_timeout: u64,
    #[arg(long, default_value_t = 3)]
    pub image_redirects: usize,
    #[arg(long)]
    pub session_persistence: bool,
    #[arg(long, default_value_t = 1073741824)]
    pub session_store_bytes: u64,
    #[arg(long, default_value_t = 536870912)]
    pub session_owner_bytes: u64,
    #[arg(long, default_value_t = 2592000)]
    pub session_retention_seconds: u64,
    #[arg(long, default_value_t = 268435456)]
    pub idle_cache_bytes: u64,
    #[arg(long, default_value_t = 67108864)]
    pub session_cache_bytes: u64,
    #[arg(long, default_value_t = 300)]
    pub cache_idle_timeout: u64,

    #[arg(long, default_value = "127.0.0.1")]
    pub host: IpAddr,
    #[arg(long, default_value_t = 11450)]
    pub port: u16,
    #[arg(long, default_value_t = 1073741824)]
    pub max_memory: u64,
    #[arg(long, default_value_t = 8192)]
    pub max_context: usize,
    #[arg(long, default_value_t = 8)]
    pub queue_capacity: usize,
    #[arg(long, default_value_t = 1)]
    pub max_concurrency: usize,
    #[arg(long, default_value_t = 4)]
    pub max_loaded_models: usize,
    #[arg(long, default_value_t = 4294967296)]
    pub max_model_bytes: u64,
    #[arg(long, default_value_t = 21474836480)]
    pub disk_quota: u64,
    #[arg(long, default_value_t = 30)]
    pub queue_timeout: u64,
    #[arg(long, default_value_t = 300)]
    pub generation_timeout: u64,
    #[arg(long, default_value_t = 300)]
    pub embedding_timeout: u64,
    #[arg(long, default_value_t = 4)]
    pub embedding_requests: usize,
    #[arg(long, default_value_t = 1)]
    pub embedding_concurrency: usize,
    #[arg(long, default_value_t = 256)]
    pub embedding_max_items: usize,
    #[arg(long, default_value_t = 65536)]
    pub embedding_max_input_bytes: usize,
    #[arg(long, default_value_t = 1048576)]
    pub embedding_max_body_bytes: usize,
    #[arg(long, default_value_t = 8388608)]
    pub embedding_max_request_bytes: usize,
    #[arg(long, default_value_t = 512)]
    pub embedding_max_input_tokens: usize,
    #[arg(long, default_value_t = 32768)]
    pub embedding_max_total_tokens: usize,
    #[arg(long, default_value_t = 8)]
    pub embedding_max_batch_size: usize,
    #[arg(long, default_value_t = 2048)]
    pub embedding_max_batch_tokens: usize,
    #[arg(long, default_value_t = 16777216)]
    pub embedding_max_response_bytes: usize,
    #[arg(long, default_value_t = 128)]
    pub max_sessions: usize,
    #[arg(long, default_value_t = 64)]
    pub sessions_per_owner: usize,
    #[arg(long, default_value_t = 1048576)]
    pub session_history_bytes: usize,
    #[arg(long, default_value_t = 256)]
    pub session_messages: usize,
    #[arg(long, default_value_t = 128)]
    pub session_turn_keys: usize,
    #[arg(long, default_value_t = 86400)]
    pub session_idle_timeout: u64,
}
pub async fn serve(cli: &Cli, args: &ServeArgs) -> Result<()> {
    let mut credentials = auth::Credentials::load_or_create(&cli.data_dir)?;
    if let Some(path) = &cli.api_token_file {
        credentials.inference_token = token_file(path)?;
    }
    if let Some(path) = &cli.admin_token_file {
        credentials.admin_token = token_file(path)?;
    }
    credentials.validate()?;
    let config = ServerConfig {
        vision: coheryn_api::vision::VisionLimits {
            allow_urls: args.allow_image_urls,
            body_bytes: args.image_max_body_bytes,
            acquisition_timeout_seconds: args.image_acquisition_timeout,
            redirects: args.image_redirects,
            images: coheryn_engine::multimodal::image::ImageLimits {
                max_encoded_bytes: args.image_max_bytes,
                max_total_encoded_bytes: args.image_max_total_bytes,
                max_images: args.image_max_count,
                ..Default::default()
            },
        },
        persistence: coheryn_api::session_store::PersistenceLimits {
            enabled: args.session_persistence,
            total_bytes: args.session_store_bytes,
            owner_bytes: args.session_owner_bytes,
            retention_seconds: args.session_retention_seconds,
        },
        cache: coheryn_api::sessions::CacheLimits {
            global_bytes: args.idle_cache_bytes,
            session_bytes: args.session_cache_bytes,
            idle_seconds: args.cache_idle_timeout,
        },
        sessions: coheryn_api::sessions::SessionLimits {
            images: coheryn_api::sessions::AssetLimits {
                total_bytes: args.image_store_bytes,
                owner_bytes: args.owner_image_bytes,
                session_bytes: args.session_image_bytes,
                session_items: args.session_max_images,
            },
            total: args.max_sessions,
            per_owner: args.sessions_per_owner,
            history_bytes: args.session_history_bytes,
            messages: args.session_messages,
            turn_keys: args.session_turn_keys,
            idle_seconds: args.session_idle_timeout,
        },
        host: args.host,
        port: args.port,
        data_dir: cli.data_dir.clone(),
        max_memory: args.max_memory,
        max_context: args.max_context,
        queue_capacity: args.queue_capacity,
        max_concurrency: args.max_concurrency,
        max_loaded_models: args.max_loaded_models,
        max_model_bytes: args.max_model_bytes,
        disk_quota: args.disk_quota,
        queue_timeout_seconds: args.queue_timeout,
        generation_timeout_seconds: args.generation_timeout,
        embedding_timeout_seconds: args.embedding_timeout,
        embeddings: coheryn_api::embeddings::EmbeddingLimits {
            requests: args.embedding_requests,
            concurrency: args.embedding_concurrency,
            body_bytes: args.embedding_max_body_bytes,
            items: args.embedding_max_items,
            input_bytes: args.embedding_max_input_bytes,
            request_bytes: args.embedding_max_request_bytes,
            input_tokens: args.embedding_max_input_tokens,
            total_tokens: args.embedding_max_total_tokens,
            batch_size: args.embedding_max_batch_size,
            batch_tokens: args.embedding_max_batch_tokens,
            response_bytes: args.embedding_max_response_bytes,
        },
        ..Default::default()
    };
    let service = Service::open(config, credentials).map_err(|e| e.error)?;
    let listener = tokio::net::TcpListener::bind((args.host, args.port)).await?;
    eprintln!(
        "{}",
        json!({"event":"listening","address":listener.local_addr()?.to_string(),"credentials_file":auth::credential_path(&cli.data_dir),"stage":"experimental-G4-E2-S6"})
    );
    let shutdown = service.shutdown.clone();
    let signal = tokio::spawn(async move {
        #[cfg(unix)]
        {
            if let Ok(mut term) =
                tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            {
                tokio::select! {_=tokio::signal::ctrl_c()=>(),_=term.recv()=>()}
            } else {
                let _ = tokio::signal::ctrl_c().await;
            }
        }
        #[cfg(not(unix))]
        {
            let _ = tokio::signal::ctrl_c().await;
        }
        shutdown.cancel();
    });
    let result = coheryn_api::http::serve(listener, service.clone()).await;
    service.shutdown.cancel();
    signal.abort();
    result.map_err(|e| e.error)
}
fn token_file(path: &Path) -> Result<String> {
    let file = std::fs::File::open(path)?;
    if file.metadata()?.len() > 8192 {
        return Err(Error::request("token file exceeds 8192 bytes"));
    }
    let mut token = String::new();
    file.take(8193).read_to_string(&mut token)?;
    let token = token.trim().to_owned();
    if !(32..=4096).contains(&token.len())
        || !token.is_ascii()
        || token
            .bytes()
            .any(|b| b.is_ascii_whitespace() || b.is_ascii_control())
    {
        return Err(Error::request("invalid token file"));
    }
    Ok(token)
}
pub fn client(cli: &Cli) -> Result<Client> {
    // Never send the local credential vault to an arbitrary remote endpoint.
    let disk = if coheryn_api::client::loopback_endpoint(&cli.endpoint)
        && auth::credential_path(&cli.data_dir).exists()
    {
        Some(auth::read_credentials(&cli.data_dir)?)
    } else {
        None
    };
    let inference = if let Some(path) = &cli.api_token_file {
        Some(token_file(path)?)
    } else {
        std::env::var("COHERYN_API_TOKEN")
            .ok()
            .or_else(|| disk.as_ref().map(|c| c.inference_token.clone()))
    };
    let admin = if let Some(path) = &cli.admin_token_file {
        Some(token_file(path)?)
    } else {
        std::env::var("COHERYN_ADMIN_TOKEN")
            .ok()
            .or_else(|| disk.as_ref().map(|c| c.admin_token.clone()))
    };
    let inference=inference.or_else(||admin.clone()).ok_or_else(||Error::new(ErrorCode::Unauthorized,"no client credential; start serve with the same --data-dir or supply COHERYN_API_TOKEN/COHERYN_ADMIN_TOKEN"))?;
    Client::new(&cli.endpoint, inference, admin)
}
pub async fn run(cli: &Cli, args: &Run, cancel: &Cancellation) -> Result<()> {
    coheryn_engine::planner::Profile::parse(&args.profile)?;
    let options = GenerationOptions {
        max_tokens: args.max_tokens,
        ..args.controls.options()?
    };
    let messages = match &args.messages {
        Some(p) => Some(crate::read_messages(p)?),
        None => None,
    };
    let mut emit = |event: coheryn_engine::TokenEvent| -> Result<()> {
        if args.stream {
            let mut out = io::stdout().lock();
            write!(out, "{}", event.text)?;
            out.flush()?;
        }
        Ok(())
    };
    let result = if args.standalone {
        let model = Model::load(&args.model, args.resources.limits())?;
        let prompt = if let Some(messages) = messages {
            model.render(&messages)?
        } else {
            args.prompt.clone().expect("clap prompt")
        };
        generate_local(
            &model,
            &prompt,
            &options,
            (&args.profile, args.threads),
            cancel,
            &mut emit,
        )?
    } else {
        let id = args
            .model
            .to_str()
            .ok_or_else(|| Error::request("managed model ID must be UTF-8"))?
            .to_owned();
        coheryn_api::registry::validate_id(&id).map_err(|_| {
            Error::request(
                "client mode requires a managed model ID; use --standalone for a local GGUF path",
            )
        })?;
        let mut request = GenerationRequest::completion(
            id,
            args.prompt.clone().unwrap_or_default(),
            &options,
            args.resources.context,
            args.stream,
        );
        request.max_memory = Some(args.resources.max_memory);
        request.profile = args.profile.clone();
        request.threads = args.threads;
        if let Some(messages) = messages {
            request.prompt = None;
            request.messages = Some(messages);
        }
        client(cli)?.generate(&request, cancel, &mut emit).await?
    };
    if cli.json {
        print_json(&result)?;
    } else if !args.stream {
        writeln!(io::stdout().lock(), "{}", result.text)?;
    } else {
        writeln!(io::stdout().lock())?;
    }
    if result.finish_reason == coheryn_engine::FinishReason::Cancelled {
        return Err(Error::new(ErrorCode::Cancelled, "generation cancelled"));
    }
    Ok(())
}

#[derive(Args)]
pub struct ChatArgs {
    /// Attach a local PNG/JPEG. Repeat for multiple images; model support is required.
    #[arg(long = "image", conflicts_with_all = ["request"])]
    pub images: Vec<PathBuf>,
    /// One image-chat prompt; if omitted, read a bounded prompt from stdin.
    #[arg(long, requires = "images")]
    pub prompt: Option<String>,
    /// Local vision component for explicit standalone image chat.
    #[arg(long, requires_all = ["images", "standalone"])]
    pub vision_model: Option<PathBuf>,
    /// One headless chat request, including tools and prior tool results, as bounded JSON.
    #[arg(long,conflicts_with_all=["session","resume","persist","system","overflow","context","profile","threads","max_tokens","temperature","top_p","top_k","seed","stop","stop_tokens","repetition_penalty","frequency_penalty","presence_penalty","response_format","schema","timeout_seconds"])]
    pub request: Option<PathBuf>,
    /// Stream image text or request-file text/tool events; --json emits JSON Lines.
    #[arg(long)]
    pub stream: bool,
    pub model: PathBuf,
    #[arg(long)]
    pub session: Option<String>,
    #[arg(long, requires = "session", conflicts_with = "resume")]
    pub persist: bool,
    #[arg(long, requires = "session", conflicts_with = "context")]
    pub resume: bool,
    #[arg(long, value_parser = ["reject", "oldest-complete-turns"])]
    pub overflow: Option<String>,
    #[arg(long)]
    pub standalone: bool,
    #[arg(long)]
    pub system: Option<String>,
    #[arg(long, default_value = "balanced")]
    pub profile: String,
    #[arg(long)]
    pub threads: Option<usize>,
    #[arg(long, default_value_t = 128)]
    pub max_tokens: usize,
    #[command(flatten)]
    pub resources: Resources,
    #[command(flatten)]
    pub controls: crate::Controls,
}
pub async fn chat(cli: &Cli, args: &ChatArgs, cancel: &Cancellation) -> Result<()> {
    if !args.images.is_empty() {
        return crate::image_chat::run(cli, args, cancel).await;
    }
    if args.stream && args.request.is_none() {
        return Err(Error::request("--stream requires --image or --request"));
    }
    if args.request.is_some() {
        return crate::tool_chat::run(cli, args, cancel).await;
    }
    coheryn_engine::planner::Profile::parse(&args.profile)?;
    if cli.json {
        return Err(Error::request(
            "interactive chat does not support --json; use run --messages",
        ));
    }
    let managed = crate::sessions::interactive_session(cli, args, cancel).await?;
    let model = if args.standalone {
        Some(Arc::new(Model::load(&args.model, args.resources.limits())?))
    } else {
        None
    };
    let remote = if args.standalone {
        None
    } else {
        Some(client(cli)?)
    };
    let mut messages = Vec::new();
    if let Some(system) = &args.system {
        messages.push(Message {
            role: "system".into(),
            content: system.clone(),
            ..Default::default()
        });
    }
    let options = GenerationOptions {
        max_tokens: args.max_tokens,
        ..args.controls.options()?
    };
    // An ordinary dedicated reader avoids an uncancellable blocking stdin task
    // in Tokio's shutdown pool. The bounded channel prevents input read-ahead growth.
    let (sender, mut input_lines) = tokio::sync::mpsc::channel(1);
    std::thread::spawn(move || {
        let stdin = io::stdin();
        let mut stdin = stdin.lock();
        loop {
            let mut input = String::new();
            let next = match Read::by_ref(&mut stdin)
                .take((1 << 20) + 1)
                .read_line(&mut input)
            {
                Ok(0) => Ok(None),
                Ok(_) if input.len() > 1 << 20 => Err(Error::request("chat input exceeds one MiB")),
                Ok(_) => Ok(Some(input)),
                Err(e) => Err(Error::from(e)),
            };
            let terminal = !matches!(next, Ok(Some(_)));
            if sender.blocking_send(next).is_err() || terminal {
                break;
            }
        }
    });
    loop {
        cancel.check()?;
        write!(io::stdout().lock(), "You: ")?;
        io::stdout().flush()?;
        let input = tokio::select! {
            next=input_lines.recv()=>next.unwrap_or(Ok(None))?,
            _=async {while cancel.check().is_ok(){tokio::time::sleep(std::time::Duration::from_millis(25)).await;}}=>return Err(Error::new(ErrorCode::Cancelled,"chat cancelled")),
        };
        let Some(input) = input else {
            break;
        };
        if matches!(input.trim(), "/exit" | "/quit") {
            break;
        }
        let content = input.trim_end_matches(['\r', '\n']).to_owned();
        if managed.is_none() {
            messages.push(Message {
                role: "user".into(),
                content: content.clone(),
                ..Default::default()
            });
        }
        let mut emit = |event: coheryn_engine::TokenEvent| -> Result<()> {
            let mut out = io::stdout().lock();
            write!(out, "{}", event.text)?;
            out.flush()?;
            Ok(())
        };
        let result = if let Some(session) = &managed {
            let remote = remote.as_ref().expect("managed daemon client");
            let request = coheryn_api::sessions::SubmitTurn {
                images: vec![],
                expected_revision: crate::sessions::revision(remote, session, cancel).await?,
                idempotency_key: crate::sessions::fresh_key(),
                content,
                tool_results: vec![],
                settings: coheryn_api::sessions::TurnSettings {
                    max_tokens: args.max_tokens,
                    profile: args.profile.clone(),
                    threads: args.threads,
                    max_memory: Some(args.resources.max_memory),
                    ..args.controls.turn_settings(args.max_tokens)?
                },
                stream: true,
            };
            remote
                .session_generate(session, &request, cancel, &mut emit)
                .await?
        } else if let Some(model) = &model {
            generate_local(
                model,
                &model.render(&messages)?,
                &options,
                (&args.profile, args.threads),
                cancel,
                &mut emit,
            )?
        } else {
            let id = args
                .model
                .to_str()
                .ok_or_else(|| Error::request("managed model ID must be UTF-8"))?
                .to_owned();
            let mut request = GenerationRequest::completion(
                id,
                String::new(),
                &options,
                args.resources.context,
                true,
            );
            request.max_memory = Some(args.resources.max_memory);
            request.profile = args.profile.clone();
            request.threads = args.threads;
            request.prompt = None;
            request.messages = Some(messages.clone());
            remote
                .as_ref()
                .expect("client")
                .generate(&request, cancel, &mut emit)
                .await?
        };
        writeln!(io::stdout().lock())?;
        if managed.is_none() {
            messages.push(Message {
                role: "assistant".into(),
                content: result.text,
                ..Default::default()
            });
        }
    }
    Ok(())
}
#[derive(Args)]
pub struct PlanArgs {
    /// Inspect local attachment headers for task=image-to-text; bytes are not uploaded by plan.
    #[arg(long = "image")]
    pub images: Vec<PathBuf>,
    #[arg(long)]
    pub constraints: bool,
    pub model: String,
    #[arg(long, default_value = "text-generation")]
    pub task: String,
    #[arg(long)]
    pub context: Option<usize>,
    #[arg(long, default_value = "balanced")]
    pub profile: String,
    #[arg(long)]
    pub threads: Option<usize>,
    #[arg(long, default_value_t = 1)]
    pub concurrency: usize,
    #[arg(long, value_delimiter = ',')]
    pub input_bytes: Option<Vec<usize>>,
    #[arg(long)]
    pub batch_size: Option<usize>,
    #[arg(long)]
    pub max_input_tokens: Option<usize>,
    #[arg(long)]
    pub max_batch_tokens: Option<usize>,
}
pub async fn plan(cli: &Cli, args: &PlanArgs, cancel: &Cancellation) -> Result<()> {
    let context = args
        .context
        .unwrap_or(if args.task == "embedding" { 512 } else { 4096 });
    let images = if args.images.is_empty() {
        None
    } else {
        if args.task != "image-to-text" || args.images.len() > 4 {
            return Err(Error::request(
                "--image planning requires task=image-to-text and at most four images",
            ));
        }
        let limits = coheryn_engine::multimodal::image::ImageLimits::default();
        let mut images = Vec::new();
        let mut total = 0;
        for path in &args.images {
            let meta = std::fs::symlink_metadata(path)?;
            if !meta.is_file()
                || meta.file_type().is_symlink()
                || meta.len() > limits.max_encoded_bytes
            {
                return Err(Error::request(
                    "planning images must be regular bounded files",
                ));
            }
            let mut bytes = Vec::new();
            std::fs::File::open(path)?
                .take(limits.max_encoded_bytes + 1)
                .read_to_end(&mut bytes)?;
            total += bytes.len() as u64;
            if total > limits.max_total_encoded_bytes {
                return Err(Error::resource(
                    "planning attachments exceed total byte limit",
                ));
            }
            let info = coheryn_engine::multimodal::image::inspect(&bytes, limits, cancel)?;
            let (width, height) = info.oriented_dimensions();
            images.push(coheryn_engine::planner::ImageDescriptor { width, height });
        }
        Some(coheryn_engine::planner::ImageWorkload {
            images,
            processor_profile: "qwen2vl-still-v1".into(),
        })
    };
    crate::print_json(&client(cli)?.json("POST","/api/v1/plans/estimate",Some(json!({"model":args.model,"constraints":args.constraints,"task":args.task,"context":context,"profile":args.profile,"concurrency":args.concurrency,"threads":args.threads,"images":images,"input_bytes":args.input_bytes,"batch_size":args.batch_size,"max_input_tokens":args.max_input_tokens,"max_batch_tokens":args.max_batch_tokens})),true,cancel).await?)
}
#[derive(Args)]
pub struct BenchResources {
    /// Defaults to 4096 for generation and 512 for embedding workloads.
    #[arg(long)]
    pub context: Option<usize>,
    #[arg(long, default_value_t = 1073741824)]
    pub max_memory: u64,
}
impl BenchResources {
    fn limits(&self) -> coheryn_engine::ResourceLimits {
        coheryn_engine::ResourceLimits {
            context: self.context.unwrap_or(4096),
            max_memory: self.max_memory,
        }
    }
}
#[derive(Args)]
pub struct BenchArgs {
    #[arg(long, default_value = "text-generation")]
    pub task: String,
    #[arg(long, default_value_t = 1)]
    pub batch_items: usize,
    #[arg(long, default_value_t = 8)]
    pub batch_size: usize,
    #[arg(long)]
    pub model: PathBuf,
    #[arg(long)]
    pub standalone: bool,
    #[arg(long, default_value = "Hello")]
    pub prompt: String,
    #[arg(long, default_value_t = 8)]
    pub max_tokens: usize,
    #[arg(long, default_value_t = 1)]
    pub iterations: usize,
    #[arg(long, default_value = "reference")]
    pub profile: String,
    #[arg(long)]
    pub threads: Option<usize>,
    #[command(flatten)]
    pub resources: BenchResources,
}
pub async fn bench(cli: &Cli, args: &BenchArgs, cancel: &Cancellation) -> Result<()> {
    if args.task == "embedding" {
        return crate::embedding::bench(cli, args, cancel).await;
    }
    if args.task != "text-generation" {
        return Err(Error::request("unsupported benchmark task"));
    }
    coheryn_engine::planner::Profile::parse(&args.profile)?;
    if !(1..=10).contains(&args.iterations) {
        return Err(Error::request("benchmarks support 1..=10 iterations"));
    }
    let options = GenerationOptions {
        max_tokens: args.max_tokens,
        ..Default::default()
    };
    let loading = std::time::Instant::now();
    let model = if args.standalone {
        Some(Model::load(&args.model, args.resources.limits())?)
    } else {
        None
    };
    let load_ms = model
        .as_ref()
        .map(|_| loading.elapsed().as_secs_f64() * 1000.);
    let remote = if args.standalone {
        None
    } else {
        Some(client(cli)?)
    };
    let mut runs: Vec<GenerationResult> = Vec::new();
    for _ in 0..args.iterations {
        let result = if let Some(model) = &model {
            generate_local(
                model,
                &args.prompt,
                &options,
                (&args.profile, args.threads),
                cancel,
                |_| Ok(()),
            )?
        } else {
            let id = args
                .model
                .to_str()
                .ok_or_else(|| Error::request("model ID must be UTF-8"))?
                .to_owned();
            let mut request = GenerationRequest::completion(
                id,
                args.prompt.clone(),
                &options,
                args.resources.context.unwrap_or(4096),
                false,
            );
            request.max_memory = Some(args.resources.max_memory);
            request.profile = args.profile.clone();
            request.threads = args.threads;
            remote
                .as_ref()
                .expect("client")
                .generate(&request, cancel, |_| Ok(()))
                .await?
        };
        runs.push(result);
    }
    print_json(
        &json!({"backend":"cpu-reference","mode":if args.standalone{"standalone"}else{"daemon-client"},"profile":args.profile,"requested_threads":args.threads,"load_ms":load_ms,"cache_condition":"uncontrolled","runs":runs}),
    )
}

pub(crate) fn generate_local(
    model: &Model,
    prompt: &str,
    options: &GenerationOptions,
    settings: (&str, Option<usize>),
    cancel: &Cancellation,
    emit: impl FnMut(coheryn_engine::TokenEvent) -> Result<()>,
) -> Result<GenerationResult> {
    use coheryn_engine::planner::{self, PlanInput, Profile};
    let plan = planner::estimate(PlanInput {
        constraints: options.response_format.enabled(),
        descriptor: &model.descriptor,
        config: &model.config,
        model: "standalone",
        profile: Profile::parse(settings.0)?,
        context: model.context_limit(),
        concurrency: 1,
        max_concurrency: 1,
        loaded: true,
        budget: model.memory(),
        hardware: coheryn_api::hardware::probe(),
        images: None,
        threads: settings.1,
    })?;
    plan.require_admission()?;
    let mut admission = model.reserve_session_with_execution(
        model.context_limit(),
        plan.additional_reserved_bytes - plan.session_reserved_bytes,
        plan.io_buffer_bytes,
        coheryn_engine::inference::ExecutionSettings {
            threads: plan.threads_per_session,
            prefill_chunk_tokens: plan.prefill_chunk_tokens,
        },
    )?;
    if plan.constraint_reserved_bytes > 0 {
        admission.constraints = Some(
            admission
                .reservation
                .split(plan.constraint_reserved_bytes)?,
        );
    }
    model.generate_admitted(prompt, options, admission, cancel, emit)
}
