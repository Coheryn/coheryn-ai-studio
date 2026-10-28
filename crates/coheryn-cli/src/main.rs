mod daemon;
mod embedding;
mod image_chat;
mod sessions;
use clap::{Args, Parser, Subcommand};
use coheryn_engine::GenerationOptions;
use coheryn_engine::{
    Cancellation, Error, ErrorCode, Model, ParseLimits, ResourceLimits, Result, inspect,
    tokenizer::{Message, TextTokenizer},
    validate,
};
use std::{
    io::{self, Read, Write},
    path::PathBuf,
    process::ExitCode,
};

#[derive(Parser)]
#[command(
    name = "coheryn",
    version,
    about = "Independent Rust GGUF CPU engine — experimental G4/E2 CPU development build"
)]
struct Cli {
    #[arg(long, global = true)]
    json: bool,
    #[arg(long, global = true, default_value = "http://127.0.0.1:11450")]
    endpoint: String,
    #[arg(long,global=true,default_value_os_t=coheryn_api::config::default_data_dir())]
    data_dir: PathBuf,
    #[arg(long, global = true)]
    api_token_file: Option<PathBuf>,
    #[arg(long, global = true)]
    admin_token_file: Option<PathBuf>,
    #[command(subcommand)]
    command: Command,
}
#[derive(Subcommand)]
enum Command {
    Models {
        #[command(subcommand)]
        command: Models,
    },
    Run(Run),
    Embed(embedding::EmbedArgs),
    Serve(daemon::ServeArgs),
    Chat(daemon::ChatArgs),
    Sessions(sessions::SessionArgs),
    Status,
    Metrics,
    Plan(daemon::PlanArgs),
    Bench(daemon::BenchArgs),
    /// Print build capabilities without acquiring a model.
    Doctor,
    /// Tokenizer/template diagnostic for reference fixture capture.
    Tokenize {
        model: PathBuf,
        #[arg(long)]
        prompt: Option<String>,
        #[arg(long)]
        messages: Option<PathBuf>,
    },
    /// Teacher-forced logits and layer activations as JSON Lines.
    Trace {
        model: PathBuf,
        #[arg(long)]
        prompt: Option<String>,
        #[arg(long, value_delimiter = ',')]
        tokens: Vec<u32>,
        #[command(flatten)]
        resources: Resources,
    },
}
#[derive(Subcommand)]
enum Models {
    List,
    #[command(alias = "register")]
    Import {
        path: PathBuf,
        #[arg(long)]
        name: String,
        #[arg(long)]
        sha256: Option<String>,
    },
    Pull {
        url: String,
        #[arg(long)]
        name: String,
        #[arg(long)]
        sha256: String,
    },
    /// Register a managed language/vision component pair without copying weights.
    RegisterPair {
        #[arg(long)]
        name: String,
        #[arg(long)]
        language: String,
        #[arg(long)]
        vision: String,
    },
    Load {
        model: String,
        #[arg(long, default_value_t = 4096)]
        context: usize,
    },
    Unload {
        model: String,
    },
    Remove {
        model: String,
    },
    Inspect {
        model: PathBuf,
    },
    Validate {
        model: PathBuf,
        #[arg(long)]
        deep: bool,
    },
}
#[derive(Args)]
struct Resources {
    #[arg(long, default_value_t = 4096)]
    context: usize,
    /// Engine admission budget in bytes; does not cap OS resident memory.
    #[arg(long, default_value_t = 1073741824)]
    max_memory: u64,
}
impl Resources {
    fn limits(&self) -> ResourceLimits {
        ResourceLimits {
            context: self.context,
            max_memory: self.max_memory,
        }
    }
}
#[derive(Args)]
struct Run {
    model: PathBuf,
    #[arg(
        long,
        required_unless_present = "messages",
        conflicts_with = "messages"
    )]
    prompt: Option<String>,
    /// JSON array of system/user/assistant text messages.
    #[arg(long)]
    messages: Option<PathBuf>,
    /// Load a separate process-local model instance without HTTP.
    #[arg(long)]
    standalone: bool,
    #[arg(long, default_value = "balanced")]
    profile: String,
    #[arg(long)]
    threads: Option<usize>,
    #[arg(long, conflicts_with = "json")]
    stream: bool,
    #[arg(long, default_value_t = 128)]
    max_tokens: usize,
    #[command(flatten)]
    controls: Controls,
    #[command(flatten)]
    resources: Resources,
}
#[derive(Args)]
pub(crate) struct Controls {
    #[arg(long, default_value_t = 0.0)]
    temperature: f32,
    #[arg(long, default_value_t = 0)]
    top_k: usize,
    #[arg(long, default_value_t = 1.0)]
    top_p: f32,
    #[arg(long, default_value_t = 0)]
    seed: u64,
    #[arg(long)]
    stop: Vec<String>,
    #[arg(long = "stop-token")]
    stop_tokens: Vec<u32>,
    #[arg(long, default_value_t = 1.0)]
    repetition_penalty: f32,
    #[arg(long, default_value_t = 0.0, allow_hyphen_values = true)]
    frequency_penalty: f32,
    #[arg(long, default_value_t = 0.0, allow_hyphen_values = true)]
    presence_penalty: f32,
    #[arg(long, default_value = "text", value_parser = ["text", "json_object", "json_schema"])]
    response_format: String,
    #[arg(long)]
    schema: Option<PathBuf>,
    #[arg(long, value_parser = clap::value_parser!(u64).range(1..=86400))]
    timeout_seconds: Option<u64>,
}
impl Controls {
    fn options(&self) -> Result<GenerationOptions> {
        use coheryn_engine::constraints::{ResponseFormat, SchemaRequest};
        let format = match (self.response_format.as_str(), &self.schema) {
            ("text", None) => ResponseFormat::Text,
            ("json_object", None) => ResponseFormat::JsonObject,
            ("json_schema", Some(path)) => ResponseFormat::JsonSchema {
                json_schema: SchemaRequest {
                    name: "result".into(),
                    strict: true,
                    schema: crate::tool_chat::read_json(
                        path,
                        coheryn_engine::constraints::MAX_SCHEMA_BYTES,
                    )?,
                },
            },
            _ => {
                return Err(Error::request(
                    "--schema requires --response-format json_schema, and json_schema requires --schema",
                ));
            }
        };
        format.compile()?;
        let penalties = coheryn_engine::sampling::Penalties {
            repetition_penalty: self.repetition_penalty,
            frequency_penalty: self.frequency_penalty,
            presence_penalty: self.presence_penalty,
        };
        penalties.validate()?;
        coheryn_engine::sampling::Sampler::with_penalties(
            coheryn_engine::sampling::Sampling {
                temperature: self.temperature,
                top_k: self.top_k,
                top_p: self.top_p,
                seed: self.seed,
            },
            penalties,
        )?;
        if self.stop.len() > 16
            || self.stop.iter().any(|s| s.is_empty() || s.len() > 4096)
            || self.stop_tokens.len() > 16
        {
            return Err(Error::request("custom stop limits exceeded"));
        }
        if format.enabled() && (!self.stop.is_empty() || !self.stop_tokens.is_empty()) {
            return Err(Error::request(
                "constrained output does not support custom stops",
            ));
        }
        Ok(GenerationOptions {
            timeout_seconds: self.timeout_seconds,
            sampling: coheryn_engine::sampling::Sampling {
                temperature: self.temperature,
                top_k: self.top_k,
                top_p: self.top_p,
                seed: self.seed,
            },
            penalties,
            stop: self.stop.clone(),
            stop_tokens: self.stop_tokens.clone(),
            response_format: format,
            ..Default::default()
        })
    }
    fn turn_settings(&self, max_tokens: usize) -> Result<coheryn_api::sessions::TurnSettings> {
        let options = self.options()?;
        Ok(coheryn_api::sessions::TurnSettings {
            timeout_seconds: self.timeout_seconds,
            max_tokens,
            temperature: self.temperature,
            top_k: self.top_k,
            top_p: self.top_p,
            seed: self.seed,
            repetition_penalty: self.repetition_penalty,
            frequency_penalty: self.frequency_penalty,
            presence_penalty: self.presence_penalty,
            stop: options.stop,
            stop_tokens: options.stop_tokens,
            response_format: options.response_format,
            ..Default::default()
        })
    }
}
fn print_json(value: &impl serde::Serialize) -> Result<()> {
    let mut output = io::stdout().lock();
    serde_json::to_writer(&mut output, value)
        .map_err(|e| Error::new(ErrorCode::Io, e.to_string()))?;
    writeln!(output)?;
    Ok(())
}
fn read_messages(path: &PathBuf) -> Result<Vec<Message>> {
    let file = std::fs::File::open(path)?;
    if file.metadata()?.len() > 1 << 20 {
        return Err(Error::request("messages file exceeds one MiB"));
    }
    serde_json::from_reader(file.take((1 << 20) + 1))
        .map_err(|e| Error::request(format!("invalid messages JSON: {e}")))
}
fn local_path(path: &std::path::Path) -> bool {
    path.exists() || path.to_string_lossy().contains(['/', '\\'])
}
async fn execute(cli: &Cli) -> Result<()> {
    if let Command::Serve(args) = &cli.command {
        return daemon::serve(cli, args).await;
    }
    let cancel = Cancellation::default();
    let handler = cancel.clone();
    ctrlc::set_handler(move || handler.cancel())
        .map_err(|e| Error::new(ErrorCode::Internal, e.to_string()))?;
    match &cli.command {
        Command::Doctor => {
            let mut capabilities = serde_json::json!({"version":env!("CARGO_PKG_VERSION"),"os":std::env::consts::OS,"architecture":std::env::consts::ARCH,"backend":"cpu-reference","default_thread_policy":"profile-and-cpu-share","max_threads_per_session":16,"execution_encodings":["F32","F16","Q4_0","Q5_0","Q8_0","Q4_K","Q6_K"],"qualification":"experimental-exact-file-evidence-in-fixtures/compatibility.json","vision":true,"vision_profile":"qwen2vl-still-f16-cpu-greedy","vision_backends":["cpu-reference"],"vision_api":true,"image_formats":["png","jpeg"],"vision_max_images":4,"embeddings":true,"embedding_cli_api":true,"embedding_architectures":["nomic-bert"],"embedding_encodings":["F32","F16","Q8_0"],"embedding_max_input_tokens":512,"embedding_max_items":256,"embedding_api_encodings":["float","base64"],"qualified_embedding_dimensions":[768],"http_api":true,"managed_sessions":true,"session_storage":"opt-in-embedded","persistent_sessions":true,"session_kv_reuse":true,"tool_calling":true,"tool_profiles":["qwen-tools-v1"],"strict_tool_arguments":true,"schema_constraints":true,"constraint_profile":"coheryn-json-v1","strict_tool_mode":"single-required-or-named-no-parallel","sampling_profile":"coheryn-sampling-v2","continuous_batching":true,"batching_profile":"coheryn-cpu-batch-v1","generation_quantum_tokens":1,"metal":false,"telemetry":false});
            capabilities["managed_image_sessions"] = true.into();
            capabilities["image_asset_storage"] = "private-original-bytes".into();
            capabilities["image_scheduler"] = "coheryn-vlm-cooperative-v1".into();
            capabilities["max_session_image_assets"] = 16.into();
            capabilities["image_prefix_reuse"] = false.into();
            print_json(&capabilities)
        }
        Command::Models { command } => match command {
            Models::List => print_json(
                &daemon::client(cli)?
                    .json("GET", "/api/v1/models", None, true, &cancel)
                    .await?,
            ),
            Models::Import { path, name, sha256 } => print_json(
                &daemon::client(cli)?
                    .import(path, name, sha256.as_deref(), &cancel)
                    .await?,
            ),
            Models::Pull { url, name, sha256 } => print_json(
                &daemon::client(cli)?
                    .json(
                        "POST",
                        "/api/v1/models/pull",
                        Some(serde_json::json!({"url":url,"name":name,"sha256":sha256})),
                        true,
                        &cancel,
                    )
                    .await?,
            ),
            Models::RegisterPair {
                name,
                language,
                vision,
            } => print_json(
                &daemon::client(cli)?
                    .json(
                        "POST",
                        "/api/v1/models/register-pair",
                        Some(serde_json::json!({"model":name,"language":language,"vision":vision})),
                        true,
                        &cancel,
                    )
                    .await?,
            ),
            Models::Load { model, context } => print_json(
                &daemon::client(cli)?
                    .json(
                        "POST",
                        "/api/v1/models/load",
                        Some(serde_json::json!({"model":model,"context":context})),
                        true,
                        &cancel,
                    )
                    .await?,
            ),
            Models::Unload { model } => print_json(
                &daemon::client(cli)?
                    .json(
                        "POST",
                        "/api/v1/models/unload",
                        Some(serde_json::json!({"model":model})),
                        true,
                        &cancel,
                    )
                    .await?,
            ),
            Models::Remove { model } => {
                coheryn_api::registry::validate_id(model).map_err(|e| e.error)?;
                print_json(
                    &daemon::client(cli)?
                        .json(
                            "DELETE",
                            &format!("/api/v1/models/{model}"),
                            None,
                            true,
                            &cancel,
                        )
                        .await?,
                )
            }
            Models::Inspect { model } => {
                if !local_path(model) {
                    let id = model
                        .to_str()
                        .ok_or_else(|| Error::request("model ID must be UTF-8"))?;
                    coheryn_api::registry::validate_id(id).map_err(|e| e.error)?;
                    return print_json(
                        &daemon::client(cli)?
                            .json(
                                "GET",
                                &format!("/api/v1/models/{id}/inspect"),
                                None,
                                true,
                                &cancel,
                            )
                            .await?,
                    );
                }
                let d = inspect(model, ParseLimits::default())?;
                let report = validate(&d);
                #[derive(serde::Serialize)]
                struct Inspection<'a> {
                    #[serde(flatten)]
                    descriptor: &'a coheryn_engine::Descriptor,
                    task: Option<coheryn_engine::ModelTask>,
                    embedding_config: Option<coheryn_engine::embedding::EmbeddingConfig>,
                }
                print_json(&Inspection {
                    descriptor: &d,
                    task: report.task,
                    embedding_config: report.embedding_config,
                })
            }
            Models::Validate { model, deep } => {
                if !local_path(model) {
                    return print_json(
                        &daemon::client(cli)?
                            .json(
                                "POST",
                                "/api/v1/models/validate",
                                Some(serde_json::json!({"model":model,"deep":deep})),
                                true,
                                &cancel,
                            )
                            .await?,
                    );
                }
                let d = inspect(model, ParseLimits::default())?;
                let mut report = validate(&d);
                if *deep && report.executable {
                    match coheryn_engine::validate_tokenizer(&d) {
                        Ok(_) => (),
                        Err(e) => {
                            report.executable = false;
                            report.diagnostics.push(e);
                        }
                    }
                    if report.executable
                        && let Err(e) = coheryn_engine::deep_validate(model, &d, &cancel)
                    {
                        report.executable = false;
                        report.diagnostics.push(e);
                    }
                }
                print_json(&report)?;
                if !report.executable {
                    return Err(report
                        .diagnostics
                        .first()
                        .cloned()
                        .unwrap_or_else(|| Error::invalid("model not executable")));
                }
                Ok(())
            }
        },
        Command::Tokenize {
            model,
            prompt,
            messages,
        } => {
            let d = inspect(model, ParseLimits::default())?;
            let tokenizer = TextTokenizer::from_descriptor(&d)?;
            let text = match messages {
                Some(p) => tokenizer.render(&read_messages(p)?)?,
                None => prompt
                    .clone()
                    .ok_or_else(|| Error::request("supply --prompt or --messages"))?,
            };
            print_json(&serde_json::json!({"prompt":text,"token_ids":tokenizer.encode(&text)?}))
        }
        Command::Trace {
            model,
            prompt,
            tokens,
            resources,
        } => {
            let m = Model::load(model, resources.limits())?;
            let ids = if tokens.is_empty() {
                m.tokenizer.encode(
                    prompt
                        .as_deref()
                        .ok_or_else(|| Error::request("supply --tokens or --prompt"))?,
                )?
            } else {
                tokens.clone()
            };
            m.trace(&ids, &cancel, &mut |position, name, values| {
                print_json(&serde_json::json!({"position":position,"name":name,"values":values}))
            })
        }
        Command::Run(args) => daemon::run(cli, args, &cancel).await,
        Command::Embed(args) => embedding::run(cli, args, &cancel).await,
        Command::Chat(args) => daemon::chat(cli, args, &cancel).await,
        Command::Sessions(args) => sessions::run(cli, args, &cancel).await,
        Command::Bench(args) => daemon::bench(cli, args, &cancel).await,
        Command::Serve(_) => unreachable!("serve handled before signal setup"),
        Command::Status | Command::Metrics => {
            let path = if matches!(cli.command, Command::Metrics) {
                "/api/v1/server/metrics"
            } else {
                "/api/v1/server/status"
            };
            print_json(
                &daemon::client(cli)?
                    .json("GET", path, None, true, &cancel)
                    .await?,
            )
        }
        Command::Plan(args) => daemon::plan(cli, args, &cancel).await,
    }
}
#[tokio::main(flavor = "multi_thread", worker_threads = 2)]
async fn main() -> ExitCode {
    let cli = match Cli::try_parse() {
        Ok(cli) => cli,
        Err(e) => {
            let display = matches!(
                e.kind(),
                clap::error::ErrorKind::DisplayHelp | clap::error::ErrorKind::DisplayVersion
            );
            if !display && std::env::args_os().any(|arg| arg == "--json") {
                let error = Error::request(e.to_string());
                eprintln!("{}", serde_json::to_string(&error).unwrap());
                return ExitCode::from(2);
            }
            e.exit()
        }
    };
    match execute(&cli).await {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            if cli.json {
                eprintln!("{}", serde_json::to_string(&e).unwrap());
            } else {
                eprintln!("{e}");
            }
            ExitCode::from(match e.code {
                ErrorCode::InvalidRequest
                | ErrorCode::InvalidImage
                | ErrorCode::ContextLimitExceeded
                | ErrorCode::ImageInputLimitExceeded
                | ErrorCode::EmbeddingInputLimitExceeded
                | ErrorCode::EmbeddingBatchLimitExceeded => 2,
                ErrorCode::Io | ErrorCode::ModelChanged => 3,
                ErrorCode::ResourceExhausted => 5,
                ErrorCode::Cancelled => 130,
                ErrorCode::DeadlineExceeded => 124,
                ErrorCode::Unauthorized => 6,
                ErrorCode::Conflict | ErrorCode::QueueFull => 7,
                ErrorCode::Unavailable => 8,
                ErrorCode::ConstraintDeadEnd
                | ErrorCode::ConstraintIncomplete
                | ErrorCode::ConstraintViolation => 9,
                ErrorCode::NotFound => 3,
                ErrorCode::Internal => 1,
                _ => 4,
            })
        }
    }
}

mod tool_chat;
