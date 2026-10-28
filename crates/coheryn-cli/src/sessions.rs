use crate::{Cli, daemon, print_json};
use clap::{Args, Subcommand};
use coheryn_api::sessions::{SubmitTurn, TurnSettings};
use coheryn_engine::{Cancellation, Error, Result};
use serde_json::{Value, json};

#[derive(Args)]
pub struct SessionArgs {
    #[command(subcommand)]
    command: SessionCommand,
}
#[derive(Subcommand)]
enum SessionCommand {
    /// Submit structured user/tool-result input and settings from a bounded JSON file.
    Submit {
        session: String,
        #[arg(long)]
        request: std::path::PathBuf,
        #[arg(long)]
        stream: bool,
    },
    Create {
        model: String,
        #[arg(long)]
        persist: bool,
        #[arg(long)]
        name: Option<String>,
        #[arg(long, default_value_t = 4096)]
        context: usize,
        #[arg(long)]
        system: Option<String>,
        #[arg(long, default_value = "reject", value_parser = ["reject", "oldest-complete-turns"])]
        overflow: String,
    },
    List,
    Save {
        session: String,
        #[arg(long)]
        expected_revision: u64,
    },
    Export {
        session: String,
        #[arg(long)]
        output: std::path::PathBuf,
        #[arg(long)]
        overwrite: bool,
    },
    Import {
        file: std::path::PathBuf,
        #[arg(long)]
        name: Option<String>,
        #[arg(long)]
        persist: bool,
    },
    Show {
        session: String,
        #[arg(long, default_value_t = 0)]
        offset: usize,
        #[arg(long, default_value_t = 100)]
        limit: usize,
    },
    Reset {
        session: String,
        #[arg(long)]
        expected_revision: u64,
        #[arg(long)]
        clear_instructions: bool,
    },
    Delete {
        session: String,
        #[arg(long)]
        expected_revision: u64,
    },
    /// Submit one revision-checked turn and print the complete JSON receipt.
    Send {
        session: String,
        #[arg(long)]
        input: String,
        #[arg(long)]
        expected_revision: u64,
        #[arg(long)]
        idempotency_key: Option<String>,
        #[arg(long, default_value_t = 128)]
        max_tokens: usize,
        #[command(flatten)]
        controls: crate::Controls,
    },
}
fn route(session: &str, suffix: &str) -> Result<String> {
    if session.is_empty()
        || session.len() > 64
        || !session
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
    {
        return Err(Error::request("invalid session ID or alias"));
    }
    Ok(format!("/coheryn/v1/sessions/{session}{suffix}"))
}
pub async fn run(cli: &Cli, args: &SessionArgs, cancel: &Cancellation) -> Result<()> {
    let client = daemon::client(cli)?;
    if let SessionCommand::Submit {
        session,
        request,
        stream,
    } = &args.command
    {
        let mut body: SubmitTurn = crate::tool_chat::read_json(request, 1 << 20)?;
        if *stream {
            body.stream = true;
        }
        if body.stream {
            let result = client
                .session_generate_with_tools(
                    session,
                    &body,
                    cancel,
                    |token| print_json(&json!({"token":token})),
                    |delta| print_json(&json!({"tool_call":delta})),
                )
                .await?;
            return print_json(&json!({"result":result}));
        }
        return print_json(
            &client
                .json(
                    "POST",
                    &route(session, "/messages")?,
                    Some(
                        serde_json::to_value(body)
                            .map_err(|_| Error::request("invalid session request"))?,
                    ),
                    false,
                    cancel,
                )
                .await?,
        );
    }
    if let SessionCommand::Export {
        session,
        output,
        overwrite,
    } = &args.command
    {
        let document = client
            .json("GET", &route(session, "/export")?, None, false, cancel)
            .await?;
        if output.exists() && !overwrite {
            return Err(Error::new(
                coheryn_engine::ErrorCode::Conflict,
                "output exists; use --overwrite explicitly",
            ));
        }
        let parent = output
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(std::path::Path::new("."));
        let mut file = tempfile::Builder::new()
            .prefix(".coheryn-history-")
            .suffix(".part")
            .tempfile_in(parent)?;
        serde_json::to_writer(&mut file, &document)
            .map_err(|_| Error::request("invalid exported history"))?;
        file.as_file().sync_all()?;
        if *overwrite {
            file.persist(output).map_err(|e| Error::from(e.error))?;
        } else {
            file.persist_noclobber(output)
                .map_err(|e| Error::from(e.error))?;
        }
        return print_json(&json!({"output":output,"session_id":document["session"]["id"]}));
    }
    if let SessionCommand::Import {
        file,
        name,
        persist,
    } = &args.command
    {
        use std::io::Read;
        let input = std::fs::File::open(file)?;
        if input.metadata()?.len() > coheryn_api::session_store::TRANSFER_BYTES as u64 {
            return Err(Error::resource("history import file exceeds limit"));
        }
        let mut data = Vec::new();
        input
            .take((coheryn_api::session_store::TRANSFER_BYTES + 1) as u64)
            .read_to_end(&mut data)?;
        if data.len() > coheryn_api::session_store::TRANSFER_BYTES {
            return Err(Error::resource("history import file grew beyond limit"));
        }
        let document: serde_json::Value = serde_json::from_slice(&data)
            .map_err(|_| Error::request("invalid history import JSON"))?;
        return print_json(
            &client
                .json(
                    "POST",
                    "/coheryn/v1/sessions/import",
                    Some(json!({"document":document,"name":name,"persist":persist})),
                    false,
                    cancel,
                )
                .await?,
        );
    }
    let (method, path, body) = match &args.command {
        SessionCommand::Create {
            model,
            persist,
            name,
            context,
            system,
            overflow,
        } => (
            "POST",
            "/coheryn/v1/sessions".into(),
            Some(
                json!({"model":model,"persist":persist,"alias":name,"context":context,"system":system,"overflow":overflow}),
            ),
        ),
        SessionCommand::Submit { .. }
        | SessionCommand::Export { .. }
        | SessionCommand::Import { .. } => {
            unreachable!("transfers handled above")
        }
        SessionCommand::Save {
            session,
            expected_revision,
        } => (
            "POST",
            route(session, "/save")?,
            Some(json!({"expected_revision":expected_revision})),
        ),
        SessionCommand::List => ("GET", "/coheryn/v1/sessions".into(), None),
        SessionCommand::Show {
            session,
            offset,
            limit,
        } => (
            "GET",
            format!("{}?offset={offset}&limit={limit}", route(session, "")?),
            None,
        ),
        SessionCommand::Reset {
            session,
            expected_revision,
            clear_instructions,
        } => (
            "POST",
            route(session, "/reset")?,
            Some(
                json!({"expected_revision":expected_revision,"clear_instructions":clear_instructions}),
            ),
        ),
        SessionCommand::Delete {
            session,
            expected_revision,
        } => (
            "DELETE",
            route(session, "")?,
            Some(json!({"expected_revision":expected_revision})),
        ),
        SessionCommand::Send {
            session,
            input,
            expected_revision,
            idempotency_key,
            max_tokens,
            controls,
        } => {
            let request = SubmitTurn {
                expected_revision: *expected_revision,
                idempotency_key: idempotency_key.clone().unwrap_or_else(fresh_key),
                content: input.clone(),
                tool_results: vec![],
                settings: TurnSettings {
                    max_tokens: *max_tokens,
                    ..controls.turn_settings(*max_tokens)?
                },
                stream: false,
            };
            (
                "POST",
                route(session, "/messages")?,
                Some(
                    serde_json::to_value(request)
                        .map_err(|_| Error::request("invalid managed turn"))?,
                ),
            )
        }
    };
    print_json(&client.json(method, &path, body, false, cancel).await?)
}
pub fn fresh_key() -> String {
    coheryn_api::sessions::idempotency_key()
}
pub async fn interactive_session(
    cli: &Cli,
    args: &daemon::ChatArgs,
    cancel: &Cancellation,
) -> Result<Option<String>> {
    let Some(alias) = &args.session else {
        if args.resume || args.overflow.is_some() {
            return Err(Error::request("--resume and --overflow require --session"));
        }
        return Ok(None);
    };
    let path = route(alias, "")?;
    if args.standalone {
        return Err(Error::request(
            "named managed sessions require the daemon; ordinary standalone chat remains available",
        ));
    }
    let model = args
        .model
        .to_str()
        .ok_or_else(|| Error::request("managed model ID must be UTF-8"))?;
    let client = daemon::client(cli)?;
    let view = if args.resume {
        if args.system.is_some() || args.overflow.is_some() {
            return Err(Error::request(
                "resume uses stored instructions and context policy; do not supply --system or --overflow",
            ));
        }
        client.json("GET", &path, None, false, cancel).await?
    } else {
        client.json("POST", "/coheryn/v1/sessions", Some(json!({"model":model,"persist":args.persist,"alias":alias,"context":args.resources.context,"system":args.system,"overflow":args.overflow.as_deref().unwrap_or("reject")})), false, cancel).await?
    };
    if view["model"].as_str() != Some(model) {
        return Err(Error::request("session is bound to a different model"));
    }
    view["id"]
        .as_str()
        .map(|id| Some(id.to_owned()))
        .ok_or_else(|| Error::request("invalid session response"))
}
pub async fn revision(
    client: &coheryn_api::client::Client,
    session: &str,
    cancel: &Cancellation,
) -> Result<u64> {
    let value: Value = client
        .json("GET", &route(session, "")?, None, false, cancel)
        .await?;
    value["revision"]
        .as_u64()
        .ok_or_else(|| Error::request("invalid session revision"))
}
