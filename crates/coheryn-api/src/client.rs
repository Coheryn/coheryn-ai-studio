use crate::{auth::Credentials, wire::GenerationRequest};
use coheryn_engine::{Cancellation, Error, ErrorCode, GenerationResult, Result, TokenEvent};
use futures_util::StreamExt;
use serde_json::Value;
use std::{path::Path, time::Duration};
use tokio_util::io::ReaderStream;

pub struct Client {
    http: reqwest::Client,
    endpoint: reqwest::Url,
    inference: String,
    admin: Option<String>,
}
pub struct ClientEmbeddings {
    pub response: crate::embeddings::EmbeddingResponse,
    pub space_id: String,
    pub model_sha256: String,
    pub dimension: usize,
    pub token_counts: Vec<usize>,
    pub truncated: Vec<usize>,
    pub batch_size: usize,
    pub token_capacity: usize,
}
pub fn loopback_endpoint(endpoint: &str) -> bool {
    reqwest::Url::parse(endpoint)
        .ok()
        .and_then(|url| url.host_str().map(str::to_owned))
        .is_some_and(|host| {
            host == "localhost"
                || host
                    .trim_matches(['[', ']'])
                    .parse::<std::net::IpAddr>()
                    .is_ok_and(|ip| ip.is_loopback())
        })
}
async fn cancelled(cancel: &Cancellation) {
    while cancel.check().is_ok() {
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
}
fn transport(_: impl std::fmt::Display) -> Error {
    Error::new(
        ErrorCode::Unavailable,
        "daemon request failed; start coheryn serve or check endpoint/credentials (standalone fallback is never automatic)",
    )
}
fn decode_error(value: Value, status: u16) -> Error {
    if let Some(error) = value.get("error") {
        let code = serde_json::from_value(error.get("code").cloned().unwrap_or(Value::Null))
            .unwrap_or(if status == 401 || status == 403 {
                ErrorCode::Unauthorized
            } else {
                ErrorCode::Unavailable
            });
        return Error::new(
            code,
            error["message"]
                .as_str()
                .unwrap_or("daemon request failed")
                .chars()
                .take(8192)
                .collect::<String>(),
        );
    }
    if let Some(diagnostics) = value["compatibility"]["diagnostics"].as_array()
        && let Some(first) = diagnostics.first()
        && let Ok(error) = serde_json::from_value(first.clone())
    {
        return error;
    }
    Error::new(
        ErrorCode::Unavailable,
        format!("daemon returned HTTP {status}"),
    )
}
impl Client {
    pub fn new(endpoint: &str, inference: String, admin: Option<String>) -> Result<Self> {
        let endpoint =
            reqwest::Url::parse(endpoint).map_err(|_| Error::request("invalid daemon endpoint"))?;
        if !matches!(endpoint.scheme(), "http" | "https")
            || !endpoint.username().is_empty()
            || endpoint.password().is_some()
            || endpoint.query().is_some()
            || endpoint.fragment().is_some()
        {
            return Err(Error::request(
                "endpoint must be HTTP(S) without credentials, query or fragment",
            ));
        }
        let http = reqwest::Client::builder()
            .no_proxy()
            .redirect(reqwest::redirect::Policy::none())
            .connect_timeout(Duration::from_secs(3))
            .timeout(Duration::from_secs(600))
            .build()
            .map_err(transport)?;
        Ok(Self {
            http,
            endpoint,
            inference,
            admin,
        })
    }
    pub fn with_credentials(endpoint: &str, credentials: Credentials) -> Result<Self> {
        Self::new(
            endpoint,
            credentials.inference_token,
            Some(credentials.admin_token),
        )
    }
    fn request(&self, method: &str, path: &str, admin: bool) -> Result<reqwest::RequestBuilder> {
        let method = method
            .parse::<reqwest::Method>()
            .map_err(|_| Error::request("invalid HTTP method"))?;
        let url = self
            .endpoint
            .join(path)
            .map_err(|_| Error::request("invalid daemon route"))?;
        let token = if admin {
            self.admin.as_deref().ok_or_else(|| {
                Error::new(
                    ErrorCode::Unauthorized,
                    "administrative credential is required",
                )
            })?
        } else {
            &self.inference
        };
        Ok(self.http.request(method, url).bearer_auth(token))
    }
    async fn send(
        &self,
        request: reqwest::RequestBuilder,
        cancel: &Cancellation,
    ) -> Result<reqwest::Response> {
        cancel.check()?;
        tokio::select! {result=request.send()=>result.map_err(transport),_=cancelled(cancel)=>Err(Error::new(ErrorCode::Cancelled,"client cancelled"))}
    }
    async fn decode(&self, response: reqwest::Response, cancel: &Cancellation) -> Result<Value> {
        let status = response.status();
        let mut chunks = response.bytes_stream();
        let mut bytes = Vec::new();
        loop {
            let next = tokio::select! {next=chunks.next()=>next,_=cancelled(cancel)=>return Err(Error::new(ErrorCode::Cancelled,"client cancelled"))};
            let Some(chunk) = next else { break };
            let chunk = chunk.map_err(transport)?;
            if bytes.len().saturating_add(chunk.len()) > 16 << 20 {
                return Err(Error::resource("daemon response exceeds 16 MiB"));
            }
            bytes.extend_from_slice(&chunk);
        }
        let value: Value = serde_json::from_slice(&bytes)
            .map_err(|_| Error::new(ErrorCode::Io, "daemon returned invalid JSON"))?;
        if !status.is_success() {
            return Err(decode_error(value, status.as_u16()));
        }
        Ok(value)
    }
    pub async fn json(
        &self,
        method: &str,
        path: &str,
        body: Option<Value>,
        admin: bool,
        cancel: &Cancellation,
    ) -> Result<Value> {
        let mut request = self.request(method, path, admin)?;
        if let Some(body) = body {
            request = request.json(&body);
        }
        let response = self.send(request, cancel).await?;
        self.decode(response, cancel).await
    }
    pub async fn embeddings(
        &self,
        request: &crate::embeddings::EmbeddingRequest,
        cancel: &Cancellation,
    ) -> Result<ClientEmbeddings> {
        use crate::embeddings::{EmbeddingResponse, Encoding, Vector};
        let response = self
            .send(
                self.request("POST", "/v1/embeddings", false)?.json(request),
                cancel,
            )
            .await?;
        if !response.status().is_success() {
            let _ = self.decode(response, cancel).await?;
            return Err(Error::new(
                ErrorCode::Internal,
                "embedding error response was successful",
            ));
        }
        let header = |name: &str| -> Result<String> {
            response
                .headers()
                .get(name)
                .and_then(|v| v.to_str().ok())
                .filter(|v| v.len() <= 4096)
                .map(str::to_owned)
                .ok_or_else(|| {
                    Error::new(
                        ErrorCode::Io,
                        "daemon omitted required embedding identity/limit metadata",
                    )
                })
        };
        let space_id = header("x-coheryn-embedding-space")?;
        if space_id.len() != 85 || !space_id.starts_with("coheryn-embedding-v1-") {
            return Err(Error::new(
                ErrorCode::Io,
                "invalid embedding-space identity",
            ));
        }
        crate::registry::validate_hash(&space_id[21..]).map_err(|e| e.error)?;
        let model_sha256 = header("x-coheryn-model-sha256")?;
        crate::registry::validate_hash(&model_sha256).map_err(|e| e.error)?;
        let number = |name: &str| -> Result<usize> {
            header(name)?
                .parse()
                .map_err(|_| Error::new(ErrorCode::Io, "invalid embedding integer metadata"))
        };
        let dimension = number("x-coheryn-embedding-dimension")?;
        if !(1..=4096).contains(&dimension) {
            return Err(Error::new(
                ErrorCode::Io,
                "embedding dimension exceeds client bounds",
            ));
        }
        let numbers = |name: &str| -> Result<Vec<usize>> {
            let value = header(name)?;
            if value.is_empty() {
                return Ok(vec![]);
            }
            value
                .split(',')
                .map(|n| {
                    n.parse()
                        .map_err(|_| Error::new(ErrorCode::Io, "invalid embedding item metadata"))
                })
                .collect()
        };
        let token_counts = numbers("x-coheryn-token-counts")?;
        let truncated = numbers("x-coheryn-truncated-indices")?;
        let batch_size = number("x-coheryn-batch-size")?;
        let token_capacity = number("x-coheryn-batch-tokens")?;
        if !(1..=64).contains(&batch_size) || !(2..=4096).contains(&token_capacity) {
            return Err(Error::new(
                ErrorCode::Io,
                "invalid embedding microbatch metadata",
            ));
        }
        let mut chunks = response.bytes_stream();
        let mut bytes = Vec::new();
        loop {
            let next = tokio::select! {next=chunks.next()=>next,_=cancelled(cancel)=>return Err(Error::new(ErrorCode::Cancelled,"embedding client cancelled"))};
            let Some(chunk) = next else { break };
            let chunk = chunk.map_err(transport)?;
            if bytes.len().saturating_add(chunk.len()) > 16 << 20 {
                return Err(Error::resource(
                    "embedding response exceeds 16 MiB client limit",
                ));
            }
            bytes
                .try_reserve(chunk.len())
                .map_err(|_| Error::resource("embedding client allocation failed"))?;
            bytes.extend_from_slice(&chunk);
        }
        let result: EmbeddingResponse = serde_json::from_slice(&bytes)
            .map_err(|_| Error::new(ErrorCode::Io, "invalid embedding response JSON"))?;
        if result.object != "list"
            || result.model != request.model
            || result.data.len() != request.input.texts().len()
            || token_counts.len() != result.data.len()
            || token_counts.iter().any(|n| *n < 2 || *n > 512)
            || truncated.iter().any(|n| *n >= result.data.len())
            || truncated.windows(2).any(|p| p[0] >= p[1])
            || token_counts.iter().sum::<usize>() != result.usage.prompt_tokens
            || result.usage.total_tokens != result.usage.prompt_tokens
        {
            return Err(Error::new(
                ErrorCode::Io,
                "embedding response ordering, identity or usage mismatch",
            ));
        }
        for (index, item) in result.data.iter().enumerate() {
            let valid = match (&item.embedding, request.encoding_format) {
                (Vector::Float(v), Encoding::Float) => {
                    v.len() == dimension && v.iter().all(|v| v.is_finite())
                }
                (Vector::Base64(_), Encoding::Base64) => {
                    let v = item.embedding.floats()?;
                    v.len() == dimension && v.iter().all(|v| v.is_finite())
                }
                _ => false,
            };
            if item.index != index || item.object != "embedding" || !valid {
                return Err(Error::new(
                    ErrorCode::Io,
                    "invalid embedding vector or index",
                ));
            }
        }
        Ok(ClientEmbeddings {
            response: result,
            space_id,
            model_sha256,
            dimension,
            token_counts,
            truncated,
            batch_size,
            token_capacity,
        })
    }
    pub async fn import(
        &self,
        path: &Path,
        name: &str,
        sha256: Option<&str>,
        cancel: &Cancellation,
    ) -> Result<Value> {
        crate::registry::validate_id(name).map_err(|e| e.error)?;
        let file = tokio::fs::File::open(path).await?;
        let bytes = file.metadata().await?.len();
        let mut query = vec![("name", name)];
        if let Some(hash) = sha256 {
            crate::registry::validate_hash(hash).map_err(|e| e.error)?;
            query.push(("sha256", hash));
        }
        let request = self
            .request("POST", "/api/v1/models/import", true)?
            .query(&query)
            .header(reqwest::header::CONTENT_TYPE, "application/octet-stream")
            .header(reqwest::header::CONTENT_LENGTH, bytes)
            .body(reqwest::Body::wrap_stream(ReaderStream::new(file)));
        let response = self.send(request, cancel).await?;
        self.decode(response, cancel).await
    }
    pub async fn generate(
        &self,
        request: &GenerationRequest,
        cancel: &Cancellation,
        emit: impl FnMut(TokenEvent) -> Result<()>,
    ) -> Result<GenerationResult> {
        self.generate_with_tools(request, cancel, emit, |_| Ok(()))
            .await
    }
    pub async fn generate_with_tools(
        &self,
        request: &GenerationRequest,
        cancel: &Cancellation,
        emit: impl FnMut(TokenEvent) -> Result<()>,
        tool: impl FnMut(coheryn_engine::tools::ToolDelta) -> Result<()>,
    ) -> Result<GenerationResult> {
        let response = self
            .send(
                self.request("POST", "/api/v1/generate", false)?
                    .json(request),
                cancel,
            )
            .await?;
        self.receive_generation(response, request.stream, false, cancel, emit, tool)
            .await
    }
    pub async fn session_generate(
        &self,
        session: &str,
        request: &crate::sessions::SubmitTurn,
        cancel: &Cancellation,
        emit: impl FnMut(TokenEvent) -> Result<()>,
    ) -> Result<GenerationResult> {
        self.session_generate_with_tools(session, request, cancel, emit, |_| Ok(()))
            .await
    }
    pub async fn session_generate_with_tools(
        &self,
        session: &str,
        request: &crate::sessions::SubmitTurn,
        cancel: &Cancellation,
        emit: impl FnMut(TokenEvent) -> Result<()>,
        tool: impl FnMut(coheryn_engine::tools::ToolDelta) -> Result<()>,
    ) -> Result<GenerationResult> {
        crate::registry::validate_id(session).map_err(|e| e.error)?;
        let response = self
            .send(
                self.request(
                    "POST",
                    &format!("/coheryn/v1/sessions/{session}/messages"),
                    false,
                )?
                .json(request),
                cancel,
            )
            .await?;
        self.receive_generation(response, request.stream, true, cancel, emit, tool)
            .await
    }
    async fn receive_generation(
        &self,
        response: reqwest::Response,
        streaming: bool,
        managed: bool,
        cancel: &Cancellation,
        mut emit: impl FnMut(TokenEvent) -> Result<()>,
        mut tool: impl FnMut(coheryn_engine::tools::ToolDelta) -> Result<()>,
    ) -> Result<GenerationResult> {
        let sse = response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .is_some_and(|v| v.as_bytes().starts_with(b"text/event-stream"));
        if !streaming || !response.status().is_success() || (managed && !sse) {
            let value = self.decode(response, cancel).await?;
            let value = if managed {
                value.get("result").cloned().ok_or_else(|| {
                    Error::new(ErrorCode::Io, "managed response has no completed result")
                })?
            } else {
                value
            };
            let result: GenerationResult = serde_json::from_value(value)
                .map_err(|_| Error::new(ErrorCode::Io, "invalid native generation response"))?;
            if streaming && managed {
                emit(TokenEvent {
                    token_id: None,
                    text: result.text.clone(),
                })?;
            }
            if streaming && managed {
                for (index, c) in result.tool_calls.iter().enumerate() {
                    tool(coheryn_engine::tools::ToolDelta {
                        index,
                        id: Some(c.id.clone()),
                        kind: Some(c.kind.clone()),
                        function: coheryn_engine::tools::FunctionDelta {
                            name: Some(c.function.name.clone()),
                            arguments: c.function.arguments.clone(),
                        },
                    })?;
                }
            }
            return Ok(result);
        }
        if !response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .is_some_and(|v| v.as_bytes().starts_with(b"text/event-stream"))
        {
            return Err(Error::new(ErrorCode::Io, "daemon did not return SSE"));
        }
        let mut chunks = response.bytes_stream();
        let mut buffer = Vec::new();
        let mut result = None;
        loop {
            let next = tokio::select! {next=chunks.next()=>next,_=cancelled(cancel)=>return Err(Error::new(ErrorCode::Cancelled,"client cancelled"))};
            let Some(chunk) = next else {
                return Err(Error::new(ErrorCode::Io, "SSE ended without [DONE]"));
            };
            let chunk = chunk.map_err(transport)?;
            if buffer.len().saturating_add(chunk.len()) > 16 << 20 {
                return Err(Error::resource("SSE frame exceeds 16 MiB"));
            }
            buffer.extend_from_slice(&chunk);
            while let Some(end) = buffer.windows(2).position(|b| b == b"\n\n") {
                let frame = std::str::from_utf8(&buffer[..end])
                    .map_err(|_| Error::new(ErrorCode::Io, "invalid SSE UTF-8"))?
                    .to_owned();
                buffer.drain(..end + 2);
                let data = frame
                    .lines()
                    .filter_map(|line| line.strip_prefix("data:").map(str::trim_start))
                    .collect::<Vec<_>>()
                    .join("\n");
                if data.is_empty() {
                    continue;
                }
                if data == "[DONE]" {
                    return result.ok_or_else(|| {
                        Error::new(ErrorCode::Io, "SSE completed without a generation result")
                    });
                }
                let value: Value = serde_json::from_str(&data)
                    .map_err(|_| Error::new(ErrorCode::Io, "invalid SSE JSON"))?;
                if value.get("error").is_some_and(|v| !v.is_null()) {
                    return Err(decode_error(value, 500));
                }
                if let Some(token) = value.get("token") {
                    let token: TokenEvent = serde_json::from_value(token.clone())
                        .map_err(|_| Error::new(ErrorCode::Io, "invalid SSE token"))?;
                    emit(token)?;
                } else if let Some(delta) = value.get("tool_call") {
                    tool(
                        serde_json::from_value(delta.clone())
                            .map_err(|_| Error::new(ErrorCode::Io, "invalid SSE tool delta"))?,
                    )?;
                } else if let Some(terminal) = value.get("result") {
                    result = Some(
                        serde_json::from_value(terminal.clone())
                            .map_err(|_| Error::new(ErrorCode::Io, "invalid SSE result"))?,
                    );
                } else {
                    return Err(Error::new(ErrorCode::Io, "unknown native SSE event"));
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn local_vault_is_never_selected_for_remote_endpoints() {
        for endpoint in [
            "http://127.0.0.1:11450",
            "http://localhost:1234",
            "http://[::1]:11450",
        ] {
            assert!(super::loopback_endpoint(endpoint));
        }
        for endpoint in [
            "https://example.com",
            "http://127.0.0.1.example.com",
            "http://192.168.1.2:11450",
        ] {
            assert!(!super::loopback_endpoint(endpoint));
        }
    }
}
