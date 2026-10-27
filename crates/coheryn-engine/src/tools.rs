//! Bounded function-call protocol. Tool arguments are inert; this module executes no tools.
use crate::{Error, ErrorCode, FinishReason, Result, tokenizer::Message};
use serde::{Deserialize, Deserializer, Serialize, de};
use serde_json::Value;
use std::collections::BTreeSet;

pub const MAX_TOOLS: usize = 16;
/// Conservative retained allowance for a bounded materialized declaration tree.
pub const DECLARATION_RESERVATION: u64 = 2 << 20;
pub fn qualified_model_hash(sha: &str) -> bool {
    sha == "74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db"
}
pub const MAX_CALLS: usize = 8;
pub const MAX_SCHEMA_BYTES: usize = 65536;
pub const MAX_SCHEMA_NODES: usize = 256;
pub const MAX_SCHEMA_DEPTH: usize = 8;
pub const MAX_ARGUMENT_BYTES: usize = 65536;
pub const MAX_OUTPUT_BYTES: usize = 1 << 20;

pub fn bounded<'de, D, T, const N: usize>(d: D) -> std::result::Result<Vec<T>, D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de>,
{
    struct Items<T, const N: usize>(std::marker::PhantomData<T>);
    impl<'de, T: Deserialize<'de>, const N: usize> de::Visitor<'de> for Items<T, N> {
        type Value = Vec<T>;
        fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            write!(f, "at most {N} items")
        }
        fn visit_seq<A: de::SeqAccess<'de>>(
            self,
            mut a: A,
        ) -> std::result::Result<Vec<T>, A::Error> {
            let mut out = Vec::new();
            while let Some(v) = a.next_element()? {
                if out.len() == N {
                    return Err(de::Error::custom("tool array exceeds item limit"));
                }
                out.push(v);
            }
            Ok(out)
        }
    }
    d.deserialize_seq(Items::<T, N>(std::marker::PhantomData))
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Function {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub parameters: Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Tool {
    #[serde(rename = "type")]
    pub kind: String,
    pub function: Function,
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ChoiceMode {
    None,
    Auto,
    Required,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NamedFunction {
    pub name: String,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged, deny_unknown_fields)]
pub enum ToolChoice {
    Mode(ChoiceMode),
    Named {
        #[serde(rename = "type")]
        kind: String,
        function: NamedFunction,
    },
}
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ToolConfig {
    #[serde(default, deserialize_with = "bounded::<_,_,16>")]
    pub tools: Vec<Tool>,
    #[serde(default)]
    pub tool_choice: Option<ToolChoice>,
    #[serde(default)]
    pub parallel_tool_calls: Option<bool>,
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct CallFunction {
    pub name: String,
    pub arguments: String,
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: String,
    pub function: CallFunction,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDelta {
    pub index: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    pub kind: Option<String>,
    pub function: FunctionDelta,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    pub arguments: String,
}
pub fn identifier(s: &str) -> Result<()> {
    if s.is_empty()
        || s.len() > 64
        || !s
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
    {
        return Err(Error::request(
            "tool names and call IDs require 1..=64 ASCII letters, digits, hyphens or underscores",
        ));
    }
    Ok(())
}
fn schema(value: &Value, depth: usize, nodes: &mut usize) -> Result<()> {
    *nodes += 1;
    if depth > MAX_SCHEMA_DEPTH || *nodes > MAX_SCHEMA_NODES {
        return Err(Error::resource("tool schema exceeds nesting/node limit"));
    }
    match value {
        Value::Object(map) => {
            for (k, v) in map {
                if matches!(k.as_str(), "$ref" | "$dynamicRef" | "$id") {
                    return Err(Error::new(
                        ErrorCode::UnsupportedFeature,
                        "schema references are not supported; no schema assets are fetched",
                    ));
                }
                schema(v, depth + 1, nodes)?;
            }
        }
        Value::Array(a) => {
            for v in a {
                schema(v, depth + 1, nodes)?;
            }
        }
        _ => (),
    }
    Ok(())
}
impl ToolConfig {
    pub fn enabled(&self) -> bool {
        !self.tools.is_empty()
            && !matches!(self.tool_choice, Some(ToolChoice::Mode(ChoiceMode::None)))
    }
    pub fn validate(&self) -> Result<()> {
        if self.tools.len() > MAX_TOOLS {
            return Err(Error::resource("too many tools"));
        }
        let encoded =
            serde_json::to_vec(self).map_err(|_| Error::request("invalid tool definitions"))?;
        if encoded.len() > MAX_SCHEMA_BYTES {
            return Err(Error::resource("tool declarations exceed 64 KiB"));
        }
        let mut names = BTreeSet::new();
        let mut total_nodes = 0;
        for t in &self.tools {
            identifier(&t.function.name)?;
            if t.kind != "function"
                || !names.insert(&t.function.name)
                || t.function
                    .description
                    .as_ref()
                    .is_some_and(|s| s.len() > 4096)
            {
                return Err(Error::request("invalid or duplicate function declaration"));
            }
            if t.function.strict == Some(true) {
                return Err(Error::new(
                    ErrorCode::UnsupportedFeature,
                    "strict tool schemas require the S5 constraint decoder",
                ));
            }
            if !t.function.parameters.is_object()
                || t.function.parameters.get("type").and_then(Value::as_str) != Some("object")
            {
                return Err(Error::request(
                    "function parameters must be an object schema",
                ));
            }
            schema(&t.function.parameters, 0, &mut total_nodes)?;
        }
        match &self.tool_choice {
            Some(ToolChoice::Named { kind, function }) => {
                if kind != "function" || !names.contains(&function.name) {
                    return Err(Error::request(
                        "named tool_choice must select a declared function",
                    ));
                }
            }
            Some(ToolChoice::Mode(ChoiceMode::Required | ChoiceMode::Auto)) if names.is_empty() => {
                return Err(Error::request("this tool_choice requires declarations"));
            }
            _ => (),
        }
        if self.parallel_tool_calls.is_some() && names.is_empty() {
            return Err(Error::request("parallel_tool_calls requires declarations"));
        }
        Ok(())
    }
    pub fn validate_output(&self, calls: &[ToolCall], reason: FinishReason) -> Result<()> {
        if calls.len() > MAX_CALLS || (self.parallel_tool_calls == Some(false) && calls.len() > 1) {
            return Err(Error::request("invalid generated tool-call count"));
        }
        if calls.is_empty() {
            if reason == FinishReason::ToolCalls
                || matches!(
                    self.tool_choice,
                    Some(ToolChoice::Mode(ChoiceMode::Required) | ToolChoice::Named { .. })
                )
            {
                return Err(Error::request("required tool response is missing"));
            }
        } else {
            if reason != FinishReason::ToolCalls {
                return Err(Error::request(
                    "tool calls require a successful tool_calls finish",
                ));
            }
            let mut ids = BTreeSet::new();
            for call in calls {
                validate_call(call)?;
                if !ids.insert(&call.id) || !self.accepts(&call.function.name) {
                    return Err(Error::request(
                        "invalid generated call identity or selection",
                    ));
                }
            }
        }
        Ok(())
    }
    pub fn effective_tools(&self) -> Vec<Value> {
        if !self.enabled() {
            return vec![];
        }
        self.tools
            .iter()
            .filter(|t| match &self.tool_choice {
                Some(ToolChoice::Named { function, .. }) => t.function.name == function.name,
                _ => true,
            })
            .map(|t| serde_json::to_value(t).expect("serializable tool"))
            .collect()
    }
    fn accepts(&self, name: &str) -> bool {
        self.enabled()
            && self.tools.iter().any(|t| t.function.name == name)
            && match &self.tool_choice {
                Some(ToolChoice::Named { function, .. }) => function.name == name,
                _ => true,
            }
    }
    pub fn validate_messages(
        &self,
        messages: &[Message],
        allow_pending: bool,
    ) -> Result<Vec<String>> {
        self.validate()?;
        if messages.is_empty() || messages.len() > 256 {
            return Err(Error::request("chat needs 1..=256 messages"));
        }
        let mut pending = Vec::new();
        let mut seen = BTreeSet::new();
        let mut bytes = 0usize;
        for m in messages {
            bytes = bytes
                .checked_add(
                    serde_json::to_vec(m)
                        .map_err(|_| Error::request("invalid message"))?
                        .len(),
                )
                .ok_or_else(|| Error::resource("message size overflow"))?;
            if bytes > MAX_OUTPUT_BYTES {
                return Err(Error::resource("chat input exceeds one MiB"));
            }
            if !pending.is_empty() && m.role != "tool" {
                return Err(Error::request(
                    "all pending tool calls need matching results before the next turn",
                ));
            }
            if m.role == "tool" {
                let id = m
                    .tool_call_id
                    .as_ref()
                    .ok_or_else(|| Error::request("tool result needs tool_call_id"))?;
                if !m.tool_calls.is_empty() || pending.first() != Some(id) {
                    return Err(Error::request(
                        "tool results must match outstanding IDs in call order",
                    ));
                }
                pending.remove(0);
            } else {
                if m.tool_call_id.is_some()
                    || !matches!(m.role.as_str(), "system" | "user" | "assistant")
                {
                    return Err(Error::request("unsupported role or tool_call_id"));
                }
                if !m.tool_calls.is_empty() && m.role != "assistant" {
                    return Err(Error::request(
                        "only assistant messages can contain tool calls",
                    ));
                }
                if m.tool_calls.len() > MAX_CALLS {
                    return Err(Error::resource("too many tool calls in a turn"));
                }
                for c in &m.tool_calls {
                    validate_call(c)?;
                    if !seen.insert(c.id.clone())
                        || (!self.tools.is_empty()
                            && !self
                                .tools
                                .iter()
                                .any(|t| t.function.name == c.function.name))
                    {
                        return Err(Error::request(
                            "duplicate call ID or undeclared historical function",
                        ));
                    }
                    pending.push(c.id.clone());
                }
            }
        }
        if !allow_pending && !pending.is_empty() {
            return Err(Error::request("tool results are missing"));
        }
        Ok(pending)
    }
}
pub fn validate_call(call: &ToolCall) -> Result<()> {
    identifier(&call.id)?;
    identifier(&call.function.name)?;
    if call.kind != "function" || call.function.arguments.len() > MAX_ARGUMENT_BYTES {
        return Err(Error::request("invalid tool call type or argument size"));
    }
    validate_arguments(&call.function.arguments)?;
    Ok(())
}
/// Canonical UTF-8 JSON profile shared with the independently captured template fixtures.
pub fn template_json(v: Value) -> String {
    fn write(v: &Value, out: &mut String) {
        match v {
            Value::Array(a) => {
                out.push('[');
                for (i, x) in a.iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    write(x, out);
                }
                out.push(']');
            }
            Value::Object(m) => {
                out.push('{');
                for (i, (k, x)) in m.iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    out.push_str(&serde_json::to_string(k).unwrap());
                    out.push_str(": ");
                    write(x, out);
                }
                out.push('}');
            }
            _ => out.push_str(&serde_json::to_string(v).unwrap()),
        }
    }
    let mut out = String::new();
    write(&v, &mut out);
    out
}
pub enum ParsedEvent {
    Text(String),
    Delta(ToolDelta),
}
pub struct ToolParser {
    config: ToolConfig,
    prefix: String,
    pending: String,
    text: String,
    calls: Vec<ToolCall>,
    bytes: usize,
    in_call: bool,
    scan: usize,
    quoted: bool,
    escaped: bool,
}
impl ToolParser {
    pub fn new(config: ToolConfig, prefix: String) -> Result<Self> {
        config.validate()?;
        identifier(&prefix)?;
        Ok(Self {
            config,
            prefix,
            pending: String::new(),
            text: String::new(),
            calls: vec![],
            bytes: 0,
            in_call: false,
            scan: 0,
            quoted: false,
            escaped: false,
        })
    }
    pub fn push(&mut self, fragment: &str) -> Result<Vec<ParsedEvent>> {
        self.bytes = self
            .bytes
            .checked_add(fragment.len())
            .filter(|b| *b <= MAX_OUTPUT_BYTES)
            .ok_or_else(|| Error::resource("tool output exceeds one MiB"))?;
        self.pending.push_str(fragment);
        let mut events = Vec::new();
        loop {
            if self.in_call {
                if let Some(end) = self.closing_tag() {
                    let raw = self.pending[..end].trim();
                    #[derive(Deserialize)]
                    #[serde(deny_unknown_fields)]
                    struct Raw {
                        name: String,
                        arguments: Box<serde_json::value::RawValue>,
                    }
                    let call: Raw = serde_json::from_str(raw)
                        .map_err(|_| Error::request("model emitted malformed tool JSON"))?;
                    if raw.len() > MAX_ARGUMENT_BYTES
                        || !call.arguments.get().trim_start().starts_with('{')
                        || !self.config.accepts(&call.name)
                    {
                        return Err(Error::request(
                            "model emitted oversized arguments or an undeclared/forbidden function",
                        ));
                    }
                    if self.calls.len() >= MAX_CALLS
                        || (self.config.parallel_tool_calls == Some(false)
                            && !self.calls.is_empty())
                    {
                        return Err(Error::request(
                            "model exceeded the permitted tool-call count",
                        ));
                    }
                    let index = self.calls.len();
                    let id = format!("{}-{index}", self.prefix);
                    identifier(&id)?;
                    let arguments = call.arguments.get().to_owned();
                    validate_arguments(&arguments)?;
                    let c = ToolCall {
                        id: id.clone(),
                        kind: "function".into(),
                        function: CallFunction {
                            name: call.name.clone(),
                            arguments: arguments.clone(),
                        },
                    };
                    self.calls.push(c);
                    let mut start = 0;
                    while start < arguments.len() {
                        let mut end = (start + 256).min(arguments.len());
                        while !arguments.is_char_boundary(end) {
                            end -= 1;
                        }
                        events.push(ParsedEvent::Delta(ToolDelta {
                            index,
                            id: (start == 0).then(|| id.clone()),
                            kind: (start == 0).then(|| "function".into()),
                            function: FunctionDelta {
                                name: (start == 0).then(|| call.name.clone()),
                                arguments: arguments[start..end].into(),
                            },
                        }));
                        start = end;
                    }
                    self.pending.drain(..end + "</tool_call>".len());
                    self.in_call = false;
                    self.scan = 0;
                    self.quoted = false;
                    self.escaped = false;
                } else {
                    if self.pending.len() > MAX_ARGUMENT_BYTES + 32 {
                        return Err(Error::resource(
                            "incomplete tool call exceeds argument limit",
                        ));
                    }
                    break;
                }
            } else if let Some(index) = self.pending.find("<tool_call>") {
                let text = self.pending[..index].to_owned();
                self.text.push_str(&text);
                if !text.is_empty() {
                    events.push(ParsedEvent::Text(text));
                }
                self.pending.drain(..index + "<tool_call>".len());
                self.in_call = true;
            } else {
                let marker = "<tool_call>";
                let tail = (1..marker.len())
                    .rev()
                    .find(|n| self.pending.ends_with(&marker[..*n]))
                    .unwrap_or(0);
                let text = self.pending[..self.pending.len() - tail].to_owned();
                self.pending.drain(..text.len());
                self.text.push_str(&text);
                if !text.is_empty() {
                    events.push(ParsedEvent::Text(text));
                }
                break;
            }
        }
        Ok(events)
    }
    fn closing_tag(&mut self) -> Option<usize> {
        let bytes = self.pending.as_bytes();
        while self.scan < bytes.len() {
            let b = bytes[self.scan];
            if self.quoted {
                if self.escaped {
                    self.escaped = false;
                } else if b == b'\\' {
                    self.escaped = true;
                } else if b == b'"' {
                    self.quoted = false;
                }
            } else if b == b'"' {
                self.quoted = true;
            } else if b == b'<' {
                let remaining = &bytes[self.scan..];
                let marker = b"</tool_call>";
                if remaining.starts_with(marker) {
                    return Some(self.scan);
                }
                if marker.starts_with(remaining) {
                    return None;
                }
            }
            self.scan += 1;
        }
        None
    }
    pub fn finish(mut self, reason: FinishReason) -> Result<(String, Vec<ToolCall>, String)> {
        if reason == FinishReason::Cancelled {
            return Err(Error::new(
                ErrorCode::Cancelled,
                "tool generation cancelled",
            ));
        }
        if self.in_call || (!self.pending.is_empty() && "<tool_call>".starts_with(&self.pending)) {
            return Err(Error::request("incomplete tool-call framing"));
        }
        if !self.calls.is_empty() && reason != FinishReason::Stop {
            return Err(Error::request(
                "tool generation ended before a successful model stop",
            ));
        }
        if self.calls.is_empty()
            && matches!(
                self.config.tool_choice,
                Some(ToolChoice::Mode(ChoiceMode::Required) | ToolChoice::Named { .. })
            )
        {
            return Err(Error::request(
                "model did not produce the required tool call",
            ));
        }
        let tail = std::mem::take(&mut self.pending);
        self.text.push_str(&tail);
        Ok((self.text, self.calls, tail))
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ToolCapabilities {
    pub profile: &'static str,
    pub companion_template_sha256: String,
    pub choices: [&'static str; 4],
    pub parallel_call_protocol: bool,
    pub max_tools: usize,
    pub max_calls: usize,
    pub schema_bytes: usize,
    pub schema_nodes: usize,
    pub schema_depth: usize,
    pub argument_bytes: usize,
    pub argument_depth: usize,
    pub argument_validation: &'static str,
    pub strict_schema: bool,
    pub result_order: &'static str,
    pub qualification: &'static str,
}
pub fn capabilities(d: &crate::Descriptor) -> Option<ToolCapabilities> {
    use sha2::{Digest, Sha256};
    let template = d.text("tokenizer.chat_template")?;
    let fingerprint =
        crate::storage::digest_hex(Sha256::digest(serde_json::to_vec(template).ok()?).into());
    if d.architecture.as_deref() != Some("qwen2")
        || fingerprint != "71672240b1a3b068067b6a332bc38d3fc05aaade9d41b4b27d7510ff4335504b"
    {
        return None;
    }
    Some(ToolCapabilities {
        profile: "qwen-tools-v1",
        companion_template_sha256: crate::storage::digest_hex(
            Sha256::digest(include_bytes!(
                "../../../fixtures/tools/qwen2.5-tools-v1.jinja"
            ))
            .into(),
        ),
        choices: ["none", "auto", "required", "named"],
        parallel_call_protocol: true,
        max_tools: MAX_TOOLS,
        max_calls: MAX_CALLS,
        schema_bytes: MAX_SCHEMA_BYTES,
        schema_nodes: MAX_SCHEMA_NODES,
        schema_depth: MAX_SCHEMA_DEPTH,
        argument_bytes: MAX_ARGUMENT_BYTES,
        argument_depth: 16,
        argument_validation: "JSON object and declared function membership; schemas are non-strict conditioning",
        strict_schema: false,
        result_order: "original assistant call order",
        qualification: "template/protocol adapter; exact model behavior requires its compatibility-matrix evidence",
    })
}

fn validate_arguments(arguments: &str) -> Result<()> {
    let raw: Box<serde_json::value::RawValue> = serde_json::from_str(arguments)
        .map_err(|_| Error::request("tool arguments must be valid JSON"))?;
    if !raw.get().starts_with('{') {
        return Err(Error::request("tool arguments must be a JSON object"));
    }
    let mut quoted = false;
    let mut escaped = false;
    let mut depth = 0usize;
    for b in raw.get().bytes() {
        if quoted {
            if escaped {
                escaped = false;
            } else if b == b'\\' {
                escaped = true;
            } else if b == b'"' {
                quoted = false;
            }
        } else {
            match b {
                b'"' => quoted = true,
                b'{' | b'[' => {
                    depth += 1;
                    if depth > 16 {
                        return Err(Error::resource("tool arguments exceed nesting limit"));
                    }
                }
                b'}' | b']' => depth = depth.saturating_sub(1),
                _ => (),
            }
        }
    }
    Ok(())
}

pub fn profile_identity() -> String {
    use sha2::{Digest, Sha256};
    format!(
        "qwen-tools-v1:{}",
        crate::storage::digest_hex(
            Sha256::digest(include_bytes!(
                "../../../fixtures/tools/qwen2.5-tools-v1.jinja"
            ))
            .into()
        )
    )
}
