//! Request-owned, bounded JSON byte grammar and tokenizer-trie masks.
use crate::{Cancellation, Error, ErrorCode, Result, tokenizer::TextTokenizer};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

pub const PROFILE: &str = "coheryn-json-v1";
pub const RESERVATION_BYTES: u64 = 32 << 20;
pub const MAX_SCHEMA_BYTES: usize = 16 << 10;
pub const MAX_OUTPUT_BYTES: usize = 64 << 10;
const DEPTH: usize = 8;
const MAX_NODES: usize = 256;
const MAX_TRIE: usize = 1 << 20;
const NONE: u32 = u32::MAX;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum ResponseFormat {
    #[default]
    Text,
    JsonObject,
    JsonSchema {
        json_schema: SchemaRequest,
    },
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SchemaRequest {
    pub name: String,
    pub strict: bool,
    pub schema: Value,
}
impl ResponseFormat {
    pub fn is_text(&self) -> bool {
        !self.enabled()
    }
    pub fn enabled(&self) -> bool {
        !matches!(self, Self::Text)
    }
    pub fn compile(&self) -> Result<Option<Grammar>> {
        match self {
            Self::Text => Ok(None),
            Self::JsonObject => Ok(Some(Grammar {
                identity: crate::storage::digest_hex(
                    Sha256::digest(b"coheryn-json-v1:json_object").into(),
                ),
                nodes: vec![
                    Node::any(),
                    Node {
                        kind: Kind::Object(None, 0),
                        nullable: false,
                        choices: vec![],
                    },
                ],
                root: 1,
            })),
            Self::JsonSchema {
                json_schema: request,
            } => {
                if !request.strict
                    || request.name.is_empty()
                    || request.name.len() > 64
                    || !request
                        .name
                        .bytes()
                        .all(|c| c.is_ascii_alphanumeric() || b"_-".contains(&c))
                {
                    return Err(unsupported(
                        "json_schema requires strict=true and a 1..=64 character ASCII name",
                    ));
                }
                Grammar::compile(&request.schema).map(Some)
            }
        }
    }
}
fn unsupported(message: impl Into<String>) -> Error {
    Error::new(ErrorCode::UnsupportedFeature, message)
}
#[derive(Debug)]
struct Property {
    key: Vec<u8>,
    name: String,
    node: usize,
}
#[derive(Debug)]
enum Kind {
    Any,
    Object(Option<Vec<Property>>, u64),
    Array(usize, usize),
    String(usize),
    Number(bool),
    Bool,
    Null,
}
#[derive(Debug)]
struct Node {
    kind: Kind,
    nullable: bool,
    choices: Vec<Vec<u8>>,
}
impl Node {
    fn any() -> Self {
        Self {
            kind: Kind::Any,
            nullable: true,
            choices: vec![],
        }
    }
}
#[derive(Debug)]
pub struct Grammar {
    nodes: Vec<Node>,
    root: usize,
    identity: String,
}
impl Grammar {
    pub fn compile(schema: &Value) -> Result<Self> {
        fn bounds(value: &Value, depth: usize, count: &mut usize) -> Result<()> {
            *count += 1;
            if depth > 32 || *count > 4096 {
                return Err(Error::resource("schema source nesting/node limit exceeded"));
            }
            match value {
                Value::Object(map) => {
                    for (key, value) in map {
                        if key.len() > MAX_SCHEMA_BYTES {
                            return Err(Error::resource("oversized schema key"));
                        }
                        bounds(value, depth + 1, count)?;
                    }
                }
                Value::Array(values) => {
                    for value in values {
                        bounds(value, depth + 1, count)?;
                    }
                }
                Value::String(value) if value.len() > MAX_SCHEMA_BYTES => {
                    return Err(Error::resource("oversized schema string"));
                }
                _ => (),
            }
            Ok(())
        }
        bounds(schema, 0, &mut 0)?;
        struct Counter(usize, Sha256);
        impl std::io::Write for Counter {
            fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
                self.0 = self
                    .0
                    .checked_add(bytes.len())
                    .filter(|n| *n <= MAX_SCHEMA_BYTES)
                    .ok_or_else(|| std::io::Error::other("schema exceeds 16 KiB"))?;
                self.1.update(bytes);
                Ok(bytes.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        let mut counter = Counter(0, Sha256::new());
        serde_json::to_writer(&mut counter, schema)
            .map_err(|_| Error::resource("schema exceeds 16 KiB"))?;
        let mut grammar = Self {
            identity: crate::storage::digest_hex(counter.1.finalize().into()),
            nodes: vec![Node::any()],
            root: 0,
        };
        grammar.root = grammar.node(schema, 0)?;
        if !matches!(grammar.nodes[grammar.root].kind, Kind::Object(_, _))
            || grammar.nodes[grammar.root].nullable
        {
            return Err(unsupported(
                "strict JSON schemas require a non-nullable root object",
            ));
        }
        Ok(grammar)
    }
    fn node(&mut self, schema: &Value, depth: usize) -> Result<usize> {
        if depth >= DEPTH || self.nodes.len() >= MAX_NODES {
            return Err(Error::resource("schema depth/node limit exceeded"));
        }
        let object = schema
            .as_object()
            .ok_or_else(|| unsupported("schema nodes must be typed objects"))?;
        for key in object.keys() {
            if ![
                "type",
                "properties",
                "required",
                "additionalProperties",
                "items",
                "enum",
                "const",
                "maxLength",
                "maxItems",
                "title",
                "description",
                "$schema",
            ]
            .contains(&key.as_str())
            {
                return Err(unsupported(format!("unsupported schema keyword: {key}")));
            }
        }
        for key in ["title", "description", "$schema"] {
            if object.get(key).is_some_and(|v| !v.is_string()) {
                return Err(unsupported("schema annotations must be strings"));
            }
        }
        if object
            .get("$schema")
            .is_some_and(|v| v != "https://json-schema.org/draft/2020-12/schema")
        {
            return Err(unsupported(
                "only the 2020-12 schema identifier is qualified",
            ));
        }
        let mut nullable = false;
        let ty = match object.get("type") {
            Some(Value::String(t)) => t.as_str(),
            Some(Value::Array(types))
                if types.len() == 2
                    && types.iter().filter(|v| v.as_str() == Some("null")).count() == 1 =>
            {
                nullable = true;
                types
                    .iter()
                    .find_map(|v| v.as_str().filter(|t| *t != "null"))
                    .ok_or_else(|| unsupported("invalid nullable type"))?
            }
            _ => {
                return Err(unsupported(
                    "each schema node requires one type or [type, null]",
                ));
            }
        };
        let limit = |key: &str, default: usize, maximum: usize| -> Result<usize> {
            match object.get(key) {
                None => Ok(default),
                Some(v) => v
                    .as_u64()
                    .filter(|n| *n <= maximum as u64)
                    .map(|n| n as usize)
                    .ok_or_else(|| {
                        unsupported(format!("{key} must be an integer in 0..={maximum}"))
                    }),
            }
        };
        let index = self.nodes.len();
        self.nodes.push(Node::any());
        let kind = match ty {
            "object" => {
                if object.get("additionalProperties") != Some(&Value::Bool(false)) {
                    return Err(unsupported(
                        "object schemas require additionalProperties=false",
                    ));
                }
                let properties = object
                    .get("properties")
                    .and_then(Value::as_object)
                    .ok_or_else(|| unsupported("object schemas require properties"))?;
                if properties.len() > 64 {
                    return Err(Error::resource("object exceeds 64 properties"));
                }
                let mut props = vec![];
                for (name, child) in properties {
                    if name.len() > 256 {
                        return Err(Error::resource("property name exceeds 256 bytes"));
                    }
                    props.push(Property {
                        key: serde_json::to_vec(name).unwrap(),
                        name: name.clone(),
                        node: self.node(child, depth + 1)?,
                    });
                }
                let mut required = 0u64;
                if let Some(values) = object.get("required") {
                    for name in values
                        .as_array()
                        .ok_or_else(|| unsupported("required must be an array"))?
                    {
                        let position = props
                            .iter()
                            .position(|p| Some(p.name.as_str()) == name.as_str())
                            .ok_or_else(|| {
                                unsupported("required names must exist in properties")
                            })?;
                        let bit = 1u64 << position;
                        if required & bit != 0 {
                            return Err(unsupported("duplicate required property"));
                        }
                        required |= bit;
                    }
                }
                Kind::Object(Some(props), required)
            }
            "array" => Kind::Array(
                self.node(
                    object
                        .get("items")
                        .ok_or_else(|| unsupported("arrays require homogeneous items"))?,
                    depth + 1,
                )?,
                limit("maxItems", 32, 128)?,
            ),
            "string" => Kind::String(limit("maxLength", 256, 4096)?),
            "number" => Kind::Number(false),
            "integer" => Kind::Number(true),
            "boolean" => Kind::Bool,
            "null" => Kind::Null,
            _ => return Err(unsupported("unsupported schema type")),
        };
        for (key, allowed) in [
            ("properties", "object"),
            ("required", "object"),
            ("additionalProperties", "object"),
            ("items", "array"),
            ("maxItems", "array"),
            ("maxLength", "string"),
        ] {
            if object.contains_key(key) && ty != allowed {
                return Err(unsupported(format!("{key} is not applicable to {ty}")));
            }
        }
        self.nodes[index] = Node {
            kind,
            nullable,
            choices: vec![],
        };
        if object.contains_key("const") && object.contains_key("enum") {
            return Err(unsupported(
                "combine neither const nor enum with each other",
            ));
        }
        let values = if let Some(value) = object.get("const") {
            vec![value.clone()]
        } else if let Some(value) = object.get("enum") {
            value
                .as_array()
                .ok_or_else(|| unsupported("enum must be an array"))?
                .clone()
        } else {
            vec![]
        };
        if object.contains_key("enum") && values.is_empty() || values.len() > 32 {
            return Err(unsupported("enum must contain 1..=32 primitive values"));
        }
        let mut choices = vec![];
        for value in values {
            if value.is_array() || value.is_object() || !self.valid(index, &value) {
                return Err(unsupported(
                    "enum/const values must be valid primitives of the declared type",
                ));
            }
            let bytes = serde_json::to_vec(&value).unwrap();
            if bytes.len() > 4096 || choices.contains(&bytes) {
                return Err(unsupported("duplicate or oversized enum value"));
            }
            choices.push(bytes);
        }
        self.nodes[index].choices = choices;
        Ok(index)
    }
    fn valid(&self, index: usize, value: &Value) -> bool {
        let node = &self.nodes[index];
        if !node.choices.is_empty()
            && !node
                .choices
                .iter()
                .any(|bytes| serde_json::to_vec(value).is_ok_and(|v| &v == bytes))
        {
            return false;
        }
        if value.is_null() {
            return node.nullable || matches!(node.kind, Kind::Null | Kind::Any);
        }
        match &node.kind {
            Kind::Any => true,
            Kind::Null => false,
            Kind::Bool => value.is_boolean(),
            Kind::Number(integer) => {
                value.is_number()
                    && (!integer
                        || value.as_i64().is_some()
                        || value.as_u64().is_some()
                        || value
                            .as_f64()
                            .is_some_and(|v| v.is_finite() && v.fract() == 0.))
            }
            Kind::String(max) => value.as_str().is_some_and(|s| s.chars().count() <= *max),
            Kind::Array(child, max) => value
                .as_array()
                .is_some_and(|vs| vs.len() <= *max && vs.iter().all(|v| self.valid(*child, v))),
            Kind::Object(None, _) => value.is_object(),
            Kind::Object(Some(props), required) => value.as_object().is_some_and(|vs| {
                props.iter().enumerate().all(|(i, p)| {
                    vs.get(&p.name)
                        .map_or(required & (1 << i) == 0, |v| self.valid(p.node, v))
                }) && vs.keys().all(|k| props.iter().any(|p| &p.name == k))
            }),
        }
    }
    pub fn validate_output(&self, output: &str) -> Result<()> {
        if output.len() > MAX_OUTPUT_BYTES {
            return Err(Error::resource("constrained output exceeds 64 KiB"));
        }
        let mut state = State::new(self.root);
        for byte in output.bytes() {
            if !state.byte(byte, self) {
                return Err(Error::new(
                    ErrorCode::ConstraintViolation,
                    "output violates the qualified JSON grammar/schema profile",
                ));
            }
        }
        if !state.complete {
            return Err(Error::new(
                ErrorCode::ConstraintIncomplete,
                "constrained output is incomplete",
            ));
        }
        let value: Value = serde_json::from_str(output).map_err(|_| {
            Error::new(
                ErrorCode::ConstraintIncomplete,
                "constrained output is not complete JSON",
            )
        })?;
        if !self.valid(self.root, &value) {
            return Err(Error::new(
                ErrorCode::ConstraintViolation,
                "output failed the exact schema profile",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Default)]
struct Frame {
    node: usize,
    seen: u64,
    count: u16,
    phase: u8,
    child: usize,
    array: bool,
}
#[derive(Clone, Copy)]
enum Mode {
    Value(usize),
    Normal,
    Key {
        possible: u64,
        offset: u16,
    },
    String(StringState),
    Literal {
        node: usize,
        kind: u8,
        possible: u32,
        offset: u16,
    },
    Number {
        integer: bool,
        phase: u8,
        digits: u8,
        bytes: u8,
        exponent: u8,
    },
}
#[derive(Clone, Copy, Default)]
struct StringState {
    key: bool,
    max: u16,
    chars: u16,
    escape: u8,
    hex: u16,
    remaining: u8,
    low: u8,
    high: u8,
    surrogate: bool,
}
#[derive(Clone, Copy)]
struct State {
    frames: [Frame; DEPTH],
    depth: usize,
    mode: Mode,
    complete: bool,
    wrapped: bool,
    prefix: u8,
    suffix: u8,
}
impl State {
    fn new(root: usize) -> Self {
        Self {
            frames: [Frame::default(); DEPTH],
            depth: 0,
            mode: Mode::Value(root),
            complete: false,
            wrapped: false,
            prefix: 0,
            suffix: 0,
        }
    }
    fn finish_value(&mut self) {
        self.mode = Mode::Normal;
        if self.depth == 0 {
            self.complete = true;
        }
    }
    fn push(&mut self, node: usize, array: bool) -> bool {
        if self.depth == DEPTH {
            return false;
        }
        self.frames[self.depth] = Frame {
            node,
            array,
            ..Frame::default()
        };
        self.depth += 1;
        self.mode = Mode::Normal;
        true
    }
    fn byte(&mut self, b: u8, grammar: &Grammar) -> bool {
        const PREFIX: &[u8] = b"<tool_call>\n";
        const SUFFIX: &[u8] = b"\n</tool_call>";
        if self.wrapped && (self.prefix as usize) < PREFIX.len() {
            if PREFIX[self.prefix as usize] != b {
                return false;
            }
            self.prefix += 1;
            return true;
        }
        if self.wrapped && self.complete {
            if (self.suffix as usize) < SUFFIX.len() {
                if SUFFIX[self.suffix as usize] != b {
                    return false;
                }
                self.suffix += 1;
                return true;
            }
            return whitespace(b);
        }
        match self.mode {
            Mode::Value(node) => {
                if whitespace(b) {
                    return true;
                }
                let rule = &grammar.nodes[node];
                if !rule.choices.is_empty() {
                    self.mode = Mode::Literal {
                        node,
                        kind: 0,
                        possible: u32::MAX,
                        offset: 0,
                    };
                    return self.byte(b, grammar);
                }
                if b == b'n' && (rule.nullable || matches!(rule.kind, Kind::Null | Kind::Any)) {
                    self.mode = Mode::Literal {
                        node,
                        kind: 1,
                        possible: 1,
                        offset: 0,
                    };
                    return self.byte(b, grammar);
                }
                match (&rule.kind, b) {
                    (Kind::Object(_, _) | Kind::Any, b'{') => self.push(node, false),
                    (Kind::Array(_, _) | Kind::Any, b'[') => self.push(node, true),
                    (Kind::String(_) | Kind::Any, b'"') => {
                        let max = if let Kind::String(max) = rule.kind {
                            max
                        } else {
                            256
                        };
                        self.mode = Mode::String(StringState {
                            max: max as u16,
                            ..StringState::default()
                        });
                        true
                    }
                    (Kind::Bool | Kind::Any, b't' | b'f') => {
                        self.mode = Mode::Literal {
                            node,
                            kind: if b == b't' { 2 } else { 3 },
                            possible: 1,
                            offset: 0,
                        };
                        self.byte(b, grammar)
                    }
                    (Kind::Number(_) | Kind::Any, b'-' | b'0'..=b'9') => {
                        self.mode = Mode::Number {
                            integer: matches!(rule.kind, Kind::Number(true)),
                            phase: 0,
                            digits: 0,
                            bytes: 0,
                            exponent: 0,
                        };
                        self.byte(b, grammar)
                    }
                    _ => false,
                }
            }
            Mode::Normal => {
                if whitespace(b) {
                    return true;
                }
                if self.complete || self.depth == 0 {
                    return false;
                }
                let frame = self.frames[self.depth - 1];
                if frame.array {
                    let (child, max) = match grammar.nodes[frame.node].kind {
                        Kind::Array(c, m) => (c, m),
                        _ => (0, 32),
                    };
                    if b == b']' && (frame.phase == 0 || frame.phase == 3) {
                        self.depth -= 1;
                        self.finish_value();
                        return true;
                    }
                    if frame.phase == 3 {
                        if b != b',' || frame.count as usize >= max {
                            return false;
                        }
                        self.frames[self.depth - 1].phase = 1;
                        return true;
                    }
                    if frame.count as usize >= max {
                        return false;
                    }
                    self.frames[self.depth - 1].count += 1;
                    self.frames[self.depth - 1].phase = 3;
                    self.mode = Mode::Value(child);
                    self.byte(b, grammar)
                } else {
                    let (props, required) = match &grammar.nodes[frame.node].kind {
                        Kind::Object(p, r) => (p.as_ref(), *r),
                        _ => (None, 0),
                    };
                    if b == b'}'
                        && (frame.phase == 0 || frame.phase == 3)
                        && frame.seen & required == required
                    {
                        self.depth -= 1;
                        self.finish_value();
                        return true;
                    }
                    match frame.phase {
                        3 => {
                            if b != b','
                                || frame.count >= 64
                                || props
                                    .is_some_and(|p| frame.seen.count_ones() as usize >= p.len())
                            {
                                return false;
                            }
                            self.frames[self.depth - 1].phase = 1;
                            true
                        }
                        2 => {
                            if b != b':' {
                                return false;
                            }
                            self.frames[self.depth - 1].phase = 3;
                            self.mode = Mode::Value(frame.child);
                            true
                        }
                        _ if b == b'"' && frame.count < 64 => {
                            if let Some(props) = props {
                                let possible = if props.len() == 64 {
                                    u64::MAX
                                } else {
                                    (1u64 << props.len()) - 1
                                } & !frame.seen;
                                if possible == 0 {
                                    return false;
                                }
                                self.mode = Mode::Key {
                                    possible,
                                    offset: 1,
                                };
                            } else {
                                self.mode = Mode::String(StringState {
                                    key: true,
                                    max: 256,
                                    ..StringState::default()
                                });
                            }
                            true
                        }
                        _ => false,
                    }
                }
            }
            Mode::Key { possible, offset } => {
                let frame = &mut self.frames[self.depth - 1];
                let Kind::Object(Some(props), _) = &grammar.nodes[frame.node].kind else {
                    return false;
                };
                let mut next = 0;
                for (index, prop) in props.iter().enumerate() {
                    if possible & (1 << index) != 0 && prop.key.get(offset as usize) == Some(&b) {
                        if prop.key.len() == offset as usize + 1 {
                            frame.seen |= 1 << index;
                            frame.count += 1;
                            frame.child = prop.node;
                            frame.phase = 2;
                            self.mode = Mode::Normal;
                            return true;
                        }
                        next |= 1 << index;
                    }
                }
                if next == 0 {
                    return false;
                }
                self.mode = Mode::Key {
                    possible: next,
                    offset: offset + 1,
                };
                true
            }
            Mode::Literal {
                node,
                kind,
                possible,
                offset,
            } => {
                let builtin: &[u8] = match kind {
                    1 => b"null",
                    2 => b"true",
                    3 => b"false",
                    _ => b"",
                };
                let mut next = 0;
                if kind != 0 {
                    if builtin.get(offset as usize) != Some(&b) {
                        return false;
                    }
                    if offset as usize + 1 == builtin.len() {
                        self.finish_value();
                    } else {
                        self.mode = Mode::Literal {
                            node,
                            kind,
                            possible,
                            offset: offset + 1,
                        };
                    }
                    return true;
                }
                if (whitespace(b) || b",}]".contains(&b))
                    && grammar.nodes[node]
                        .choices
                        .iter()
                        .enumerate()
                        .any(|(i, bytes)| {
                            possible & (1 << i) != 0 && bytes.len() == offset as usize
                        })
                {
                    self.finish_value();
                    return self.byte(b, grammar);
                }
                for (index, bytes) in grammar.nodes[node].choices.iter().enumerate() {
                    if possible & (1 << index) != 0 && bytes.get(offset as usize) == Some(&b) {
                        if offset as usize + 1 == bytes.len()
                            && !bytes
                                .first()
                                .is_some_and(|b| b.is_ascii_digit() || *b == b'-')
                        {
                            self.finish_value();
                            return true;
                        }
                        next |= 1 << index;
                    }
                }
                if next == 0 {
                    return false;
                }
                self.mode = Mode::Literal {
                    node,
                    kind,
                    possible: next,
                    offset: offset + 1,
                };
                true
            }
            Mode::String(mut string) => {
                if string.remaining > 0 {
                    if b < string.low || b > string.high {
                        return false;
                    }
                    string.remaining -= 1;
                    string.low = 0x80;
                    string.high = 0xbf;
                    if string.remaining == 0 {
                        string.chars += 1;
                    }
                } else if string.escape >= 3 {
                    let digit = match b {
                        b'0'..=b'9' => b - b'0',
                        b'a'..=b'f' => b - b'a' + 10,
                        b'A'..=b'F' => b - b'A' + 10,
                        _ => return false,
                    };
                    string.hex = (string.hex << 4) | digit as u16;
                    string.escape += 1;
                    if string.escape == 7 {
                        if string.surrogate {
                            if !(0xdc00..=0xdfff).contains(&string.hex) {
                                return false;
                            }
                            string.surrogate = false;
                            string.chars += 1;
                            string.escape = 0;
                        } else if (0xd800..=0xdbff).contains(&string.hex) {
                            string.surrogate = true;
                            string.escape = 1;
                        } else if (0xdc00..=0xdfff).contains(&string.hex) {
                            return false;
                        } else {
                            string.chars += 1;
                            string.escape = 0;
                        }
                    }
                } else if string.escape == 1 {
                    if b != b'\\' {
                        return false;
                    }
                    string.escape = 2;
                } else if string.escape == 2 {
                    if b == b'u' {
                        string.hex = 0;
                        string.escape = 3;
                    } else if !string.surrogate && b"\"\\/bfnrt".contains(&b) {
                        string.chars += 1;
                        string.escape = 0;
                    } else {
                        return false;
                    }
                } else if b == b'"' {
                    if string.key {
                        let frame = &mut self.frames[self.depth - 1];
                        frame.child = 0;
                        frame.count += 1;
                        frame.phase = 2;
                        self.mode = Mode::Normal;
                    } else {
                        self.finish_value();
                    }
                    return true;
                } else {
                    if string.chars >= string.max {
                        return false;
                    }
                    if b == b'\\' {
                        string.escape = 2;
                    } else if b < 0x20 {
                        return false;
                    } else if b < 0x80 {
                        string.chars += 1;
                    } else {
                        let (remaining, low, high) = match b {
                            0xc2..=0xdf => (1, 0x80, 0xbf),
                            0xe0 => (2, 0xa0, 0xbf),
                            0xe1..=0xec | 0xee..=0xef => (2, 0x80, 0xbf),
                            0xed => (2, 0x80, 0x9f),
                            0xf0 => (3, 0x90, 0xbf),
                            0xf1..=0xf3 => (3, 0x80, 0xbf),
                            0xf4 => (3, 0x80, 0x8f),
                            _ => return false,
                        };
                        string.remaining = remaining;
                        string.low = low;
                        string.high = high;
                    }
                }
                self.mode = Mode::String(string);
                true
            }
            Mode::Number {
                integer,
                phase,
                mut digits,
                mut bytes,
                mut exponent,
            } => {
                let accepting = matches!(phase, 2 | 3 | 5 | 8);
                if whitespace(b) || b",}]".contains(&b) {
                    if !accepting {
                        return false;
                    }
                    self.finish_value();
                    return self.byte(b, grammar);
                }
                if bytes >= 32 {
                    return false;
                }
                bytes += 1;
                let next = match (phase, b) {
                    (0, b'-') => 1,
                    (0 | 1, b'0') => {
                        digits += 1;
                        2
                    }
                    (0 | 1, b'1'..=b'9') | (3, b'0'..=b'9') => {
                        digits += 1;
                        3
                    }
                    (2 | 3, b'.') if !integer => 4,
                    (4 | 5, b'0'..=b'9') => 5,
                    (2 | 3 | 5, b'e' | b'E') if !integer => 6,
                    (6, b'+' | b'-') => 7,
                    (6..=8, b'0'..=b'9') if exponent < 2 => {
                        exponent += 1;
                        8
                    }
                    _ => return false,
                };
                if integer && digits > 18 {
                    return false;
                }
                self.mode = Mode::Number {
                    integer,
                    phase: next,
                    digits,
                    bytes,
                    exponent,
                };
                true
            }
        }
    }
}
fn whitespace(b: u8) -> bool {
    b" \n\r\t".contains(&b)
}

#[derive(Clone, Copy)]
struct TrieNode {
    child: u32,
    sibling: u32,
    terminal: u32,
    byte: u8,
}
impl TrieNode {
    fn new(byte: u8) -> Self {
        Self {
            child: NONE,
            sibling: NONE,
            terminal: NONE,
            byte,
        }
    }
}
pub struct ConstraintDecoder {
    grammar: Grammar,
    state: State,
    trie: Vec<TrieNode>,
    aliases: Vec<u32>,
    allowed: Vec<bool>,
    bytes: usize,
}
impl ConstraintDecoder {
    pub fn schema_fingerprint(&self) -> &str {
        &self.grammar.identity
    }
    pub fn capacity_bytes(&self) -> u64 {
        let mut bytes = self.trie.capacity() * std::mem::size_of::<TrieNode>()
            + self.aliases.capacity() * std::mem::size_of::<u32>()
            + self.allowed.capacity()
            + self.grammar.nodes.capacity() * std::mem::size_of::<Node>();
        for node in &self.grammar.nodes {
            bytes += node.choices.capacity() * std::mem::size_of::<Vec<u8>>();
            bytes += node.choices.iter().map(Vec::capacity).sum::<usize>();
            if let Kind::Object(Some(props), _) = &node.kind {
                bytes += props.capacity() * std::mem::size_of::<Property>();
                bytes += props
                    .iter()
                    .map(|p| p.key.capacity() + p.name.capacity())
                    .sum::<usize>();
            }
        }
        bytes as u64
    }
    pub fn new(grammar: Grammar, tokenizer: &TextTokenizer, cancel: &Cancellation) -> Result<Self> {
        Self::with_tool_framing(grammar, tokenizer, cancel, false)
    }
    pub fn with_tool_framing(
        grammar: Grammar,
        tokenizer: &TextTokenizer,
        cancel: &Cancellation,
        wrapped: bool,
    ) -> Result<Self> {
        if tokenizer.vocabulary() > 262144 {
            return Err(unsupported("constraint vocabulary exceeds 262144 tokens"));
        }
        let mut trie = Vec::new();
        trie.try_reserve_exact(MAX_TRIE)
            .map_err(|_| Error::resource("constraint trie allocation failed"))?;
        trie.push(TrieNode::new(0));
        let mut aliases = Vec::new();
        aliases
            .try_reserve_exact(tokenizer.vocabulary())
            .map_err(|_| Error::resource("constraint aliases allocation failed"))?;
        aliases.resize(tokenizer.vocabulary(), NONE);
        for (id, alias) in aliases.iter_mut().enumerate() {
            if id % 256 == 0 {
                cancel.check()?;
            }
            if (tokenizer.is_special(id as u32)
                && !(wrapped && tokenizer.is_tool_marker(id as u32)))
                || tokenizer.is_eog(id as u32)
            {
                continue;
            }
            let bytes = tokenizer.token_bytes(id as u32)?;
            if bytes.is_empty() || bytes.len() > 128 {
                continue;
            }
            let mut node = 0;
            for byte in bytes {
                let mut child = trie[node].child;
                while child != NONE && trie[child as usize].byte != *byte {
                    child = trie[child as usize].sibling;
                }
                if child == NONE {
                    if trie.len() == MAX_TRIE {
                        return Err(Error::resource(
                            "constraint tokenizer trie exceeds node limit",
                        ));
                    }
                    child = trie.len() as u32;
                    let mut entry = TrieNode::new(*byte);
                    entry.sibling = trie[node].child;
                    trie.push(entry);
                    trie[node].child = child;
                }
                node = child as usize;
            }
            *alias = trie[node].terminal;
            trie[node].terminal = id as u32;
        }
        let mut state = State::new(grammar.root);
        state.wrapped = wrapped;
        let mut allowed = Vec::new();
        allowed
            .try_reserve_exact(tokenizer.vocabulary())
            .map_err(|_| Error::resource("constraint mask allocation failed"))?;
        allowed.resize(tokenizer.vocabulary(), false);
        Ok(Self {
            grammar,
            state,
            trie,
            aliases,
            allowed,
            bytes: 0,
        })
    }
    pub fn mask(&mut self, cancel: &Cancellation) -> Result<&[bool]> {
        self.allowed.fill(false);
        let mut visited = 0;
        self.walk(self.trie[0].child, self.state, cancel, &mut visited)?;
        if !self.allowed.iter().any(|v| *v) {
            return Err(Error::new(
                ErrorCode::ConstraintDeadEnd,
                "no tokenizer token can continue the requested JSON constraint",
            ));
        }
        Ok(&self.allowed)
    }
    fn walk(
        &mut self,
        mut node: u32,
        state: State,
        cancel: &Cancellation,
        visited: &mut usize,
    ) -> Result<()> {
        while node != NONE {
            *visited += 1;
            if (*visited).is_multiple_of(256) {
                cancel.check()?;
            }
            let entry = self.trie[node as usize];
            let mut next = state;
            if next.byte(entry.byte, &self.grammar) {
                let mut terminal = entry.terminal;
                while terminal != NONE {
                    self.allowed[terminal as usize] = true;
                    terminal = self.aliases[terminal as usize];
                }
                self.walk(entry.child, next, cancel, visited)?;
            }
            node = entry.sibling;
        }
        Ok(())
    }
    pub fn advance(&mut self, bytes: &[u8]) -> Result<()> {
        self.bytes = self
            .bytes
            .checked_add(bytes.len())
            .filter(|n| *n <= MAX_OUTPUT_BYTES)
            .ok_or_else(|| Error::resource("constrained output exceeds 64 KiB"))?;
        for byte in bytes {
            if !self.state.byte(*byte, &self.grammar) {
                return Err(Error::new(
                    ErrorCode::ConstraintViolation,
                    "selected token violates native grammar",
                ));
            }
        }
        Ok(())
    }
    pub fn complete(&self) -> bool {
        self.state.complete
            && (!self.state.wrapped || self.state.suffix as usize == b"\n</tool_call>".len())
    }
    pub fn validate_output(&self, output: &str) -> Result<()> {
        let output = if self.state.wrapped {
            output
                .trim_end_matches([' ', '\n', '\r', '\t'])
                .strip_prefix("<tool_call>\n")
                .and_then(|s| s.strip_suffix("\n</tool_call>"))
                .ok_or_else(|| {
                    Error::new(
                        ErrorCode::ConstraintViolation,
                        "invalid strict tool framing",
                    )
                })?
        } else {
            output
        };
        self.grammar.validate_output(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    fn field(schema: Value) -> Grammar {
        Grammar::compile(&json!({"type":"object","properties":{"value":schema},"required":["value"],"additionalProperties":false})).unwrap()
    }
    fn accepts(grammar: &Grammar, bytes: &[u8]) -> bool {
        let mut state = State::new(grammar.root);
        bytes.iter().all(|b| state.byte(*b, grammar))
            && state.complete
            && std::str::from_utf8(bytes).is_ok_and(|text| grammar.validate_output(text).is_ok())
    }
    #[test]
    fn json_object_grammar_accepts_nested_unicode_escapes_and_rejects_invalid_bytes() {
        let grammar = ResponseFormat::JsonObject.compile().unwrap().unwrap();
        for text in [
            "{}",
            " {\"k\":[null,true,false,0,-12,1.25e+2,{\"\":\"বাংলা 日本語 🌍\"}]} \n",
            r#"{"x":"\uD83C\uDF0D\u0000\n\\\""}"#,
        ] {
            assert!(accepts(&grammar, text.as_bytes()), "{text}");
        }
        for text in [
            "[]",
            "null",
            "{",
            "{,}",
            "{\"x\":01}",
            "{\"x\":1.}",
            "{\"x\":1e}",
            "{\"x\":true,}",
            r#"{"x":"\uD800"}"#,
            r#"{"x":"\uDC00"}"#,
            r#"{"x":"\x01"}"#,
        ] {
            assert!(!accepts(&grammar, text.as_bytes()), "{text}");
        }
        for bad in [
            b"{\"x\":\"\xc0\x80\"}".as_slice(),
            b"{\"x\":\"\xed\xa0\x80\"}",
            b"{\"x\":\"\xf4\x90\x80\x80\"}",
        ] {
            assert!(!accepts(&grammar, bad));
        }
    }
    #[test]
    fn required_unknown_duplicate_and_optional_properties_are_enforced_in_any_order() {
        let grammar = Grammar::compile(&json!({"type":"object","properties":{"a":{"type":"integer"},"b":{"type":["string","null"],"maxLength":1}},"required":["a"],"additionalProperties":false})).unwrap();
        for text in [
            r#"{"a":1}"#,
            r#"{"b":null,"a":-2}"#,
            "{\"a\":0,\"b\":\"🌍\"}",
        ] {
            assert!(accepts(&grammar, text.as_bytes()), "{text}");
        }
        for text in [
            "{}",
            r#"{"a":1,"a":2}"#,
            r#"{"a":1,"c":0}"#,
            r#"{"a":1,"b":"ab"}"#,
            r#"{"a":1.5}"#,
        ] {
            assert!(!accepts(&grammar, text.as_bytes()), "{text}");
        }
        let escaped = field(json!({"type":"string","maxLength":1}));
        assert!(accepts(&escaped, br#"{"value":"\ud83c\udf0d"}"#));
        assert!(!accepts(&escaped, br#"{"value":"\ud83c\udf0dx"}"#));
    }
    #[test]
    fn enum_number_prefixes_constants_array_bounds_and_nullable_values() {
        let grammar = field(json!({"type":"integer","enum":[1,12,123,-1]}));
        for text in [
            r#"{"value":1}"#,
            r#"{"value":12}"#,
            r#"{"value":123}"#,
            r#"{"value":-1}"#,
        ] {
            assert!(accepts(&grammar, text.as_bytes()), "{text}");
        }
        assert!(!accepts(&grammar, br#"{"value":122}"#));
        let grammar = field(
            json!({"type":"array","maxItems":2,"items":{"type":["string","null"],"enum":["yes",null]}}),
        );
        for text in [r#"{"value":[]}"#, r#"{"value":["yes",null]}"#] {
            assert!(accepts(&grammar, text.as_bytes()));
        }
        for text in [
            r#"{"value":["no"]}"#,
            r#"{"value":[null,null,null]}"#,
            r#"{"value":[null,]}"#,
        ] {
            assert!(!accepts(&grammar, text.as_bytes()));
        }
        assert!(accepts(
            &field(json!({"type":"string","const":"ok"})),
            br#"{"value":"ok"}"#
        ));
        assert!(accepts(
            &field(json!({"type":"string","maxLength":0})),
            br#"{"value":""}"#
        ));
    }
    #[test]
    fn unsupported_constraints_are_never_silently_dropped() {
        for extra in [
            json!({"minimum":0}),
            json!({"pattern":".*"}),
            json!({"format":"date"}),
            json!({"$ref":"https://example.com/schema"}),
            json!({"anyOf":[]}),
            json!({"minLength":1}),
        ] {
            let mut schema = json!({"type":"object","properties":{},"additionalProperties":false});
            schema
                .as_object_mut()
                .unwrap()
                .extend(extra.as_object().unwrap().clone());
            assert_eq!(
                Grammar::compile(&schema).unwrap_err().code,
                ErrorCode::UnsupportedFeature
            );
        }
        for schema in [
            json!({"type":"object","properties":{}}),
            json!({"type":"object","properties":{},"additionalProperties":true}),
            json!({"type":"object","properties":{},"required":["missing"],"additionalProperties":false}),
            json!({"type":"array","items":{"type":"integer"}}),
        ] {
            assert!(Grammar::compile(&schema).is_err());
        }
    }
    #[test]
    fn every_prefix_of_valid_bounded_values_is_viable_at_byte_boundaries() {
        let grammar = ResponseFormat::JsonObject.compile().unwrap().unwrap();
        for n in 0..128 {
            let text = serde_json::to_string(
                &json!({"n":n,"s":format!("🌍\\\"\n{n}"),"a":[true,null,-4.25]}),
            )
            .unwrap();
            let mut state = State::new(grammar.root);
            for byte in text.bytes() {
                assert!(state.byte(byte, &grammar), "{text}");
            }
            assert!(state.complete);
            grammar.validate_output(&text).unwrap();
        }
    }
}
