use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ErrorCode {
    Io,
    InvalidGguf,
    UnsupportedGgufVersion,
    UnsupportedGgufTensorType,
    UnsupportedModelArchitecture,
    UnsupportedTokenizer,
    UnsupportedChatTemplate,
    UnsupportedFeature,
    UnsupportedEmbeddingDimensions,
    EmbeddingInputLimitExceeded,
    EmbeddingBatchLimitExceeded,
    ResourceExhausted,
    InvalidRequest,
    InvalidImage,
    UnsupportedImageFormat,
    UnsupportedImageProcessor,
    ImageInputLimitExceeded,
    ContextLimitExceeded,
    ConstraintDeadEnd,
    ConstraintIncomplete,
    ConstraintViolation,
    Cancelled,
    DeadlineExceeded,
    ModelChanged,
    Internal,
    Unauthorized,
    NotFound,
    Conflict,
    QueueFull,
    Unavailable,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Error {
    pub code: ErrorCode,
    pub message: String,
}

impl Error {
    pub fn new(code: ErrorCode, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }
    pub fn invalid(message: impl Into<String>) -> Self {
        Self::new(ErrorCode::InvalidGguf, message)
    }
    pub fn resource(message: impl Into<String>) -> Self {
        Self::new(ErrorCode::ResourceExhausted, message)
    }
    pub fn request(message: impl Into<String>) -> Self {
        Self::new(ErrorCode::InvalidRequest, message)
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}: {}", self.code, self.message)
    }
}
impl std::error::Error for Error {}
impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Self::new(ErrorCode::Io, e.to_string())
    }
}
pub type Result<T> = std::result::Result<T, Error>;
