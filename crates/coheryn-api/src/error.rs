use axum::{
    Json,
    http::StatusCode,
    response::{IntoResponse, Response},
};
use coheryn_engine::{Error, ErrorCode};
use serde_json::{Value, json};
use std::fmt;

#[derive(Debug, Clone)]
pub struct ApiError {
    pub status: StatusCode,
    pub error: Error,
    pub details: Option<Value>,
}
pub type Result<T> = std::result::Result<T, ApiError>;
impl ApiError {
    pub fn new(status: StatusCode, code: ErrorCode, message: impl Into<String>) -> Self {
        Self {
            status,
            error: Error::new(code, message),
            details: None,
        }
    }
    pub fn bad(message: impl Into<String>) -> Self {
        Error::request(message).into()
    }
    pub fn conflict(message: impl Into<String>) -> Self {
        Self::new(StatusCode::CONFLICT, ErrorCode::Conflict, message)
    }
    pub fn missing(message: impl Into<String>) -> Self {
        Self::new(StatusCode::NOT_FOUND, ErrorCode::NotFound, message)
    }
    pub fn body(&self) -> Value {
        json!({"error":{"message":self.error.message,"type":if self.status.is_server_error(){"server_error"}else{"invalid_request_error"},"code":self.error.code,"param":null,"details":self.details}})
    }
}
impl From<Error> for ApiError {
    fn from(error: Error) -> Self {
        let status = match error.code {
            ErrorCode::InvalidRequest
            | ErrorCode::InvalidGguf
            | ErrorCode::UnsupportedGgufVersion
            | ErrorCode::UnsupportedGgufTensorType
            | ErrorCode::UnsupportedModelArchitecture
            | ErrorCode::UnsupportedTokenizer
            | ErrorCode::UnsupportedChatTemplate
            | ErrorCode::UnsupportedFeature
            | ErrorCode::UnsupportedEmbeddingDimensions => StatusCode::BAD_REQUEST,
            ErrorCode::EmbeddingInputLimitExceeded | ErrorCode::EmbeddingBatchLimitExceeded => {
                StatusCode::PAYLOAD_TOO_LARGE
            }
            ErrorCode::ResourceExhausted | ErrorCode::Unavailable => {
                StatusCode::SERVICE_UNAVAILABLE
            }
            ErrorCode::QueueFull => StatusCode::TOO_MANY_REQUESTS,
            ErrorCode::Unauthorized => StatusCode::UNAUTHORIZED,
            ErrorCode::NotFound => StatusCode::NOT_FOUND,
            ErrorCode::Conflict | ErrorCode::ModelChanged => StatusCode::CONFLICT,
            ErrorCode::Cancelled => StatusCode::REQUEST_TIMEOUT,
            _ => StatusCode::INTERNAL_SERVER_ERROR,
        };
        Self {
            status,
            error,
            details: None,
        }
    }
}
impl From<std::io::Error> for ApiError {
    fn from(e: std::io::Error) -> Self {
        Error::from(e).into()
    }
}
impl From<tokio::task::JoinError> for ApiError {
    fn from(_: tokio::task::JoinError) -> Self {
        Error::new(ErrorCode::Internal, "background operation failed").into()
    }
}
impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.status, Json(self.body())).into_response()
    }
}
impl fmt::Display for ApiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.error)
    }
}
impl std::error::Error for ApiError {}
