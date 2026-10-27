//! Privileged credentials are stored locally, never returned by HTTP or logs.
use crate::error::{ApiError, Result};
use axum::http::{HeaderMap, StatusCode, header::AUTHORIZATION};
use coheryn_engine::{Error, ErrorCode};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    fs::{File, OpenOptions},
    io::{Read, Write},
    path::{Path, PathBuf},
};
use subtle::ConstantTimeEq;

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Credentials {
    pub inference_token: String,
    pub admin_token: String,
}
pub fn credential_path(root: &Path) -> PathBuf {
    root.join("credentials.json")
}
pub fn private_directory(path: &Path) -> std::io::Result<()> {
    let existed = path.exists();
    std::fs::create_dir_all(path)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if existed {
            if std::fs::metadata(path)?.permissions().mode() & 0o077 != 0 {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::PermissionDenied,
                    "storage directory must be private (mode 0700)",
                ));
            }
        } else {
            std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o700))?;
        }
    }
    #[cfg(not(unix))]
    let _ = existed;
    Ok(())
}
pub fn private_file(path: &Path) -> std::io::Result<File> {
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    options.open(path)
}
pub fn read_credentials(root: &Path) -> coheryn_engine::Result<Credentials> {
    let file = File::open(credential_path(root))?;
    if file.metadata()?.len() > 16384 {
        return Err(Error::request("credentials file is too large"));
    }
    let result: Credentials = serde_json::from_reader(file.take(16385))
        .map_err(|_| Error::request("invalid credentials file"))?;
    result.validate()?;
    Ok(result)
}
impl Credentials {
    pub fn validate(&self) -> coheryn_engine::Result<()> {
        for value in [&self.inference_token, &self.admin_token] {
            if !(32..=4096).contains(&value.len())
                || !value.is_ascii()
                || value
                    .bytes()
                    .any(|b| b.is_ascii_whitespace() || b.is_ascii_control())
            {
                return Err(Error::request(
                    "credentials must be 32..=4096 non-whitespace ASCII bytes",
                ));
            }
        }
        if self.inference_token == self.admin_token {
            return Err(Error::request(
                "inference and administrative credentials must differ",
            ));
        }
        Ok(())
    }
    pub fn load_or_create(root: &Path) -> coheryn_engine::Result<Self> {
        private_directory(root)?;
        let path = credential_path(root);
        let disk = if path.exists() {
            read_credentials(root)?
        } else {
            let token = || {
                format!(
                    "{}{}",
                    uuid::Uuid::new_v4().simple(),
                    uuid::Uuid::new_v4().simple()
                )
            };
            let c = Self {
                inference_token: token(),
                admin_token: token(),
            };
            let mut file = private_file(&path)?;
            file.write_all(&serde_json::to_vec(&c).map_err(|_| {
                Error::new(ErrorCode::Internal, "credential serialization failed")
            })?)?;
            file.sync_all()?;
            c
        };
        let result = Self {
            inference_token: std::env::var("COHERYN_API_TOKEN").unwrap_or(disk.inference_token),
            admin_token: std::env::var("COHERYN_ADMIN_TOKEN").unwrap_or(disk.admin_token),
        };
        result.validate()?;
        Ok(result)
    }
}
#[derive(Clone)]
pub struct Auth {
    inference: [u8; 32],
    admin: [u8; 32],
}
impl Auth {
    /// Credentials define isolated ephemeral principals. Rotation revokes access rather
    /// than silently transferring histories; durable principal management is separate.
    pub fn principal(&self, headers: &HeaderMap) -> Result<String> {
        self.authorize(headers, false)?;
        let token = headers
            .get(AUTHORIZATION)
            .and_then(|h| h.to_str().ok())
            .and_then(|h| h.strip_prefix("Bearer "))
            .ok_or_else(|| ApiError::bad("missing authenticated principal"))?;
        Ok(coheryn_engine::storage::digest_hex(
            Sha256::digest(token.as_bytes()).into(),
        ))
    }
    pub fn new(credentials: &Credentials) -> coheryn_engine::Result<Self> {
        credentials.validate()?;
        Ok(Self {
            inference: Sha256::digest(credentials.inference_token.as_bytes()).into(),
            admin: Sha256::digest(credentials.admin_token.as_bytes()).into(),
        })
    }
    pub fn authorize(&self, headers: &HeaderMap, administrative: bool) -> Result<()> {
        let values: Vec<_> = headers.get_all(AUTHORIZATION).iter().collect();
        let token = if values.len() == 1 {
            values[0]
                .to_str()
                .ok()
                .and_then(|v| v.strip_prefix("Bearer "))
        } else {
            None
        };
        let token = token.filter(|v| v.len() <= 4096).ok_or_else(|| {
            ApiError::new(
                StatusCode::UNAUTHORIZED,
                ErrorCode::Unauthorized,
                "a Bearer credential is required",
            )
        })?;
        let digest: [u8; 32] = Sha256::digest(token.as_bytes()).into();
        let admin = bool::from(digest.ct_eq(&self.admin));
        let inference = bool::from(digest.ct_eq(&self.inference));
        if admin || (!administrative && inference) {
            return Ok(());
        }
        Err(ApiError::new(
            if inference {
                StatusCode::FORBIDDEN
            } else {
                StatusCode::UNAUTHORIZED
            },
            ErrorCode::Unauthorized,
            if inference {
                "administrative credentials required"
            } else {
                "invalid credentials"
            },
        ))
    }
}
