//! HTTPS-only acquisition with validated, pinned DNS addresses on every redirect.
use crate::error::{ApiError, Result};
use reqwest::{Url, redirect::Policy};
use std::{
    net::{IpAddr, SocketAddr},
    time::Duration,
};

pub fn public_address(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(a) => {
            let [x, y, z, _] = a.octets();
            !(a.is_private()
                || a.is_loopback()
                || a.is_link_local()
                || a.is_unspecified()
                || a.is_multicast()
                || a.is_broadcast()
                || x == 0
                || x >= 240
                || (x == 100 && (64..=127).contains(&y))
                || (x == 192 && y == 0)
                || (x == 192 && y == 88 && z == 99)
                || (x == 198 && (y == 18 || y == 19))
                || (x == 198 && y == 51 && z == 100)
                || (x == 203 && y == 0 && z == 113))
        }
        IpAddr::V6(a) => {
            if let Some(a) = a.to_ipv4_mapped() {
                return public_address(IpAddr::V4(a));
            }
            let o = a.octets();
            (o[0] & 0xe0) == 0x20
                && !(o[0] == 0x20 && o[1] == 2)
                && !(o[0] == 0x20 && o[1] == 1 && o[2] < 2)
                && !(o[0] == 0x20 && o[1] == 1 && o[2] == 0x0d && o[3] == 0xb8)
        }
    }
}
fn validate_url(url: &Url) -> Result<()> {
    if url.scheme() != "https"
        || !url.username().is_empty()
        || url.password().is_some()
        || url.fragment().is_some()
        || url.as_str().len() > 4096
        || url.port_or_known_default() != Some(443)
    {
        return Err(ApiError::bad(
            "model downloads require HTTPS on port 443 without URL credentials or fragments",
        ));
    }
    Ok(())
}
pub async fn response(raw: &str) -> Result<reqwest::Response> {
    response_with_limits(raw, 5, 600).await
}
pub async fn response_with_limits(
    raw: &str,
    redirects: usize,
    timeout_seconds: u64,
) -> Result<reqwest::Response> {
    if redirects > 5 || timeout_seconds == 0 || timeout_seconds > 600 {
        return Err(ApiError::bad("invalid finite acquisition policy"));
    }
    let mut url = Url::parse(raw).map_err(|_| ApiError::bad("invalid download URL"))?;
    for _ in 0..=redirects {
        validate_url(&url)?;
        let host = url
            .host_str()
            .ok_or_else(|| ApiError::bad("download URL needs a host"))?
            .to_owned();
        let addresses: Vec<SocketAddr> = tokio::time::timeout(
            Duration::from_secs(10),
            tokio::net::lookup_host((host.as_str(), 443)),
        )
        .await
        .map_err(|_| ApiError::bad("download DNS timeout"))?
        .map_err(|_| ApiError::bad("download DNS resolution failed"))?
        .take(65)
        .collect();
        if addresses.is_empty()
            || addresses.len() > 64
            || addresses.iter().any(|a| !public_address(a.ip()))
        {
            return Err(ApiError::bad(
                "downloads cannot target private, local, reserved or unspecified addresses",
            ));
        }
        let client = reqwest::Client::builder()
            .no_proxy()
            .redirect(Policy::none())
            .connect_timeout(Duration::from_secs(10))
            .timeout(Duration::from_secs(timeout_seconds))
            .resolve_to_addrs(&host, &addresses)
            .build()
            .map_err(|_| ApiError::bad("download client initialization failed"))?;
        let result = client
            .get(url.clone())
            .send()
            .await
            .map_err(|_| ApiError::bad("HTTPS model download failed"))?;
        if result.status().is_redirection() {
            let location = result
                .headers()
                .get(reqwest::header::LOCATION)
                .and_then(|v| v.to_str().ok())
                .ok_or_else(|| ApiError::bad("download redirect has no valid location"))?;
            url = url
                .join(location)
                .map_err(|_| ApiError::bad("invalid download redirect"))?;
            continue;
        }
        if !result.status().is_success() {
            return Err(ApiError::bad(format!(
                "download returned HTTP {}",
                result.status().as_u16()
            )));
        }
        return Ok(result);
    }
    Err(ApiError::bad("too many model download redirects"))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rejects_local_and_reserved_networks() {
        for ip in [
            "127.0.0.1",
            "10.0.0.1",
            "169.254.169.254",
            "100.64.0.1",
            "198.18.0.1",
            "192.0.2.1",
            "203.0.113.4",
            "224.0.0.1",
            "::1",
            "::ffff:127.0.0.1",
            "fc00::1",
            "fe80::1",
            "2001:db8::1",
        ] {
            assert!(!public_address(ip.parse().unwrap()), "{ip}");
        }
        for ip in ["1.1.1.1", "8.8.8.8", "2606:4700:4700::1111"] {
            assert!(public_address(ip.parse().unwrap()));
        }
    }
}
