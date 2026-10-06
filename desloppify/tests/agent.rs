//! `agent::classify` reads rig's provider errors as rate signals.

use std::time::Duration;

use desloppify::agent::classify;
use desloppify::rate_limit::{BoxError, Signal};
use rig_core::ProviderError;

fn reply(status: u16, retry_after: Option<&str>) -> BoxError {
    let status = http::StatusCode::from_u16(status).unwrap();
    let headers = retry_after.map(|value| {
        let mut headers = http::HeaderMap::new();
        headers.insert("retry-after", value.parse().unwrap());
        headers
    });
    Box::new(ProviderError::from_http_response(status, "").with_response_headers(headers))
}

#[test]
fn a_429_is_a_throttle_carrying_its_retry_after_seconds() {
    assert_eq!(
        classify(&*reply(429, Some("7"))),
        Signal::Throttled {
            retry_after: Some(Duration::from_secs(7))
        }
    );
}

#[test]
fn a_retry_after_date_is_ignored() {
    assert_eq!(
        classify(&*reply(429, Some("Wed, 21 Oct 2015 07:28:00 GMT"))),
        Signal::Throttled { retry_after: None }
    );
}

#[test]
fn an_outage_is_a_failure_not_a_throttle() {
    assert_eq!(
        classify(&*reply(503, Some("3"))),
        Signal::Failed {
            retry_after: Some(Duration::from_secs(3))
        }
    );
}

#[test]
fn an_error_that_is_not_the_providers_is_a_failure() {
    let other: BoxError = Box::new(std::io::Error::other("reset"));
    assert_eq!(classify(&*other), Signal::Failed { retry_after: None });
}
