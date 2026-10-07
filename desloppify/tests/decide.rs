//! System One through `decide`'s public constructors: Jev against a local
//! server speaking its API, the dry run, and none.

use std::num::NonZeroUsize;

use desloppify::agent::Usage;
use desloppify::decide::{self, Choice, Decide, Decided, JevConfig};
use serde_json::json;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::task::JoinHandle;

const API_KEY: &str = "test-key";
const MODEL: &str = "jev-test";
/// Most headers a request to the fake server may carry.
const MAX_HEADERS: usize = 64;

/// One reply the fake server sends.
struct Reply {
    status: u16,
    headers: &'static [(&'static str, &'static str)],
    body: String,
}

impl Reply {
    fn ok(body: serde_json::Value) -> Self {
        Self {
            status: 200,
            headers: &[],
            body: body.to_string(),
        }
    }

    fn status(status: u16, headers: &'static [(&'static str, &'static str)]) -> Self {
        Self {
            status,
            headers,
            body: r#"{"error": "no"}"#.into(),
        }
    }
}

/// One request the fake server read.
struct Received {
    path: String,
    authorization: Option<String>,
    body: serde_json::Value,
}

/// A server on localhost that answers one request per connection with the
/// next of `replies`, then stops; its task returns what it was sent.
async fn serve(replies: Vec<Reply>) -> (JevConfig, JoinHandle<Vec<Received>>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let config = JevConfig {
        base_url: format!("http://{}/", listener.local_addr().unwrap()),
        api_key: API_KEY.into(),
        model: MODEL.into(),
        jobs: NonZeroUsize::MIN,
    };
    let server = tokio::spawn(async move {
        let mut received = Vec::new();
        for reply in replies {
            let (mut stream, _) = listener.accept().await.unwrap();
            received.push(read_request(&mut stream).await);
            write_reply(&mut stream, &reply).await;
        }
        received
    });
    (config, server)
}

async fn read_request(stream: &mut TcpStream) -> Received {
    let mut buffer = Vec::new();
    loop {
        let mut chunk = [0_u8; 4096];
        let read = stream.read(&mut chunk).await.unwrap();
        assert!(read > 0, "the client hung up mid-request");
        buffer.extend_from_slice(&chunk[..read]);

        let mut headers = [httparse::EMPTY_HEADER; MAX_HEADERS];
        let mut request = httparse::Request::new(&mut headers);
        let httparse::Status::Complete(head) = request.parse(&buffer).unwrap() else {
            continue;
        };
        let header = |name: &str| {
            request
                .headers
                .iter()
                .find(|h| h.name.eq_ignore_ascii_case(name))
                .map(|h| String::from_utf8(h.value.to_vec()).unwrap())
        };
        let length: usize = header("content-length").map_or(0, |v| v.parse().unwrap());
        if buffer.len() < head + length {
            continue;
        }
        return Received {
            path: request.path.unwrap().to_owned(),
            authorization: header("authorization"),
            body: serde_json::from_slice(&buffer[head..head + length]).unwrap(),
        };
    }
}

async fn write_reply(stream: &mut TcpStream, reply: &Reply) {
    let mut response = format!(
        "HTTP/1.1 {} Reply\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n",
        reply.status,
        reply.body.len()
    );
    for (name, value) in reply.headers {
        response.push_str(&format!("{name}: {value}\r\n"));
    }
    response.push_str("\r\n");
    response.push_str(&reply.body);
    stream.write_all(response.as_bytes()).await.unwrap();
    stream.shutdown().await.unwrap();
}

const LABELS: [(&str, &str); 2] = [("flat", "The main path is flat."), ("nested", "It nests.")];

fn questions() -> [Choice<'static>; 2] {
    [
        Choice {
            name: "guard-clauses",
            instructions: "Is the main path nested?",
            labels: &LABELS,
        },
        Choice {
            name: "magic-numbers",
            instructions: "Are there magic numbers?",
            labels: &[("named", "All named."), ("magic", "A bare literal.")],
        },
    ]
}

fn answers() -> serde_json::Value {
    json!({
        "model": "jev-1",
        "answers": {
            "guard-clauses": {
                "type": "choice",
                "choice": "nested",
                "confidence": 0.93,
                "probabilities": {"flat": 0.07, "nested": 0.93}
            },
            "magic-numbers": {"type": "score", "score": 4}
        },
        "usage": {"input_tokens": 1234, "output_tokens": 0}
    })
}

#[tokio::test]
async fn jev_posts_every_question_as_a_choice_and_reads_back_the_choices() {
    let (config, server) = serve(vec![Reply::ok(answers())]).await;
    let decisions = decide::jev(config)
        .decide("fn f() {}", &questions())
        .await
        .unwrap();

    let received = server.await.unwrap();
    assert_eq!(received.len(), 1);
    assert_eq!(received[0].path, "/v1/systemone");
    assert_eq!(
        received[0].authorization.as_deref(),
        Some("Bearer test-key")
    );
    assert_eq!(
        received[0].body,
        json!({
            "state": "fn f() {}",
            "model": MODEL,
            "questions": {
                "guard-clauses": {
                    "type": "choice",
                    "instructions": "Is the main path nested?",
                    "criteria": {"flat": "The main path is flat.", "nested": "It nests."}
                },
                "magic-numbers": {
                    "type": "choice",
                    "instructions": "Are there magic numbers?",
                    "criteria": {"named": "All named.", "magic": "A bare literal."}
                }
            }
        })
    );

    // An answer of another type is not a choice, so not an answer.
    assert_eq!(decisions.answers.len(), 1);
    assert_eq!(
        decisions.answers["guard-clauses"],
        Decided {
            label: "nested".into(),
            confidence: 0.93
        }
    );
    assert_eq!(
        decisions.usage,
        Usage {
            calls: 1,
            input: 1234,
            output: 0
        }
    );
}

#[tokio::test]
async fn jev_retries_a_throttled_request_after_the_wait_the_server_names() {
    let (config, server) = serve(vec![
        Reply::status(429, &[("retry-after-ms", "1")]),
        Reply::status(503, &[("retry-after", "0.001")]),
        Reply::ok(answers()),
    ])
    .await;
    let decisions = decide::jev(config).decide("s", &questions()).await.unwrap();
    assert_eq!(server.await.unwrap().len(), 3);
    assert_eq!(decisions.answers["guard-clauses"].label, "nested");
}

#[tokio::test]
async fn jev_gives_up_after_two_retries_naming_the_status() {
    let (config, server) = serve(vec![
        Reply::status(500, &[("retry-after-ms", "1")]),
        Reply::status(500, &[("retry-after-ms", "1")]),
        Reply::status(500, &[("retry-after-ms", "1")]),
    ])
    .await;
    let error = decide::jev(config)
        .decide("s", &questions())
        .await
        .unwrap_err();
    assert_eq!(server.await.unwrap().len(), 3);
    assert!(format!("{error:#}").contains("500"), "{error:#}");
}

#[tokio::test]
async fn jev_does_not_retry_a_refused_request() {
    let (config, server) = serve(vec![Reply::status(401, &[])]).await;
    let error = decide::jev(config)
        .decide("s", &questions())
        .await
        .unwrap_err();
    assert_eq!(server.await.unwrap().len(), 1);
    assert!(format!("{error:#}").contains("401"), "{error:#}");
}

#[tokio::test]
async fn jev_waits_its_own_backoff_when_the_servers_wait_is_not_one() {
    let (config, server) = serve(vec![
        Reply::status(429, &[("retry-after", "-5")]),
        Reply::status(429, &[("retry-after-ms", "NaN")]),
        Reply::ok(answers()),
    ])
    .await;
    let decisions = decide::jev(config).decide("s", &questions()).await;
    assert_eq!(server.await.unwrap().len(), 3);
    assert!(decisions.is_ok());
}

#[tokio::test]
async fn jev_answering_something_other_than_answers_is_an_error() {
    let (config, server) = serve(vec![Reply::ok(json!({"oops": true}))]).await;
    let decided = decide::jev(config).decide("s", &questions()).await;
    server.await.unwrap();
    assert!(decided.is_err());
}

#[tokio::test]
async fn the_dry_run_answers_every_question_with_its_first_label_and_prices_its_input() {
    let decisions = decide::dry_run()
        .decide("fn f() {}", &questions())
        .await
        .unwrap();
    let chosen: Vec<_> = decisions
        .answers
        .iter()
        .map(|(name, d)| (name.as_str(), d.label.as_str(), d.confidence))
        .collect();
    assert_eq!(
        chosen,
        [
            ("guard-clauses", "flat", 1.0),
            ("magic-numbers", "named", 1.0)
        ]
    );
    assert_eq!((decisions.usage.calls, decisions.usage.output), (1, 0));
    assert!(decisions.usage.input > 0);
}

#[tokio::test]
async fn none_answers_nothing_and_costs_nothing() {
    let decisions = decide::none().decide("s", &questions()).await.unwrap();
    assert!(decisions.answers.is_empty());
    assert_eq!(decisions.usage, Usage::default());
}
