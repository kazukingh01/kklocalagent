use std::net::SocketAddr;
use std::time::Duration;

use audio_io::{build_router, build_state, config::Config};
use tokio::net::TcpListener;

async fn spawn_test_server() -> SocketAddr {
    let mut config = Config::default();
    config.server.port = 0;
    config.runtime.autostart = false;
    let state = build_state(config);
    let app = build_router(state);

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        let _ = axum::serve(listener, app).await;
    });
    // Give the runtime a tick to fully register the server.
    tokio::time::sleep(Duration::from_millis(20)).await;
    addr
}

#[tokio::test]
async fn health_returns_ok() {
    let addr = spawn_test_server().await;
    let client = reqwest::Client::new();
    let resp = client
        .get(format!("http://{addr}/health"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["status"], "ok");
}

#[tokio::test]
async fn stop_is_idempotent_without_services() {
    let addr = spawn_test_server().await;
    let client = reqwest::Client::new();
    for _ in 0..3 {
        let resp = client
            .post(format!("http://{addr}/stop"))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
    }
}

#[tokio::test]
async fn spk_stop_without_playback_is_ok() {
    let addr = spawn_test_server().await;
    let client = reqwest::Client::new();
    let resp = client
        .post(format!("http://{addr}/spk/stop"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
}

#[tokio::test]
async fn ws_spk_closes_when_playback_not_running() {
    use futures_util::StreamExt;
    use tokio_tungstenite::connect_async;

    let addr = spawn_test_server().await;
    let url = format!("ws://{addr}/spk");
    let (mut ws, _resp) = connect_async(&url).await.unwrap();
    // Server should close the socket immediately because playback is not running.
    let msg = ws.next().await;
    assert!(
        matches!(
            msg,
            Some(Ok(tokio_tungstenite::tungstenite::Message::Close(_))) | None
        ),
        "expected close, got {msg:?}"
    );
}
