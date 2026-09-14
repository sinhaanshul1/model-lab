from fastapi.testclient import TestClient
import json

from modellab.api.main import app


client = TestClient(app)


def test_mock_chat_completion_is_deterministic() -> None:
    request = {
        "model": "modellab-mock",
        "messages": [{"role": "user", "content": "Summarize this test."}],
    }

    first_response = client.post("/v1/mock-model/chat/completions", json=request)
    second_response = client.post("/v1/mock-model/chat/completions", json=request)

    assert first_response.status_code == 200
    assert first_response.json() == second_response.json()
    assert first_response.json()["object"] == "chat.completion"
    assert "Summarize this test." in first_response.json()["choices"][0]["message"]["content"]


def test_mock_chat_completion_supports_openai_compatible_streaming() -> None:
    response = client.post(
        "/v1/mock-model/chat/completions",
        json={
            "model": "modellab-mock",
            "messages": [{"role": "user", "content": "Stream this test."}],
            "stream": True,
            "stream_options": {"include_usage": True},
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    data_lines = [line.removeprefix("data: ") for line in response.text.splitlines() if line]
    assert data_lines[-1] == "[DONE]"
    events = [json.loads(line) for line in data_lines[:-1]]
    assert any(
        choice.get("delta", {}).get("content")
        for event in events
        for choice in event["choices"]
    )
    assert events[-1]["usage"]["completion_tokens"] > 0
