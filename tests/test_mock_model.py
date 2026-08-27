from fastapi.testclient import TestClient

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
