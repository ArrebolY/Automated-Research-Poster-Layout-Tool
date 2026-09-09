import http.client
from unittest.mock import patch

from postergen.config import LLMConfig
from postergen.llm import OpenAICompatibleTextGenerator


class RecordingGenerator(OpenAICompatibleTextGenerator):
    def _api_key(self) -> str:
        return "test-key"

    def _post_json(self, url: str, payload: dict, headers: dict[str, str]) -> dict:
        self.recorded_payload = payload
        return {"choices": [{"message": {"content": "[1, 2]"}}]}


class EmptyResponseGenerator(RecordingGenerator):
    def _post_json(self, url: str, payload: dict, headers: dict[str, str]) -> dict:
        return {"choices": [{"message": {"content": "", "reasoning_content": "internal reasoning"}}]}


def test_deepseek_requests_disable_thinking_for_structured_tasks():
    generator = RecordingGenerator(LLMConfig(provider="deepseek", model="deepseek-v4-flash"))

    response = generator.generate("Select sections")

    assert response == "[1, 2]"
    assert generator.recorded_payload["thinking"] == {"type": "disabled"}


def test_openai_compatible_generator_rejects_empty_final_content():
    generator = EmptyResponseGenerator(LLMConfig(provider="deepseek", model="deepseek-v4-flash"))

    try:
        generator.generate("Select sections")
    except RuntimeError as exc:
        assert "empty final response" in str(exc)
    else:
        raise AssertionError("Expected an empty LLM response to raise RuntimeError")


def test_openai_compatible_generator_retries_remote_disconnect():
    generator = OpenAICompatibleTextGenerator(LLMConfig(provider="deepseek", model="deepseek-v4-flash"))
    calls = {"count": 0}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def read(self):
            return b'{"choices":[{"message":{"content":"ok"}}]}'

    def fake_urlopen(_request, timeout):
        calls["count"] += 1
        if calls["count"] == 1:
            raise http.client.RemoteDisconnected("closed")
        return Response()

    with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "test-key"}), patch(
        "urllib.request.urlopen",
        side_effect=fake_urlopen,
    ), patch("time.sleep"):
        response = generator.generate("hello")

    assert response == "ok"
    assert calls["count"] == 2
