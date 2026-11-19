"""The AI step through Claude (Anthropic's Messages API), with the network replaced by a fake."""
import json

import llm

INTENTS = {"buy_system": {"category": "purchase", "enabled": True}, "price": {"category": "purchase", "enabled": True},
           "maintenance_fault": {"category": "maintenance", "enabled": True}}
EXAMPLES = [("buy_system", "i want to buy a solar system"), ("price", "how much is a 5kva system"),
            ("maintenance_fault", "my inverter is beeping")]


class Reply:
    def __init__(self, status=200, payload=None, text=""):
        self.status_code, self._payload, self.text = status, payload, text or json.dumps(payload or {})

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def claude_says(text):
    return Reply(payload={"content": [{"type": "text", "text": text}], "stop_reason": "end_turn"})


def use_claude(monkeypatch, answer, seen):
    import requests
    monkeypatch.setenv("LLM_API_KEY", "sk-ant-test-key")
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("LLM_API_URL", raising=False)

    def post(url, json=None, headers=None, timeout=None):
        seen.append({"url": url, "body": json, "headers": headers})
        return answer if isinstance(answer, Reply) else claude_says(answer)
    monkeypatch.setattr(requests, "post", post)
    llm.set_menu(INTENTS, EXAMPLES)


def test_a_claude_key_selects_claude_with_haiku(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("LLM_API_URL", raising=False)
    monkeypatch.setenv("LLM_API_KEY", "sk-ant-abc")
    assert llm.provider() == "anthropic" and llm.model_name() == "claude-haiku-5-5"
    monkeypatch.setenv("LLM_MODEL", "claude-sonnet-5-5")
    assert llm.model_name() == "claude-sonnet-5-5"
    monkeypatch.setenv("LLM_API_KEY", "some-mistral-key")
    assert llm.provider() == "openai"
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    assert llm.provider() == "anthropic"                         # can be forced
    monkeypatch.setenv("LLM_PROVIDER", "")
    monkeypatch.setenv("LLM_API_URL", "https://api.anthropic.com/v1")
    assert llm.provider() == "anthropic"                         # or recognised from the address


def test_classify_calls_the_messages_api_the_way_anthropic_expects(monkeypatch):
    seen = []
    use_claude(monkeypatch, 'Sure: {"intent": "price", "confidence": 0.93}', seen)
    assert llm.classify("what would a 3kva system with two batteries set me back", "How can I help?") == ("price", 0.93)
    call = seen[0]
    assert call["url"] == "https://api.anthropic.com/v1/messages"
    assert call["headers"]["x-api-key"] == "sk-ant-test-key" and call["headers"]["anthropic-version"] == "2023-06-01"
    body = call["body"]
    assert body["model"] == "claude-haiku-5-5" and body["max_tokens"] == 60 and body["temperature"] == 0
    system = body["system"][0]
    assert "buy_system" in system["text"] and "maintenance_fault" in system["text"]      # the intent list
    assert system["cache_control"] == {"type": "ephemeral"}                              # kept warm between calls
    user = body["messages"][0]["content"]
    assert "what would a 3kva system" in user and "How can I help?" in user and "buy_system" not in user


def test_an_unknown_intent_from_claude_is_ignored(monkeypatch):
    use_claude(monkeypatch, '{"intent": "make_coffee", "confidence": 1}', [])
    assert llm.classify("make me coffee") == (None, 0.0)


def test_a_wrong_key_is_reported_on_the_status_and_never_breaks_the_bot(monkeypatch):
    seen = []
    use_claude(monkeypatch, Reply(401, text='{"type":"error","error":{"type":"authentication_error","message":"invalid x-api-key"}}'), seen)
    before = llm.status()["errors"]
    assert llm.classify("anything new at all") == (None, 0.0)
    status = llm.status()
    assert status["errors"] == before + 1 and "401" in status["last_error"] and "invalid x-api-key" in status["last_error"]
    assert status["provider"] == "anthropic" and status["model"] == "claude-haiku-5-5"


def test_answers_from_company_facts_use_claude_and_keep_the_numbers_honest(monkeypatch):
    seen = []
    facts = "Our 5kVA system is USD 1,985 installed. A site visit is free."
    use_claude(monkeypatch, "A 5kVA system is USD 1,985 installed, and the site visit is free.", seen)
    monkeypatch.setenv("LLM_ANSWERS", "on")
    out = llm.answer_from_facts("how much is the big one and do you charge to come", facts, "GieBee Engineering")
    assert out and "1,985" in out
    assert seen[0]["url"].endswith("/v1/messages") and "COMPANY FACTS" in seen[0]["body"]["messages"][0]["content"]
    assert "cache_control" not in seen[0]["body"]["system"][0]
    use_claude(monkeypatch, "It is USD 1,200 installed.", [])                 # a number that is not in the facts
    assert llm.answer_from_facts("how much", facts, "GieBee Engineering") is None


def test_the_other_providers_still_work_as_before(monkeypatch):
    import requests
    seen = []
    monkeypatch.setenv("LLM_API_KEY", "plain-key")
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("LLM_API_URL", raising=False)

    def post(url, json=None, headers=None, timeout=None):
        seen.append((url, headers))
        return Reply(payload={"choices": [{"message": {"content": '{"intent": "buy_system", "confidence": 0.8}'}}]})
    monkeypatch.setattr(requests, "post", post)
    llm.set_menu(INTENTS, EXAMPLES)
    assert llm.classify("i would like solar please") == ("buy_system", 0.8)
    assert seen[0][0].endswith("/chat/completions") and seen[0][1]["Authorization"] == "Bearer plain-key"
