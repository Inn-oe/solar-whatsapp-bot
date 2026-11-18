"""The large-language-model fallback (llm.py), with the provider's API replaced by a fake."""
import json

import llm

INTENTS = {"buy_system": {"category": "purchase", "enabled": True}, "deny": {"category": "other", "enabled": True},
           "maintenance_fault": {"category": "maintenance", "enabled": True}}
EXAMPLES = [("buy_system", "i want to buy a solar system"), ("deny", "no"), ("maintenance_fault", "my inverter beeps")]


class FakeResponse:
    def __init__(self, content):
        self._content = content

    def raise_for_status(self):
        pass

    def json(self):
        return {"choices": [{"message": {"content": self._content}}]}


def fake_post(answer, seen):
    def post(url, json=None, timeout=None, headers=None):
        seen.append({"url": url, "body": json, "headers": headers})
        return FakeResponse(answer)
    return post


def setup(monkeypatch, answer, seen):
    import requests
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setattr(requests, "post", fake_post(answer, seen))
    llm.set_menu(INTENTS, EXAMPLES)


def test_off_without_a_key(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    llm.set_menu(INTENTS, EXAMPLES)
    assert llm.classify("no i need new system") == (None, 0.0)


def test_picks_a_known_intent_and_sends_the_context(monkeypatch):
    seen = []
    setup(monkeypatch, '{"intent": "buy_system", "confidence": 0.9}', seen)
    assert llm.classify("No I need new system", "Do you need help with a system you already have?") == ("buy_system", 0.9)
    prompt = seen[0]["body"]["messages"][1]["content"]
    assert "buy_system" in prompt and "already have" in prompt and seen[0]["url"].endswith("/chat/completions")
    assert seen[0]["headers"]["Authorization"] == "Bearer k"


def test_unknown_or_none_intent_is_ignored(monkeypatch):
    setup(monkeypatch, '{"intent": "make_coffee", "confidence": 1}', [])
    assert llm.classify("make me coffee") == (None, 0.0)
    setup(monkeypatch, 'Sure! {"intent": "none", "confidence": 0.2}', [])
    assert llm.classify("blorp") == (None, 0.0)


def test_api_errors_never_break_the_bot(monkeypatch):
    import requests

    def boom(*a, **k):
        raise requests.ConnectionError("down")
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setattr(requests, "post", boom)
    llm.set_menu(INTENTS, EXAMPLES)
    assert llm.classify("something new") == (None, 0.0)
    assert llm.status()["errors"] >= 1


def test_brain_uses_the_llm_when_the_quick_model_is_unsure(monkeypatch):
    import brain
    import nlu
    seen = []
    setup(monkeypatch, json.dumps({"intent": "maintenance_fault", "confidence": 0.8}), seen)
    monkeypatch.setattr(nlu.engine, "understood", lambda res, **k: False)   # force "unsure"
    monkeypatch.setitem(brain._intents, "maintenance_fault", {"category": "maintenance", "priority": "high",
                                                              "handler": "maintenance", "replies": []})
    a = brain.analyze("the thing on the wall makes a noise and the lights are off", context="How can I help?")
    assert a["intent"] == "maintenance_fault" and a["via"] == "llm" and a["confidence"] < 0.4
    assert "How can I help?" in seen[0]["body"]["messages"][1]["content"]
