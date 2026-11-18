"""The remote-embeddings backend (NLU_BACKEND=remote), with the embedding service replaced by a fake."""
import hashlib

import numpy as np
import pytest

import nlu

EXAMPLES = [("price", "how much does a system cost"), ("price", "what are your prices"),
            ("price", "price of a 5kva system"), ("fault", "my inverter is beeping"),
            ("fault", "the inverter shows an error"), ("fault", "my solar stopped working")]


def fake_embed(texts):
    """Deterministic bag-of-words vectors: similar sentences land close together."""
    out = np.zeros((len(texts), 64), dtype=np.float32)
    for row, t in enumerate(texts):
        for w in t.lower().split():
            out[row, int(hashlib.md5(w.encode()).hexdigest(), 16) % 64] += 1
    return nlu.normalize(out)


@pytest.fixture
def remote(monkeypatch):
    monkeypatch.setenv("NLU_REMOTE_URL", "https://example.test")
    monkeypatch.setenv("NLU_REMOTE_KEY", "k")
    monkeypatch.setattr(nlu._Remote, "_post", lambda self, texts, timeout: fake_embed(texts))


def test_remote_backend_trains_and_predicts(remote):
    eng = nlu.Engine()
    info = eng.train(EXAMPLES, backend="remote")
    assert info["backend"] == "remote"
    assert eng.predict("how much is a system")["intent"] == "price"
    assert eng.predict("my inverter is beeping")["backend"] == "remote"


def test_remote_failure_falls_back_to_the_quick_model(remote, monkeypatch):
    eng = nlu.Engine()
    eng.train(EXAMPLES, backend="tfidf")
    eng.train(EXAMPLES, backend="remote")

    def down(self, texts, timeout):
        raise ConnectionError("space is asleep")
    monkeypatch.setattr(nlu._Remote, "_post", down)
    res = eng.predict("my inverter is beeping")
    assert res["intent"] == "fault" and res["backend"] == "tfidf"


def test_remote_without_url_uses_the_quick_model(monkeypatch):
    monkeypatch.delenv("NLU_REMOTE_URL", raising=False)
    info = nlu.Engine().train(EXAMPLES, backend="remote")
    assert info["backend"] == "tfidf" and "unavailable" in info["note"]
