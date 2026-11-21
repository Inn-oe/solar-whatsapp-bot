"""A model trained on the owner's PC (train_local.py) is saved to a file and loaded by the bot instead of training."""
import pytest

import brain
import db
import nlu

EXAMPLES = ([("greet", t) for t in ["hello there", "hi good morning", "hey how are you", "good afternoon"]] +
            [("price", t) for t in ["how much is a 5kva system", "what does a solar kit cost", "price of panels",
                                    "how much for batteries"]])


def _trained(tmp_path):
    eng = nlu.Engine()
    eng.train(EXAMPLES, backend="tfidf")
    path = tmp_path / "model.pkl"
    meta = eng.save(path, digest=nlu.examples_digest(EXAMPLES))
    return path, meta


def test_saved_model_answers_the_same_after_loading(tmp_path):
    path, meta = _trained(tmp_path)
    fresh = nlu.Engine()
    info = fresh.load(path, digest=nlu.examples_digest(EXAMPLES))
    assert info["trained"] and info["pretrained"] and info["in_sync"]
    assert (info["examples"], info["intents"]) == (meta["examples"], 2)
    assert fresh.predict("how much is a solar system")["intent"] == "price"
    assert fresh.predict("hi good morning")["intent"] == "greet"


def test_it_says_when_the_phrases_changed_after_training(tmp_path):
    path, _ = _trained(tmp_path)
    changed = EXAMPLES + [("price", "is there a discount")]
    assert nlu.Engine().load(path, digest=nlu.examples_digest(changed))["in_sync"] is False


def test_a_model_from_another_scikit_learn_is_refused(tmp_path, monkeypatch):
    path, _ = _trained(tmp_path)
    import sklearn
    monkeypatch.setattr(sklearn, "__version__", "0.0.1")
    with pytest.raises(ValueError, match="scikit-learn"):
        nlu.Engine().load(path)


def test_bot_loads_the_file_instead_of_training(tmp_path, monkeypatch):
    path, _ = _trained(tmp_path)
    monkeypatch.setattr(brain, "MODEL_FILE", path)
    monkeypatch.setenv("NLU_PRETRAINED", "auto")
    def boom(*a, **k):
        raise AssertionError("must not train when a model file is present")
    monkeypatch.setattr(nlu.engine, "train", boom)
    info = brain.retrain()
    assert info["pretrained"] and info["trained"]
    assert "changed after this model was trained" in info["note"]    # the test database has other phrases


def test_bot_trains_when_the_file_is_missing_or_switched_off(tmp_path, monkeypatch):
    monkeypatch.setattr(brain, "MODEL_FILE", tmp_path / "none.pkl")
    monkeypatch.setenv("NLU_PRETRAINED", "auto")
    assert brain.retrain().get("pretrained") is None
    path, _ = _trained(tmp_path)
    monkeypatch.setattr(brain, "MODEL_FILE", path)
    monkeypatch.setenv("NLU_PRETRAINED", "off")
    assert brain.retrain().get("pretrained") is None


def test_a_broken_file_falls_back_to_training(tmp_path, monkeypatch):
    bad = tmp_path / "bad.pkl"
    bad.write_bytes(b"not a model")
    monkeypatch.setattr(brain, "MODEL_FILE", bad)
    monkeypatch.setenv("NLU_PRETRAINED", "auto")
    assert brain.retrain()["trained"]
