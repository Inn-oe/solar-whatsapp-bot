"""Quotations learned from GieBee's own quotations, and the language model's guard rails."""
import json

import brain
import llm
import quote
from training import build_quote_book


def test_quotation_adds_up_and_reads_like_giebee():
    q = quote.build(5)
    assert q["size_kva"] == 5 and q["total"] == sum(l["qty"] * l["unit"] for l in q["lines"])
    items = [l["item"] for l in q["lines"]]
    assert items[0] == "5kVA hybrid inverter" and "Protection kit" in items and items[-1] == "Labour"
    text = quote.as_text(q, "GieBee Engineering", "Rudo", "Harare", quote.reference("263771234567"))
    assert "*QUOTATION*" in text and "Ref WA-" in text and "*Total: USD 1,985*" in text and "Rudo, Harare" in text
    assert "6 × USD 92.50 = *USD 555*" in text and "Transport" in text


def test_closest_standard_size_is_used_and_said():
    q = quote.build(7)
    assert q["size_kva"] in (6.2, 8)
    assert "closest standard system" in quote.as_text(q, "GieBee Engineering")
    assert quote.build(2.9)["size_kva"] == 3


def test_book_builder_takes_medians_from_line_items():
    rows = []
    for qid, inv, panels in ((1, 300, 6), (2, 320, 6), (3, 340, 8)):
        rows += [(qid, "5kva hybrid inverter", 1, inv), (qid, "550 watts solar panels", panels, 92.5),
                 (qid, "48v 100ah lithium battery", 1, 720), (qid, "protection kit", 1, 60),
                 (qid, "labour", 1, 150), (qid, "transport", 1, 100)]
    rows += [(9, "nylon rope", 1, 20)]                              # not a solar system quote
    systems = build_quote_book.build(rows)
    assert len(systems) == 1 and systems[0]["size_kva"] == 5 and systems[0]["quotes"] == 3
    lines = {l["item"]: l for l in systems[0]["lines"]}
    assert lines["5kVA hybrid inverter"]["unit"] == 320 and lines["550W solar panels"]["qty"] == 6
    assert systems[0]["transport"] == 100 and all(l["item"] != "Transport" for l in systems[0]["lines"])


def test_book_has_no_customer_data():
    text = json.dumps(quote.book()).lower()
    assert "customer" not in text.replace("customer data", "") and "263" not in text


def test_language_model_may_reword_but_never_change_numbers():
    approved = "A 5kVA system is USD 1,985 installed. Shall I book a site visit?"
    assert llm._safe("Sure! Our 5kVA comes to USD 1,985 fitted. Want a free site visit?", approved, 400)
    assert llm._safe("Our 5kVA comes to USD 1,850 fitted.", approved, 400) is None      # a changed price
    assert llm._safe("UNKNOWN", approved, 400) is None and llm._safe("", approved, 400) is None


def test_natural_reply_is_used_when_safe_and_falls_back_otherwise(monkeypatch, chat):
    monkeypatch.setenv("LLM_API_KEY", "test")
    monkeypatch.setenv("LLM_REWORD", "on")
    monkeypatch.setattr(llm, "_chat", lambda *a, **k: "Sawubona! Yes, we're here to help with solar. 😊")
    assert "Sawubona" in chat.say("hello")                                  # reworded greeting
    monkeypatch.setattr(llm, "_chat", lambda *a, **k: "We charge USD 99 for everything.")
    reply = chat.say("hello")
    assert "99" not in reply and len(reply) > 10                            # invented number: approved text instead


def test_unrecognised_question_is_answered_from_company_facts(monkeypatch, chat):
    monkeypatch.setenv("LLM_API_KEY", "test")
    seen = {}

    def fake(system, user, **k):
        seen["user"] = user
        return "Our 5kVA system comes to USD 1,985 installed, transport extra."
    monkeypatch.setattr(llm, "_chat", fake)
    monkeypatch.setattr(brain, "analyze", lambda *a, **k: dict(brain_analysis_none()))
    reply = chat.say("zvakaoma here kuisa 5kva pamba pangu?")
    assert "USD 1,985" in reply and "COMPANY FACTS" in seen["user"] and "550W solar panels" in seen["user"]


def brain_analysis_none():
    return {"intent": None, "guess": None, "confidence": 0.0, "similarity": 0.0, "ranked": [], "matched": "",
            "category": "enquiry", "priority": "medium", "reasons": [], "entities": {}, "handler": "reply",
            "understood": False, "via": None}


def test_without_a_key_nothing_changes(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    assert llm.natural_reply("Hello", "hi", "GieBee Engineering") is None
    assert llm.answer_from_facts("hi?", "facts", "GieBee Engineering") is None


def test_rewording_is_off_by_default_but_fact_answers_are_on(monkeypatch, chat):
    monkeypatch.setenv("LLM_API_KEY", "test")
    monkeypatch.delenv("LLM_REWORD", raising=False)
    monkeypatch.delenv("LLM_ANSWERS", raising=False)
    monkeypatch.setattr(llm, "_chat", lambda *a, **k: "Sawubona! Yes, we're here to help with solar. 😊")
    assert llm.natural_reply("Hello", "hi", "GieBee") is None              # rewording off
    assert llm.answer_from_facts("how much?", "5kVA USD 1,985", "GieBee")  # fact answers on
    monkeypatch.setenv("LLM_ANSWERS", "off")
    assert llm.answer_from_facts("how much?", "5kVA USD 1,985", "GieBee") is None
