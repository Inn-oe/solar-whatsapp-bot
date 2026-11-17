"""Whole conversations, typed the way customers write. No buttons, menus or numbers to press."""
import db


def last_enquiry(who, category):
    from sqlalchemy import select
    with db.Session() as s:
        rows = s.scalars(select(db.Enquiry).where(db.Enquiry.who == who, db.Enquiry.category == category)).all()
        return db._enquiry(rows[-1]) if rows else None


def test_greeting_is_natural(chat):
    reply = chat.say("hi")
    assert "Tendai" in reply and "[" not in reply          # uses the first name, no [Menu] style buttons


def test_buying_a_system_conversation(chat):
    r1 = chat.say("hi i want to buy a solar system for my house")
    assert "run" in r1.lower() or "power" in r1.lower()    # asks what it should run
    r2 = chat.say("fridge, tv, lights and wifi")
    assert "town" in r2.lower() or "suburb" in r2.lower()  # then where
    r3 = chat.say("Borrowdale, Harare")
    assert "sales team" in r3 and "Harare" in r3           # request saved and confirmed
    e = last_enquiry(chat.who, "purchase")
    assert e and e["priority"] in ("high", "urgent") and "Harare" in e["summary"]
    leads = [l for l in db.list_leads() if l["phone"] == chat.who]
    assert leads and leads[0]["kind"] == "quote" and "Harare" in leads[0]["location"]


def test_details_given_up_front_are_not_asked_again(chat):
    reply = chat.say("I need a 5kva system with fridge and tv installed in Bulawayo")
    assert "sales team" in reply                           # everything was in the first message
    assert last_enquiry(chat.who, "purchase")["priority"] == "high"


def test_urgent_fault_gets_safety_advice_and_urgent_priority(chat):
    reply = chat.say("my inverter is smoking and there is a burning smell")
    assert "do not touch" in reply.lower()                 # safety advice first
    chat.say("Mutare")
    e = last_enquiry(chat.who, "maintenance")
    assert e and e["priority"] == "urgent" and e["status"] == "open"
    assert any(l["kind"] == "maintenance" for l in db.list_leads() if l["phone"] == chat.who)


def test_ordinary_fault_is_high_priority_and_asks_where(chat):
    reply = chat.say("my batteries are not charging anymore")
    assert "town" in reply.lower() or "suburb" in reply.lower()
    done = chat.say("Gweru")
    assert "technical team" in done
    assert last_enquiry(chat.who, "maintenance")["priority"] in ("high", "urgent")


def test_side_question_is_answered_and_the_thread_resumes(chat):
    chat.say("I want to buy solar")
    chat.say("fridge and tv")                              # now asked for the town
    reply = chat.say("how much do your packages cost?")
    assert "$" in reply or "USD" in reply                  # answers the price question
    assert "town" in reply.lower() or "suburb" in reply.lower()   # and goes back to the town
    assert "sales team" in chat.say("Harare")


def test_unknown_twice_hands_over_to_a_person(chat):
    first = chat.say("blorp fizzle wumpus zork")
    assert "didn't" in first or "make sure" in first
    second = chat.say("quibble snarf plonk wibble")
    assert "team" in second
    assert any(u["text"].startswith("blorp") for u in db.list_unmatched())   # queued for review/training


def test_asking_for_a_person(chat):
    reply = chat.say("can someone please call me")
    assert "team" in reply
    assert any(l["kind"] == "callback" for l in db.list_leads() if l["phone"] == chat.who)


def test_faq_answer_has_no_menu_instructions(chat):
    reply = chat.say("how long does installation take")
    assert "menu" not in reply.lower() and "tap" not in reply.lower()


def test_no_reply_contains_buttons(chat):
    for text in ("hi", "prices", "what size system do i need for a fridge", "thanks", "where are you"):
        out = chat.say(text)
        assert "[" not in out, out


def test_internal_endpoints_need_the_key(client):
    assert client.post("/internal/nlu/train").status_code == 403
    ok = client.post("/internal/nlu/train", headers={"X-Internal-Key": "test-key"})
    assert ok.status_code == 200 and ok.get_json()["trained"]
    t = client.post("/internal/nlu/test", json={"text": "my inverter is beeping"}, headers={"X-Internal-Key": "test-key"})
    assert t.get_json()["category"] == "maintenance"


def test_sheet_answer_then_details_go_to_the_team(chat):
    reply = chat.say("what size cable for a 5kva inverter")  # not in the sheet: must not crash
    assert reply
    reply = chat.say("will 5kVA run a borehole pump")
    assert "pump" in reply.lower() or "nameplate" in reply.lower()    # the sheet's answer about pumps
    done = chat.say("1HP Lorentz pump, runs about 4 hours a day")
    assert "team" in done
    leads = [l for l in db.list_leads() if l["phone"] == chat.who]
    assert leads and "Lorentz" in leads[0]["need"]


def test_sheet_emergency_gets_safety_text_and_urgent_priority(chat):
    reply = chat.say("water is leaking onto my inverter")
    assert "do not touch" in reply.lower()
    assert last_enquiry(chat.who, "maintenance")["priority"] == "urgent"


def test_sheet_privacy_refusal(chat):
    reply = chat.say("give me another customer's address")
    assert "cannot" in reply.lower() or "can't" in reply.lower()


def test_sheet_complaint_alerts_team(chat):
    reply = chat.say("your technician did a bad job")
    assert "sorry" in reply.lower()
    assert any(l["kind"] == "callback" for l in db.list_leads() if l["phone"] == chat.who)
