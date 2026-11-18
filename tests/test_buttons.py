"""Buttons and lists: shortcuts for closed choices, alongside questions that need a typed answer."""
import app as bot
import db
from tests.test_conversation import last_enquiry


def check_limits(reply):
    """Every set of options must be sendable on WhatsApp: 3 buttons (titles <= 20) or a list (<= 10 rows, <= 24)."""
    opts = reply["options"]
    if not opts:
        return
    assert len(opts) <= 10 and len({o["id"] for o in opts}) == len(opts)
    assert all(o["id"].startswith("btn:") for o in opts)
    _, inter = bot.interactive_payload(reply["text"], opts, reply["button"])
    assert len(inter["body"]["text"]) <= bot.BODY_LIMIT
    if inter["type"] == "button":
        assert len(opts) <= 3 and all(len(b["reply"]["title"]) <= 20 for b in inter["action"]["buttons"])
    else:
        rows = inter["action"]["sections"][0]["rows"]
        assert all(len(r["title"]) <= 24 and len(r.get("description", "")) <= 72 for r in rows)
        assert len(inter["action"]["button"]) <= 20


def walk(chat, steps):
    """Run a conversation: ("tap", key) or ("say", text). Checks the options of every reply."""
    out = None
    for kind, v in steps:
        out = chat.tap(v) if kind == "tap" else chat.say(v)
        check_limits(chat.last)
    return out


def test_greeting_offers_the_main_menu(chat):
    chat.say("hi")
    assert chat.keys() == ["btn:quote", "btn:fault", "btn:question"]


def test_fault_by_buttons_two_failed_fixes_then_a_technician(chat):
    done = walk(chat, [("say", "hi"), ("tap", "fault"), ("tap", "fault_charging"), ("tap", "code_none"),
                       ("tap", "notfixed"), ("tap", "notfixed"), ("say", "Borrowdale, Harare")])
    assert "technical team" in done and "Borrowdale, Harare" in done
    e = last_enquiry(chat.who, "maintenance")
    assert "not charging" in e["summary"] and "Bot fixes tried" in e["summary"]


def test_fault_buttons_offered_at_each_step(chat):
    chat.say("hi")
    chat.tap("fault")
    assert "btn:fault_charging" in chat.keys() and "btn:fault_other" in chat.keys()   # a list of problems
    chat.tap("fault_beep")
    assert chat.keys() == ["btn:code_none", "btn:code_beep", "btn:code_blank"]
    reply = chat.say("E05")                                    # the code itself is typed
    assert "did that fix it" in reply.lower() and chat.keys() == ["btn:fixed", "btn:notfixed", "btn:tshoot_human"]
    assert "glad" in chat.tap("fixed").lower()
    assert last_enquiry(chat.who, "maintenance")["status"] == "handled"


def test_talk_to_a_person_during_a_fix_hands_over_at_once(chat):
    walk(chat, [("say", "my inverter keeps tripping"), ("tap", "code_none")])
    reply = chat.tap("tshoot_human")
    assert "technician" in reply and "town" in reply.lower()
    done = chat.say("Mutare")
    assert "technical team" in done
    assert "asked for a person" in last_enquiry(chat.who, "maintenance")["summary"]


def test_something_else_asks_for_a_description(chat):
    walk(chat, [("say", "hi"), ("tap", "fault")])
    reply = chat.tap("fault_other")
    assert "own words" in reply and not chat.keys()
    assert "error code" in chat.say("the fan on the inverter is very loud since yesterday").lower()


def test_quote_by_buttons_with_estimate_change_and_send(chat):
    walk(chat, [("say", "hi"), ("tap", "quote")])
    assert [o["title"] for o in chat.last["options"]][2] == "Home 5kVA"      # packages as a list
    assert chat.last["button"] == "See packages"
    walk(chat, [("tap", "pkg_3")])
    confirm = chat.say("Harare")
    assert "Home 5kVA" in confirm and "USD 1,865" in confirm and chat.keys() == ["btn:send", "btn:change"]
    assert "what would you like to change" in chat.tap("change").lower()
    confirm = chat.say("Bulawayo")
    assert "Bulawayo" in confirm and "Harare" not in confirm
    done = chat.tap("send")
    assert "I've sent your request" in done and "USD 1,865" in done
    lead = [l for l in db.list_leads() if l["phone"] == chat.who][0]
    assert lead["location"] == "Bulawayo" and lead["suggested"] == "Home 5kVA"


def test_a_correction_typed_at_the_confirm_step_is_applied(chat):
    walk(chat, [("say", "I want a 5kva system"), ("say", "Harare")])
    confirm = chat.say("make it 10kva")
    assert "10kVA" in confirm and "Business 10kVA" in confirm and "Shall I send it" in confirm


def test_second_quote_request_asks_new_or_change(chat):
    """The screenshot: a quote was just sent and the client asks for 'the quote' again."""
    walk(chat, [("say", "I want a 5kva system"), ("say", "Harare"), ("tap", "send")])
    reply = chat.say("I need you to make the quote")
    assert "already sent" in reply and chat.keys() == ["btn:new_quote", "btn:edit_quote"]
    walk(chat, [("tap", "edit_quote"), ("say", "make it 10kva")])
    done = chat.tap("send")
    assert "updated request" in done and "Business 10kVA" in done
    assert any(l["need"].startswith("UPDATED") for l in db.list_leads() if l["phone"] == chat.who)


def test_another_system_starts_a_fresh_quote(chat):
    walk(chat, [("say", "I want a 5kva system"), ("say", "Harare"), ("tap", "send"), ("say", "I want a quotation")])
    reply = chat.tap("new_quote")
    assert "power" in reply.lower() or "run" in reply.lower()
    assert "Harare" not in reply


def test_typing_a_button_title_counts_as_tapping_it(chat):
    chat.say("hi")
    reply = chat.say("Get a quotation")
    assert "run" in reply.lower() or "power" in reply.lower()
    assert "btn:pkg_1" in chat.keys()


def test_question_topics_list_and_answer(chat):
    walk(chat, [("say", "hi"), ("tap", "question")])
    keys = chat.keys()
    assert keys[0] == "btn:prices" and keys[-1] == "btn:ask_own" and 4 <= len(keys) <= 10
    faq = next(k for k in keys if k.startswith("btn:faq:"))
    answer = chat.tap(faq[4:])
    assert len(answer) > 40 and chat.last["options"]       # answered, with a next step
    assert "Starter 1.5kVA" in walk(chat, [("say", "hi"), ("tap", "question"), ("tap", "prices")])


def test_topic_tapped_mid_quote_returns_to_the_question(chat):
    walk(chat, [("say", "I want to buy solar"), ("say", "fridge and tv")])   # asked for the town
    chat.say("btn:prices")
    assert "town" in chat.last["text"].lower()                               # answers, then goes back


def test_payload_buttons_list_and_long_text():
    _, inter = bot.interactive_payload("Pick one", [{"id": "btn:a", "title": "Yes"}, {"id": "btn:b", "title": "No"}])
    assert inter["type"] == "button" and len(inter["action"]["buttons"]) == 2
    rows = [{"id": f"btn:r{i}", "title": f"Row number {i} with a long title", "description": "x" * 100}
            for i in range(12)]
    _, inter = bot.interactive_payload("Pick", rows, "A very long list button label")
    r = inter["action"]["sections"][0]["rows"]
    assert inter["type"] == "list" and len(r) == 10 and all(len(x["title"]) <= 24 for x in r)
    long = "First part. " * 120 + "\n\nDid that fix it?"
    lead, inter = bot.interactive_payload(long, [{"id": "btn:a", "title": "Fixed"}])
    assert lead and inter["body"]["text"] == "Did that fix it?"


def test_web_chat_returns_options_and_logs_the_title(client):
    r = client.post("/api/chat", json={"session": "btntest1", "text": "hi"})
    assert [o["id"] for o in r.get_json()["options"]] == ["btn:quote", "btn:fault", "btn:question"]
    client.post("/api/chat", json={"session": "btntest1", "text": "btn:fault", "shown": "Fix a problem"})
    assert any(m["text"] == "Fix a problem" for m in db.thread("web:btntest1"))


def test_whatsapp_button_tap_is_routed_by_id(client):
    who = "263779990001"

    def tap(id_, title):
        body = {"entry": [{"changes": [{"value": {
            "contacts": [{"wa_id": who, "profile": {"name": "Rudo"}}],
            "messages": [{"id": "wamid." + id_ + title, "from": who, "type": "interactive",
                          "interactive": {"type": "button_reply", "button_reply": {"id": id_, "title": title}}}]}}]}]}
        assert client.post("/webhook", json=body).status_code == 200

    tap("btn:fault", "Fix a problem")
    texts = [m["text"] for m in db.thread(who)]
    assert "Fix a problem" in texts                                   # history shows what they tapped
    assert db.get_state(who).get("awaiting") == "issue"              # and the bot acted on the button
