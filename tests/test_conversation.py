"""Whole conversations, typed the way customers write. No buttons, menus or numbers to press."""
import time

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
    assert "Borrowdale, Harare" in r3 and "btn:send" in chat.keys()   # read back before it is sent
    assert not db.list_leads() or all(l["phone"] != chat.who for l in db.list_leads())
    r4 = chat.tap("send")
    assert "I've sent your request to our sales team" in r4 and "Harare" in r4
    e = last_enquiry(chat.who, "purchase")
    assert e and e["priority"] in ("high", "urgent") and "Harare" in e["summary"]
    leads = [l for l in db.list_leads() if l["phone"] == chat.who]
    assert leads and leads[0]["kind"] == "quote" and "Harare" in leads[0]["location"]


def test_details_given_up_front_are_not_asked_again(chat):
    reply = chat.say("I need a 5kva system with fridge and tv installed in Bulawayo")
    assert "Shall I send it" in reply and "Bulawayo" in reply   # everything was in the first message: confirm
    assert "I've sent" in chat.say("yes")                       # a typed yes works as well as the button
    assert last_enquiry(chat.who, "purchase")["priority"] == "high"


def test_urgent_fault_gets_safety_advice_and_urgent_priority(chat):
    reply = chat.say("my inverter is smoking and there is a burning smell")
    assert "do not touch" in reply.lower()                 # safety advice first
    assert "inverter" in chat.say("Mutare").lower()        # then what the inverter shows
    chat.say("no")
    e = last_enquiry(chat.who, "maintenance")
    assert e and e["priority"] == "urgent" and e["status"] == "open"
    assert any(l["kind"] == "maintenance" for l in db.list_leads() if l["phone"] == chat.who)


def test_ordinary_fault_is_high_priority_and_escalates_after_two_failed_fixes(chat):
    first = chat.say("my batteries are not charging anymore")
    assert "error code" in first.lower()                   # a technician's first question: what does it show?
    fix1 = chat.say("no")
    assert "pv" in fix1.lower() and "did that fix it" in fix1.lower()      # first fix to try, asks if it worked
    fix2 = chat.say("still the same")
    assert "restart" in fix2.lower()                        # a different second fix
    ask_town = chat.say("no change")
    assert "sorry" in ask_town.lower() and ("town" in ask_town.lower() or "suburb" in ask_town.lower())
    done = chat.say("Gweru")
    assert "technical team" in done and "fixes we tried" in done
    e = last_enquiry(chat.who, "maintenance")
    assert e["priority"] in ("high", "urgent") and e["status"] == "open"
    assert "Bot fixes tried" in e["summary"] and "no change" in e["summary"]


def test_side_question_is_answered_and_the_thread_resumes(chat):
    chat.say("I want to buy solar")
    chat.say("fridge and tv")                              # now asked for the town
    reply = chat.say("how much do your packages cost?")
    assert "$" in reply or "USD" in reply                  # answers the price question
    assert "town" in reply.lower() or "suburb" in reply.lower()   # and goes back to the town
    assert "Shall I send it" in chat.say("Harare")
    assert "I've sent" in chat.say("yes please")


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
    ok = client.post("/internal/nlu/train", json={"wait": True}, headers={"X-Internal-Key": "test-key"})
    assert ok.status_code == 200 and ok.get_json()["trained"]
    bg = client.post("/internal/nlu/train", headers={"X-Internal-Key": "test-key"})
    assert bg.status_code == 200 and bg.get_json()["training"] is True
    for _ in range(300):   # the background run finishes and the status says so
        status = client.get("/internal/nlu/status", headers={"X-Internal-Key": "test-key"}).get_json()
        if not status["training"]:
            break
        time.sleep(0.1)
    assert status["training"] is False and status["trained"] and not status["train_error"]
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


def test_transformer_is_not_started_on_a_small_server(monkeypatch):
    import brain
    monkeypatch.setenv("NLU_BACKEND", "transformer")
    monkeypatch.setattr(brain, "_memory_limit_mb", lambda: 512)
    started = []
    monkeypatch.setattr(brain, "_upgrade_to_transformer", lambda examples: started.append(1))
    info = brain.retrain()
    assert not started and "512 MB" in info["note"]
    monkeypatch.setattr(brain, "_memory_limit_mb", lambda: 4096)
    brain.retrain()
    assert started


def test_vague_fault_gets_a_human_follow_up_not_a_bare_location_question(chat):
    """'My solar is malfunctioning' must ask what is happening, then what the inverter shows, then help."""
    r1 = chat.say("My solar is malfunctioning")
    assert "sorry" in r1.lower() or "frustrating" in r1.lower()
    assert "beeping" in r1.lower() or "error" in r1.lower()          # asks what the system is doing
    assert "town" not in r1.lower() and not r1.startswith("Thanks")  # no pointless thanks, no town yet
    r2 = chat.say("it keeps beeping since this morning")
    assert r2.startswith(("Thanks", "Got it", "Okay", "Noted")) and "error code" in r2.lower()
    assert "which brand" in chat.say("it shows E12 on the display").lower()
    r3 = chat.say("not sure")
    assert "restart" not in r3.lower() and "did that fix it" in r3.lower()     # first fix, for a code/beeping fault
    assert "restart" in chat.say("still beeping").lower()            # second fix
    assert "town" in chat.say("same").lower()                        # both failed: now collect where
    r6 = chat.say("Harare")
    assert "beeping" in r6 and "Harare" in r6 and "don't open" in r6
    e = last_enquiry(chat.who, "maintenance")
    assert e and "malfunctioning" in e["summary"] and "beeping" in e["summary"] and "E12" in e["summary"]


def test_specific_fault_is_acknowledged_with_sympathy(chat):
    reply = chat.say("my batteries are not charging anymore")
    assert reply.lower().startswith(("sorry", "that sounds"))
    assert "error code" in reply.lower()


def test_coverage_question_is_answered_not_a_contact_card(chat):
    reply = chat.say("do you install in Bulawayo?")
    assert reply.startswith("Yes, we install in Bulawayo") and "📞" not in reply
    follow = chat.say("yes please")                              # accepts the offer: the quote conversation starts
    assert ("run" in follow.lower() or "power" in follow.lower()) and follow.lower().count("what") == 1


def test_appliance_statement_is_sized_not_answered_from_the_faq(chat):
    reply = chat.say("i have a fridge, tv and a borehole pump")
    assert "kWh" in reply and "suggest" in reply


def test_yes_please_after_an_answer_starts_the_quote(chat):
    chat.say("how much is a 5kva system")                         # offers a quotation
    chat.say("ok and installation included?")                      # answered, no details demanded
    reply = chat.say("yes please send quote")
    assert "run" in reply.lower() or "power" in reply.lower()      # the quote conversation starts


def test_no_code_answer_keeps_the_extra_detail(chat):
    chat.say("my inverter is off and the lights are out")           # asked what the inverter shows
    reply = chat.say("no error just dead")                          # "no" answers it; "just dead" is kept
    assert "did that fix it" in reply.lower() and "No Error" not in reply     # moves on to the first fix
    chat.say("still the same")
    assert "town" in chat.say("same").lower()
    done = chat.say("Kwekwe")
    assert "Kwekwe" in done and "no error just dead" in done.lower()


def test_service_request_keeps_the_complaint_and_is_not_marked_priority(chat):
    chat.say("can u come check my panels they are dirty")
    done = chat.say("Mutare")
    assert "dirty" in done and "as a priority" not in done and "Mutare; Routine" not in done
    assert last_enquiry(chat.who, "maintenance")["priority"] == "medium"


def test_fault_reported_while_asked_what_to_run_is_a_maintenance_report(chat):
    """'I also need another quotation', then 'My inverter is having an error' must not become the quotation's
    'need'. It is a fault report: ask what the inverter shows, try fixes, then hand over."""
    first = chat.say("I also need another quotation").lower()
    assert "run" in first or "power" in first                  # asks what the system should run
    assert "error code" in chat.say("My inverter is having an error").lower()
    assert "which brand" in chat.say("E05 on the display").lower()
    assert "did that fix it" in chat.say("not sure").lower()
    chat.say("still the same")
    assert "town" in chat.say("no change").lower()
    done = chat.say("Harare")
    assert "technical team" in done and "sales team" not in done
    assert "E05" in last_enquiry(chat.who, "maintenance")["summary"]
    assert not any(l["kind"] == "quote" for l in db.list_leads() if l["phone"] == chat.who)

def test_fault_fixed_on_the_first_try_is_logged_as_handled_without_alerting_staff(chat):
    chat.say("my solar is not charging")
    chat.say("no")
    done = chat.say("yes it's working now, thank you")
    assert "glad" in done.lower() and "technical team" not in done
    e = last_enquiry(chat.who, "maintenance")
    assert e["status"] == "handled" and "solved by bot" in e["summary"]
    assert not any(l["kind"] == "maintenance" for l in db.list_leads() if l["phone"] == chat.who)


def test_fault_fixed_on_the_second_try(chat):
    chat.say("my inverter keeps tripping and shutting down")
    chat.say("no")
    assert "culprit" in chat.say("no, still trips").lower()  # second fix: find the appliance that trips it
    done = chat.say("fixed, it was the geyser")
    assert "glad" in done.lower()
    assert last_enquiry(chat.who, "maintenance")["status"] == "handled"


def test_side_question_during_a_fix_is_answered_and_the_fix_resumes(chat):
    chat.say("my solar is not charging")
    chat.say("no")                                          # now trying the first fix
    reply = chat.say("how much do your packages cost?")
    assert "$" in reply or "USD" in reply
    assert "back to the fix" in reply.lower()


def test_safety_risk_mid_fix_stops_the_fixes_and_goes_to_a_person(chat):
    chat.say("my solar is not charging")
    chat.say("no")
    reply = chat.say("wait, there is smoke coming from the inverter")
    assert "do not touch" in reply.lower() and "did that fix it" not in reply.lower()   # safety first, no more DIY fixes
    done = chat.say("Harare")
    assert "URGENT" in done and "fixes we tried" not in done
    assert last_enquiry(chat.who, "maintenance")["priority"] == "urgent"


def test_plain_fault_statements_are_understood_without_a_menu(chat):
    """Fault phrases the trained model is unsure about must still start troubleshooting, not a 'what would you
    like?' menu. An error code said up front is kept, so the bot does not ask for it again."""
    assert "error code" in chat.say("solar not giving power").lower()
    chat.say("no")
    chat.say("it works now")                                 # close that conversation
    second = chat.say("inverter showing E05")
    assert "which brand" in second.lower() and "does the inverter show" not in second.lower()      # code known: only the brand is missing
    assert "did that fix it" in chat.say("not sure").lower()


def test_screenshot_price_question_and_did_you_understand(chat):
    """'Marii 3kva' gets the price and keeps the size; 'did you understand' / 'what did I say' get a recap of
    what the bot has, then the question it still needs answered (never a fake 'got it')."""
    chat.say("I want a 3.2kva system")
    chat.say("Masvingo")
    chat.tap("send")
    assert "already sent" in chat.say("Get a quotation")
    price = chat.say("Marii 3kva")
    assert "3kVA system: *USD 1,210*" in price and "450W solar panels" in price and "town" in price.lower()
    for meta in ("Did you understand", "What did I say", "wanzwa here"):
        reply = chat.say(meta)
        assert "following" in reply or "understood" in reply, meta
        assert "3kVA" in reply and "town" in reply.lower() and "Got it" not in reply
    assert "Shall I send it" in chat.say("Masvingo")


def test_price_question_while_asked_for_the_town(chat):
    chat.say("I want a 5kva system")                        # asked for the town
    reply = chat.say("how much is it")
    assert "5kVA" in reply and "1,985" in reply and "town" in reply.lower()


def test_unclear_reply_mid_request_is_admitted_not_faked(chat):
    chat.say("my solar is not charging")                   # asked what the inverter shows
    chat.say("no")                                          # first fix
    reply = chat.say("purple monkey dishwasher")
    assert "did that fix it" in reply.lower() or "sorry" in reply.lower()
    assert "Got it, thanks" not in reply


def test_shona_question_without_a_question_mark_is_a_question():
    import brain
    for t in ("marii 3kva", "zvinoita sei", "did you understand me", "ko kuisa kunotora mazuva mangani"):
        assert brain.is_question(t), t
    assert not brain.is_question("Borrowdale, Harare")
    assert not brain.is_question("the inverter is here in the garage and it is still beeping")


def test_known_fault_code_is_explained_from_the_table_and_a_serious_one_goes_to_a_person(chat):
    assert "error code" in chat.say("My Deye inverter is having an error").lower()
    reply = chat.say("F24")                                    # DC insulation low: a real fault, not a do-it-yourself one
    assert "insulation" in reply.lower() and "did that fix it" not in reply.lower()
    assert "technician" in reply.lower()


def test_unknown_code_is_never_guessed(chat):
    assert "error code" in chat.say("My Growatt inverter is having an error").lower()
    reply = chat.say("error 77")
    assert "won't guess" in reply.lower() and "technician" in reply.lower()


def test_sizing_shows_the_working_with_zimbabwe_sun_hours(chat):
    reply = chat.say("I use about 10kWh a day in Harare, what size system do I need?")
    assert "sun-hours" in reply and "÷" in reply


def test_shona_names_are_taken_as_names_not_as_thanks(chat):
    chat.say("I want a 5kVA solar system")
    chat.say("Harare")
    chat.say("0771234567")
    reply = chat.say("Tendai Moyo")
    assert "what name" not in reply.lower() and "welcome" not in reply.lower()
