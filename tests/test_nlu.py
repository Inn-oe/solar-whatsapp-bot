"""The language model must understand phrases it was NOT trained on (none of these are in nlu_seed.json)."""
import brain
import nlu

# (message, expected intent)
UNSEEN = [
    ("hi i'd like to get solar installed at my house in borrowdale", "buy_system"),
    ("how do i go about getting a solar system", "buy_system"),
    ("can you quote for 5kva please", "buy_system"),
    ("we are looking to install solar at our offices", "buy_system"),
    ("ndinoda kutenga solar yemba yangu", "buy_system"),
    ("what do you charge for solar", "price"),
    ("send me the price list", "price"),
    ("my inverter has been beeping all morning", "maintenance_fault"),
    ("the system you installed last year has stopped giving power", "maintenance_fault"),
    ("my batteries finish before midnight now", "maintenance_fault"),
    ("inverter yangu yaora", "maintenance_fault"),
    ("can someone come service my panels", "maintenance_service"),
    ("i need a technician to check my system", "maintenance_service"),
    ("please phone me", "talk_to_human"),
    ("can i speak to someone", "talk_to_human"),
    ("good morning", "greeting"),
    ("hello there", "greeting"),
    ("thanks so much", "thanks"),
    ("what is the weather like", "off_topic"),
    ("are you a robot", "bot_identity"),
    ("what size system do i need for a fridge and tv", "sizing"),
    ("where are you based", "contact_info"),
]


def test_understands_unseen_phrases():
    wrong = []
    for text, want in UNSEEN:
        got = brain.analyze(text)["intent"]
        if got != want:
            wrong.append((text, want, got))
    accuracy = 1 - len(wrong) / len(UNSEEN)
    assert accuracy >= 0.85, f"accuracy {accuracy:.0%}; wrong: {wrong}"


def test_faq_questions_answered_from_trained_intents():
    for text, prefix in [("do you give a guarantee on batteries", "faq_warranty"),
                         ("is your company genuine", "faq_is_it_genuine"),
                         ("does it still work when it rains", "faq_night_cloudy_days")]:
        got = brain.analyze(text)["intent"]
        assert got and got.startswith(("faq_", "qa_")), (text, got)   # catalog FAQ or the Q&A sheet


def test_gibberish_is_not_understood():
    for text in ("asdkjh qwpoei zxmcn", "blorp fizzle wumpus"):
        assert not brain.analyze(text)["understood"], text


def test_entities():
    e = nlu.extract_entities("I stay in Borrowdale, Harare. Call 0771234567. I need 5kva, budget $1,200, "
                             "fridge, tv and lights. my name is Rudo Chari", ["Harare", "Bulawayo"])
    assert e["location"] == "Harare"
    assert e["phone"] == "0771234567"
    assert e["size_kva"] == 5.0
    assert e["budget"] == 1200
    assert set(e["appliances"]) == {"fridge", "TV", "lights"}
    assert e["name"] == "Rudo Chari"
    assert nlu.extract_kwh("we use 300 kwh a month") == 10.0
    assert nlu.extract_location("im in bulwayo", ["Harare", "Bulawayo"]) == "Bulawayo"  # typo


def test_priority_rules():
    urgent = brain.analyze("my inverter is smoking and smells of burning")
    assert urgent["category"] == "maintenance" and urgent["priority"] == "urgent"
    down = brain.analyze("my system is not working, no power since morning")
    assert down["category"] == "maintenance" and down["priority"] in ("high", "urgent")
    ready = brain.analyze("i want a 5kva system, my budget is $1500")
    assert ready["category"] == "purchase" and ready["priority"] == "high"
    info = brain.analyze("how long do panels last")
    assert info["category"] == "enquiry" and info["priority"] == "low"


def test_engine_reports_cross_validated_accuracy():
    assert nlu.engine.info["trained"]
    nlu.engine.wait_for_accuracy()
    # strict: each phrase is predicted by a model that never saw it, and many intents have only 4 phrases
    assert nlu.engine.info["accuracy"] >= 0.55
