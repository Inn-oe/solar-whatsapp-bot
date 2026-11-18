"""GieBee Engineering's real systems and services (built from its 2026 quotations) and the training-file retire step."""
import app as bot
import brain
import db


def test_catalog_is_valid_and_is_giebees():
    assert bot.validate_catalog(bot.CATALOG) is None
    assert bot.CATALOG["company"] == "GieBee Engineering"
    assert bot.CATALOG.get("market_prices") is False          # quote our own prices, not the market's range


def test_packages_are_real_sizes_in_order():
    pk = bot.CATALOG["packages"]
    assert [p["name"] for p in pk] == ["Starter 1.5kVA", "Home 3.2kVA", "Home 5kVA", "Family 6.2kVA", "Business 10kVA"]
    assert [p["price"] for p in pk] == sorted(p["price"] for p in pk)                 # bigger system, bigger price
    assert [p["daily_kwh"] for p in pk] == sorted(p["daily_kwh"] for p in pk)         # so recommend() walks up the list
    assert bot.package_size(pk[1]) == 3.2 and bot.package_size(pk[4]) == 10.0
    assert bot.recommend(5)["name"] == "Home 3.2kVA" and bot.recommend(9)["name"] == "Home 5kVA"
    assert bot.recommend(99)["name"] == "Business 10kVA"


def test_package_prices_are_shown_as_the_catalog_price(chat):
    reply = chat.say("what packages do you have")
    assert "Home 5kVA" in reply and "1,865" in reply and "Business 10kVA" in reply


def test_services_questions_get_the_right_answer():
    for text, intent in [("do you install cctv cameras", "faq_cctv_security"),
                         ("i want to pump water from my borehole using solar", "faq_solar_water_pumping"),
                         ("how much for a 1hp pump", "faq_solar_water_pumping"),
                         ("can you install a solar geyser for me", "faq_solar_geysers"),
                         ("i need someone to wire my house", "faq_electrical_work"),
                         ("what is the price of a 5kwh lithium battery", "faq_component_prices"),
                         ("is transport included in the price", "faq_whats_in_a_quote")]:
        assert brain.analyze(text)["intent"] == intent, text


def test_pump_answer_has_the_real_prices(chat):
    reply = chat.say("how much for a 1hp pump")
    assert "1,050" in reply and "5,000L" in reply


def test_retired_phrases_go_but_staff_phrases_stay():
    db.add_examples_bulk([("buy_system", "zz retired starter phrase")], "seed")
    db.add_example("buy_system", "zz staff phrase", "staff")
    db.add_examples_bulk([("buy_system", "zz same text staff made")], "seed")
    removed = db.delete_seed_examples([("buy_system", "zz retired starter phrase"), ("buy_system", "zz staff phrase")])
    left = {e["text"] for e in db.list_examples("buy_system")}
    assert removed == 1
    assert "zz retired starter phrase" not in left and "zz staff phrase" in left and "zz same text staff made" in left
