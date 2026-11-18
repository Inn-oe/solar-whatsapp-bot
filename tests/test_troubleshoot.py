import re

import troubleshoot as t

UNSAFE = re.compile(r"\b(unscrew|remove the (?:cover|lid)|open the (?:inverter|battery|cabinet|box)|"
                    r"touch the wiring|bypass|jump|short)\b", re.I)


def test_every_guide_has_two_safe_steps():
    for g in t.GUIDES:
        assert len(g["steps"]) >= t.MAX_ATTEMPTS, g["id"]
        for step in g["steps"]:
            assert not UNSAFE.search(step), (g["id"], step)


def test_guides_match_what_customers_say():
    cases = {
        "my solar is not charging": "not_charging",
        "my batteries are not charging anymore": "not_charging",
        "my inverter is having an error": "error_beep",
        "it keeps beeping since this morning": "error_beep",
        "the inverter keeps tripping and shutting down": "overload",
        "batteries drain too fast at night": "batteries_drain",
        "panels are producing very little power": "low_output",
        "my geyser is not working on solar": "pump_geyser",
        "my solar is malfunctioning": "generic",
    }
    for text, guide in cases.items():
        assert t.guide_for(text)["id"] == guide, text


def test_reply_verdicts():
    for text in ("yes it works now", "fixed", "it's working now", "all good thanks", "sorted"):
        assert t.verdict(text) == "fixed", text
    for text in ("no", "still the same", "it is still not working", "nothing changed", "no luck", "same problem"):
        assert t.verdict(text) == "not_fixed", text
    assert t.verdict("the display shows 12.3 volts") is None
