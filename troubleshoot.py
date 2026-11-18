"""Solar technician knowledge base: first-line fixes a customer can safely try, in order.

Each guide has a symptom pattern and up to MAX_ATTEMPTS steps. The bot gives one step, asks whether it worked,
and if it did not gives the next. After the last step it hands over to a person. Steps never ask the customer to
open the inverter, touch wiring or work on the roof; they use only switches, breakers, the display and what can be
seen from the ground.

Brand-neutral on purpose: exact error-code meanings differ per make, so for codes the bot records the code and
runs the universal checks, and the technician reads the code against the manual.
"""
import re

MAX_ATTEMPTS = 2

RESTART = ("Restart it properly: 1) switch off or unplug the big appliances, 2) switch off the inverter's AC "
           "output breaker, 3) switch off the solar (PV) isolator, 4) switch off the battery breaker. Wait 5 "
           "minutes. Then switch back on in the opposite order: battery, solar, AC output.")

GUIDES = [
    {
        "id": "overload",
        "title": "overload or tripping",
        "pattern": r"\b(overload\w*|trip\w*|shuts?\w* (?:down|off)|switch\w* off by itself|keeps? (?:going|turning|cutting) off|"
                   r"cuts? (?:off|out))\b",
        "steps": [
            "That sounds like the inverter being overloaded. Please switch off the heavy appliances (geyser, "
            "stove, kettle, heater, iron, borehole pump, air conditioner), wait a minute, then switch the inverter "
            "on again. If it stays on, add the appliances back one at a time.",
            "Let's find the culprit. With only lights and the TV on, does it run normally? Then switch the "
            "heavy appliances on one by one and tell me which one makes it trip or beep. Some appliances (geysers, "
            "pumps, fridges) draw far more power when starting than the system can supply.",
        ],
    },
    {
        "id": "not_charging",
        "title": "not charging",
        "pattern": r"\b(not charg\w*|no charg\w*|wont charge|won'?t charge|charg\w* (?:issue|problem|fault)|"
                   r"battery(?:ies)? (?:not|wont|won'?t) (?:charg\w*|fill|full)|panels? (?:not|no) (?:charg\w*|working))\b",
        "steps": [
            "Let's check the basics. 1) Is it daytime with reasonable sun? 2) Are the panels in shade, covered in "
            "dust, leaves or bird droppings (look from the ground)? 3) Is the solar (PV) breaker or isolator "
            "switched ON? 4) What does the display show for solar/PV power or voltage: a number above zero, or zero?",
            "Please restart the system, because a stuck controller can stop charging. " + RESTART +
            " After about 10 minutes in daylight, does the display show solar power coming in?",
        ],
    },
    {
        "id": "batteries_drain",
        "title": "batteries drain fast",
        "pattern": r"\b(batter\w* (?:drain\w*|dies?|dying|flat|finish\w*|run\w* out|not (?:last\w*|hold\w*)|low|weak)|"
                   r"(?:drain\w*|runs? out|finish\w*|not last\w*) (?:fast|quickly|too fast|early)|power (?:finish\w*|runs? out)|"
                   r"lights? (?:go|goes) off (?:early|at night))\b",
        "steps": [
            "Let's check what is using the energy. Are any heavy appliances running overnight or in the evening "
            "(geyser, heater, freezer, pump, air conditioner)? Please switch those to the day, or off, for one "
            "evening. Also tell me what battery percentage the display shows at sunset.",
            "Battery life depends on a full charge each day. After a sunny day, does the display reach 100% (or "
            "float/full) by mid-afternoon? If it never gets full, the problem is charging, not the battery. Please "
            "also tell me how old the batteries are.",
        ],
    },
    {
        "id": "low_output",
        "title": "low solar production",
        "pattern": r"\b(panels? (?:are |is )?(?:producing|giving|making|generating) (?:very )?(?:little|low|less|nothing)|"
                   r"low (?:production|generation|output|yield)|(?:little|less|no) (?:production|generation)|"
                   r"producing (?:very )?(?:little|low|less))\b",
        "steps": [
            "Check from the ground at midday on a clear day: are any panels shaded by a tree, wall or new building, "
            "or covered in dust, leaves or droppings? Please wash the panels from the ground with a hose or bucket "
            "of clean water early morning or evening (never cold water on hot panels, and never climb on the roof). "
            "Then tell me what the display shows at midday.",
            "Please write down the solar power (watts) the display shows at about noon on a clear day, and tell me "
            "the size of your panel array (number of panels and watts each, if you know). A healthy system should "
            "reach roughly 70 to 80% of the panel rating at midday.",
        ],
    },
    {
        "id": "pump_geyser",
        "title": "geyser or pump not running",
        "pattern": r"\b(geyser|borehole|pump|heater|stove|aircon|air conditioner)\b.*\b(not|wont|won'?t|doesn'?t|stopped|trips?|off)\b|"
                   r"\b(not|wont|won'?t|doesn'?t|stopped)\b.*\b(geyser|borehole|pump|heater|stove|aircon|air conditioner)\b",
        "steps": [
            "Please check the appliance's own breaker or switch in the distribution board (DB) is ON and has not "
            "tripped, and that the inverter is on and has power. Is the appliance's timer or thermostat set "
            "correctly? Does it work at all when you switch the inverter's load off and try it?",
            "Heavy loads like geysers and borehole pumps often need a lot more than the system can supply, "
            "especially at night or on cloudy days. Try running it at midday in full sun with all other appliances "
            "off. Does it run then? Tell me what the inverter display shows (load in watts, battery %).",
        ],
    },
    {
        "id": "no_power",
        "title": "no power or dead inverter",
        "pattern": r"\b(no power|no electricity|no lights?|power (?:is )?(?:off|out|gone)|dead|blank|display (?:is )?(?:off|blank|dark)|"
                   r"won'?t (?:switch|turn) on|not (?:switch\w*|turn\w*) on|(?:is|are) off|inverter off|"
                   r"not (?:working|giving power|supplying))\b",
        "steps": [
            "Let's check the simple things first. 1) Is the inverter's display on? 2) Are the battery breaker and "
            "the inverter's AC output breaker switched ON? 3) Check the distribution board for any tripped "
            "breakers or the earth-leakage switch. Please tell me what you find, including whether the display "
            "is on or blank.",
            "Please restart the system. " + RESTART + " Does the display come on, and does it show a battery "
            "percentage or voltage? If so, please tell me the number.",
        ],
    },
    {
        "id": "error_beep",
        "title": "error code or beeping",
        "pattern": r"\b(error|fault|code|alarm|beep\w*|warning|flash\w*|blink\w*|\be-?\d{1,3}\b|\bf-?\d{1,3}\b)\b",
        "steps": [
            "Please note exactly what the inverter shows: the error or fault code (for example E05 or F12), "
            "any warning light colour, and whether it beeps constantly or in pulses. Then switch off the heaviest "
            "appliances and see whether the beeping or code clears. Does it?",
            "Let's try a full restart, which clears most temporary faults. " + RESTART + " Does the same code or "
            "beeping come back? Tell me the code if it does and how long after starting it returned.",
        ],
    },
    {
        "id": "generic",
        "title": "general fault",
        "pattern": r"",
        "steps": [
            "Let's start with the basics a technician checks first. 1) Is the inverter's display on, and does it "
            "show any error code, warning light or beeping? 2) Are the battery breaker, solar isolator and AC "
            "output breaker all ON? 3) Any tripped breaker in the distribution board? Tell me what you see.",
            "Please restart the system, because it clears most temporary faults. " + RESTART + " Is the problem "
            "still there afterwards? Tell me what the display shows.",
        ],
    },
]

_COMPILED = [(g, re.compile(g["pattern"], re.I) if g["pattern"] else None) for g in GUIDES]
_BY_ID = {g["id"]: g for g in GUIDES}


def guide_for(text):
    """The most specific guide for what the customer described (the generic one when nothing fits)."""
    for g, rx in _COMPILED:
        if rx is not None and rx.search(text or ""):
            return g
    return _BY_ID["generic"]


def get(guide_id):
    return _BY_ID.get(guide_id) or _BY_ID["generic"]


def attempts(guide):
    return min(MAX_ATTEMPTS, len(guide["steps"]))


NOT_FIXED = re.compile(r"^\s*(no|nope|nah|not)\b|\b(still|same|nothing|didn'?t|did not|doesn'?t|does not|not (?:work\w*|fix\w*|help\w*|"
                       r"chang\w*|better|on)|no (?:change|luck|difference)|worse|failed|again|returned|"
                       r"(?:code|beep\w*|error|problem|fault|alarm)\w* (?:came|is|has come) back)\b", re.I)
FIXED = re.compile(r"^\s*(yes|yeah|yep|yebo|ehe)\b|\b(fixed|sorted|solved|resolved|works|worked|working (?:now|again|fine)|"
                   r"it(?:'s| is) (?:on|back|working|fine)|back on|came on|all good|ok(?:ay)? now|problem gone)\b", re.I)


def verdict(text):
    """'fixed', 'not_fixed' or None (unclear) for the customer's reply to 'did that help?'."""
    if NOT_FIXED.search(text or ""):
        return "not_fixed"
    if FIXED.search(text or ""):
        return "fixed"
    return None
