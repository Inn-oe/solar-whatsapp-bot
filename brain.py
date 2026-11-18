"""Conversation brain: reads what the customer typed, understands it (nlu.py) and answers like a person.

The customer can write however they like ("hi, my inverter keeps beeping"); the language model works out what
they want and the bot talks it through: one thing at a time, remembering what was already said, answering side
questions without losing its place, and handing over to the team when it is not sure. Closed choices also come
with buttons or a list (fixed / still the same, send / change, which package, which problem); open details (what
is wrong, the error code, the town, the name) are typed. Every button is a shortcut for words the bot already
understands, so typing always works too.

The three things a customer can want are recorded as an *enquiry* (see db.Enquiry), each with a priority:
  purchase     wants a system, a quote, a price or sizing advice
  enquiry      general questions, calls, documents, order status
  maintenance  a fault, a service, a warranty claim or a complaint
"""
import json
import logging
import os
import random
import re
import threading
import time
from pathlib import Path

import db
import llm
import nlu
import troubleshoot

log = logging.getLogger("solar-bot.brain")
SEED_FILE = Path(__file__).parent / "nlu_seed.json"

_intents = {}  # name -> intent row (category, priority, handler, replies); refreshed after training


def core():
    """The main module (catalog, alerts, lead saving). Imported lazily to avoid a circular import."""
    import app
    return app


# ---------------------------------------------------------------- training data and model
def _faq_slug(title):
    return "faq_" + re.sub(r"[^a-z0-9]+", "_", re.sub(r"[^\w\s]", "", title.lower())).strip("_")[:40]


def _clean_answer(text):
    """Old FAQ answers told people to 'Reply *menu*'; there are no menus any more."""
    text = re.sub(r"\s*Reply \*menu\*[^.]*\.", "", text or "")
    return re.sub(r"\s+", " ", text).strip()


SKIP_FAQ_WORDS = ("cost", "how many panels", "contact", "maintenance")  # covered by dedicated intents


def seed(catalog):
    """Create the starting intents and example phrases (and one intent per catalog FAQ). Safe to repeat:
    it only adds what is missing, so staff edits are never overwritten."""
    data = json.loads(SEED_FILE.read_text(encoding="utf-8"))
    version = int(data.get("version", 1))
    have = db.list_intents()
    if have and int(db.get_setting("nlu_seed_version") or 0) >= version:
        return False
    rows = []
    for it in data["intents"]:
        if it["name"] not in have:
            db.save_intent(it["name"], it["category"], it["priority"], it["handler"], it.get("replies") or [])
        rows += [(it["name"], t) for t in it["examples"]]
    for f in catalog.get("faqs", []):
        title = re.sub(r"^[^\w]+", "", f.get("title") or f["keys"][0]).strip()
        if any(w in title.lower() for w in SKIP_FAQ_WORDS):
            continue
        slug = _faq_slug(title)
        if slug not in have:
            db.save_intent(slug, "enquiry", "low", "reply", [_clean_answer(f["answer"])])
        for k in f["keys"]:
            rows += [(slug, k), (slug, f"tell me about {k}"), (slug, f"i have a question about {k}"),
                     (slug, f"what about {k}")]
        rows.append((slug, title))
    known = db.list_intents()
    for slug, phrases in (data.get("faq_examples") or {}).items():
        if slug in known or any(slug == _faq_slug(re.sub(r"^[^\w]+", "", f.get("title") or f["keys"][0]).strip())
                                for f in catalog.get("faqs", [])):
            rows += [(slug, p) for p in phrases]
    retired = [(name, t) for name, phrases in (data.get("retired_examples") or {}).items() for t in phrases]
    removed = db.delete_seed_examples([r for r in retired if r not in set(rows)]) if retired else 0
    added = db.add_examples_bulk(rows, "seed")
    db.set_setting("nlu_seed_version", str(version))
    log.info("NLU seed: %d new example phrases, %d retired", added, removed)
    return True


QA_SHEET = Path(__file__).parent / "training" / "solar_customer_qa.csv"


def _import_sheet():
    """Load training/solar_customer_qa.csv (question/answer sheet) once per version of the file."""
    if not QA_SHEET.exists():
        return
    import qa_import
    digest = qa_import.file_hash(QA_SHEET) + ":2"   # ":2" = version of the import rules
    if db.get_setting("nlu_qa_hash") == digest:
        return
    qa_import.import_csv(QA_SHEET)
    db.set_setting("nlu_qa_hash", digest)


IDENTITY_REPLIES = [
    "I'm the virtual assistant for {company} 🤖, so no name as such, just call me the {company} assistant. I can "
    "answer questions, arrange quotes and book maintenance, and a real person on our team steps in whenever you "
    "like. Just say \"call me\".",
    "Good question{first_comma}! I'm an automated assistant for {company}, not a person. I can help with prices, "
    "quotes and faults, and I'll bring in someone from the team the moment you ask.",
    "I'm a virtual assistant, {company}'s first point of contact 🤖. Anything I can't handle goes straight to a "
    "real person on the team, and you can ask for one at any time.",
]
_OLD_IDENTITY = "I'm the virtual assistant for {company} 🤖. I can answer questions, sort out quotes and book maintenance"


def _migrate():
    """One-time updates to texts that were already saved in the database by an earlier version."""
    if db.get_setting("nlu_migration") == "2":
        return
    meta = db.list_intents().get("bot_identity")
    if meta and len(meta["replies"]) == 1 and meta["replies"][0].startswith(_OLD_IDENTITY):
        db.save_intent("bot_identity", meta["category"], meta["priority"], meta["handler"], IDENTITY_REPLIES,
                       meta["enabled"])
    db.set_setting("nlu_migration", "2")


def retrain():
    """Rebuild the model from every enabled intent's example phrases. Returns the training summary."""
    global _intents
    intents = db.list_intents()
    ok = {n for n, i in intents.items() if i["enabled"]}
    examples = [(e["intent"], e["text"]) for e in db.list_examples() if e["intent"] in ok]
    # The quick model is trained first and answers immediately. The transformer (if NLU_BACKEND=transformer)
    # loads and embeds every phrase in the background, then replaces it, so start-up and the platform's
    # "Train now" never wait on a model download.
    info = nlu.engine.train(examples, backend="tfidf")
    _intents = intents
    llm.set_menu({n: i for n, i in intents.items() if n in ok}, examples)
    wanted = os.getenv("NLU_BACKEND", "tfidf").lower()
    if wanted == "remote":
        _upgrade_to_transformer(examples, "remote")      # the model runs on the embedding service: no memory needed here
        info = dict(info, note="Connecting to the embedding service in the background. The quick model answers "
                               "until it is ready; check this page again in a minute.")
    elif wanted == "transformer":
        problem = _memory_problem()
        if problem:
            log.warning(problem)
            nlu.engine.info["note"] = problem
            info = dict(info, note=problem)
        else:
            _upgrade_to_transformer(examples)
            info = dict(info, note="The transformer model is loading in the background. The quick model answers "
                                   "until it is ready; check this page again in a few minutes.")
    return info


def _memory_limit_mb():
    """The container's memory limit in MB (Linux cgroups), or None when it cannot be read."""
    for path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            value = Path(path).read_text().strip()
        except OSError:
            continue
        if value.isdigit():
            return int(value) / 1048576
    return None


def _memory_problem():
    """Why the transformer must not be started here, or '' when it is safe.

    The model plus the quick model peak at roughly 600 MB: on a 512 MB server the process is killed by the
    host and restarts in a loop, dropping customer messages. Override the requirement with NLU_MIN_MEMORY_MB."""
    need = int(os.getenv("NLU_MIN_MEMORY_MB", "1500"))
    limit = _memory_limit_mb()
    if limit is not None and limit < need:
        return (f"Transformer not started: this server has {limit:.0f} MB of memory and it needs about {need} MB. "
                "The quick model is answering. Upgrade the hosting plan to enable the transformer.")
    return ""


_upgrade_lock = threading.Lock()


def _upgrade_to_transformer(examples, backend="transformer"):
    if not _upgrade_lock.acquire(blocking=False):
        return  # one is already running; it will pick up the phrases at its next run

    def run():
        try:
            nlu.engine.train(examples, backend=backend)
        except Exception as exc:
            log.exception("%s training failed; staying on the quick model", backend)
            nlu.engine.info["note"] = (f"The {backend} model could not be reached ({type(exc).__name__}); "
                                       "the quick model is answering. Press Train now to try again.")
        finally:
            _upgrade_lock.release()

    threading.Thread(target=run, daemon=True, name="nlu-transformer").start()


def start(catalog):
    """Called once at start-up."""
    try:
        seed(catalog)
        _import_sheet()
        _migrate()
        return retrain()
    except Exception:
        log.exception("NLU start-up failed; the bot will hand every message to the team")
        return {"trained": False}


# ---------------------------------------------------------------- understanding
LEADING_YES_NO = re.compile(r"^\s*(no+|nope|yes|yeah|yep|ok|okay|sure|ehe|aiwa|hongu|kwete)\b[\s,.!-]*", re.I)


EQUIPMENT = re.compile(r"\b(solar|inverter|batter\w*|panels?|geyser|borehole|pump|power|electricity|lights?|system|"
                       r"charger|controller)\b")
NOT_A_FAULT = re.compile(r"\b(buy|quot\w*|price|prices|cost|how much|install\w*|new system|upgrade)\b")


def _looks_like_fault(raw, t):
    """A plain statement about solar equipment that matches a known fault pattern (never a question or a purchase)."""
    if "?" in raw or re.match(r"\s*(can|could|do|does|will|would|is|are|how|what|which|why|when|where|who)\b", raw, re.I):
        return False
    if not EQUIPMENT.search(t) or NOT_A_FAULT.search(t):
        return False
    return troubleshoot.guide_for(t)["id"] != "generic" or bool(FAULT_WORDS.search(t))


def analyze(text, towns=None, existing_customer=False, context=""):
    """Everything the model knows about one message (also used by the 'test a phrase' tool).

    `context` is the bot's previous message, so the language-model fallback can read short answers."""
    res = nlu.engine.predict(text)
    ents = nlu.extract_entities(text, towns)
    t = nlu.clean(text)
    intent = res["intent"] if nlu.engine.understood(res) else None
    via = "model" if intent else None
    # "No I need new system": the leading yes/no answers our question, the rest says what they want
    rest = LEADING_YES_NO.sub("", text, count=1)
    if intent is None and rest != text and len(rest.split()) >= 2:
        res2 = nlu.engine.predict(rest)
        if nlu.engine.understood(res2):
            res, intent, via = res2, res2["intent"], "model"
    # still unsure: ask the large language model (if one is configured) to pick one of our intents
    if intent is None and llm.enabled():
        guess, conf = llm.classify(text, context)
        if guess and conf >= 0.5 and guess in _intents:
            intent, via = guess, "llm"
            res = dict(res, intent=guess, confidence=min(res["confidence"], 0.39))  # low: staff review it later
    # "Marii 3kva" / "how much for 5kva": a price question, even when the model is unsure
    if intent in (None, "qa_ambiguous") and "price" in _intents and PRICE_ASK.search(text) \
            and not FAULT_WORDS.search(t):
        intent, via = "price", "rules"
    # still unsure, but it plainly describes a solar fault ("solar not giving power"): treat it as a fault report
    if intent in (None, "qa_ambiguous") and "maintenance_fault" in _intents and _looks_like_fault(text, t):
        intent, via = "maintenance_fault", "rules"
    # safety words mean a fault report, unless the person is clearly talking about buying something
    category_now = (_intents.get(intent) or {}).get("category")
    if nlu.SAFETY.search(t) and category_now not in ("maintenance", "purchase") and "maintenance_fault" in _intents:
        intent = "maintenance_fault"
    # figures or appliances with no clear intent: treat it as a sizing / buying question
    if intent is None and (ents.get("appliances") or ents.get("kwh")) and "sizing" in _intents:
        intent = "sizing"
    # "I have a fridge, tv and a borehole pump" (a statement, not a question) is a sizing request
    if (len(ents.get("appliances") or []) >= 2 and "sizing" in _intents and "?" not in text
            and not re.match(r"\s*(can|do|does|will|is|are|how|what|which|why)\b", text, re.I)
            and (category_now in (None, "enquiry", "other") or (intent or "").startswith(("faq_", "qa_")))):
        intent = "sizing"
    meta = _intents.get(intent) or {}
    category = meta.get("category", "enquiry")
    priority, reasons = nlu.decide_priority(meta.get("priority", "medium"), category, text, ents, existing_customer)
    return {"intent": intent, "guess": res["intent"], "confidence": round(res["confidence"], 3),
            "similarity": round(res["similarity"], 3), "ranked": res["ranked"], "matched": res["matched"],
            "category": category, "priority": priority, "reasons": reasons, "entities": ents,
            "handler": meta.get("handler", "reply"), "understood": intent is not None, "via": via}


QUESTION_START = re.compile(r"\s*(can|could|do|does|will|would|is|are|how|what|which|why|when|where|who)\b", re.I)
AGREES = re.compile(r"\s*(yes|yeah|yep|ok|okay|sure|please|go ahead|ehe|send)\b", re.I)


# Shona and English question words that need no "?": "marii 3kva", "zvinoita sei", "did you understand"
QUESTION_WORDS = re.compile(r"\b(marii|imarii|zvii|chii|sei|ndeipi|ripi|kupi|rini|how much|how many|"
                            r"did you|do you|can you|could you|have you|what did|what does|what is|whats)\b"
                            r"|^\s*ko\b|\bhere\s*[.!]*\s*$", re.I)   # Shona "ko ..." and "... here" mark a question
PRICE_ASK = re.compile(r"\b(marii|imarii|mari(?:yei)?|how much|price[sd]?|pricing|cost[s]?|kangani|charge[s]?)\b", re.I)
META = re.compile(r"(did you (?:understand|get (?:that|it|me|what)|hear|read)|do you understand|understand me|"
                  r"what did i (?:say|tell you|ask|just say)|you (?:didn'?t|did not|don'?t|do not) (?:understand|answer|get|"
                  r"listen|read)|are you (?:there|listening|following)|anyone there|you there|hello\s*\?|"
                  r"wanzwa|wandinzwa|wanzwisisa|unonzwa|wafamba nazvo|ndati chii)", re.I)


def is_question(text):
    return "?" in text or bool(QUESTION_START.match(text)) or bool(QUESTION_WORDS.search(text))


# ---------------------------------------------------------------- wording helpers
def first_name(name):
    return core().first_name(name)


def say(variants, s, **kw):
    """One of several wordings, never the same one twice in a row, with {placeholders} filled in."""
    variants = [v for v in variants if v] or [""]
    last = s.get("_last_say")
    pool = [v for v in variants if v != last] or variants
    choice = random.choice(pool)
    s["_last_say"] = choice
    kw.setdefault("first_comma", f", {kw['first']}" if kw.get("first") else "")
    try:
        text = choice.format_map(_Safe(kw))
    except (ValueError, IndexError, KeyError, AttributeError):  # a stray brace typed into a reply
        text = choice
    return re.sub(r"\s+([,.!?])", r"\1", text).strip()


class _Safe(dict):
    def __missing__(self, key):
        return ""


def greet_word(text):
    t = nlu.clean(text)
    if "mhoro" in t or "makadii" in t:
        return "Mhoro"
    for word in ("morning", "afternoon", "evening"):
        if word in t:
            return f"Good {word}"
    return random.choice(["Hi", "Hello", "Hi there"])


def listing(items):
    items = list(items)
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1] if items else ""


# ---------------------------------------------------------------- the conversation
class Ctx:
    """Everything a handler needs about the message being answered."""

    def __init__(self, phone, name, text, s, a):
        self.phone, self.name, self.text, self.s, self.a = phone, name, text.strip(), s, a
        self.is_web = phone.startswith("web:")
        self.tap = None
        self.ents = a["entities"]
        self.first = first_name(s.get("slots", {}).get("name") or name)
        c = core()
        self.catalog = c.CATALOG
        self.promise = c.reply_promise()

    def kw(self, **extra):
        return {"first": self.first, "company": self.catalog["company"], "phone": self.catalog["phone"],
                "hours": self.catalog["hours"], "promise": self.promise, **extra}


def respond(phone, name, text, s):
    """The reply for one message: {"text", "options", "button"}. `s` is the saved conversation state (mutated).

    `options` are buttons (up to 3) or list rows (up to 10) for the question the reply ends with; empty when the
    reply needs a typed answer or none at all."""
    s.pop("_ui", None)
    reply = _respond(phone, name, text, s)
    reply = reply.get("text", "") if isinstance(reply, dict) else str(reply or "")
    ui = s.pop("_ui", None) or {}
    options = ui.get("options") or []
    s["_last_opts"] = {_norm_title(o["title"]): o["id"][4:] for o in options}
    s["_last_reply"] = reply[:400]
    return {"text": reply, "options": options, "button": ui.get("button", "")}


def _respond(phone, name, text, s):
    c = core()
    towns = [t for t in (c.CATALOG.get("towns") or []) if isinstance(t, str)]
    # a tapped button arrives as "btn:<key>"; typing a button's title ("Still the same") counts as tapping it
    tap = text.strip()[4:] if text.strip().startswith("btn:") else (s.get("_last_opts") or {}).get(_norm_title(text))
    if tap:
        text = _tap_text(tap, c)
    a = analyze(text, towns, context=s.get("_last_reply", ""))
    ctx = Ctx(phone, name, text, s, a)
    ctx.tap = tap
    s.pop("offer", None)
    if tap:
        direct = _direct_tap(ctx)
        if direct is not None:
            return direct

    # 0. mid-quote, asked what the system should run, but the customer reports a fault ("my inverter is having an
    #    error"): that is a maintenance report, not the quotation's need, so switch flows instead of quoting it
    if s.get("flow") == "purchase" and s.get("awaiting") == "need" and FAULT_WORDS.search(nlu.clean(text)) \
            and not is_question(text):
        reset_flow(s)
        a = dict(a, intent=a["intent"] or "maintenance_fault", category="maintenance", handler="maintenance",
                 priority="high")
        ctx = Ctx(phone, name, text, s, a)
        return _h_maintenance(ctx)

    # "did you understand?" / "what did I say?": answer like a person, with what we have so far
    if not tap and META.search(text):
        return _meta_reply(ctx)
    # a price question in the middle of a quotation: give the price, keep the size, carry on
    if (not tap and s.get("flow") == "purchase" and PRICE_ASK.search(text)
            and s.get("awaiting") not in ("contact", "name")):
        return _price_in_quote(ctx)

    # 1. is the customer answering the question we just asked?
    filled = _fill_awaiting(ctx)
    if filled == "more":
        return _keep_detail_and_reask(ctx)
    if filled == "change":
        s["awaiting"] = "change"
        return _ask(ctx, "change")
    if filled == "repeat_new":
        s.pop("awaiting", None)
        return _advance(ctx)
    if filled == "repeat_edit":
        s["slots"] = dict((s.get("_last_quote") or {}).get("slots") or {})
        s["_editing_quote"] = True
        return _ask(ctx, "change")
    if filled == "edit":
        s.pop("awaiting", None)
        return _ask(ctx, "confirm", filled=True)
    if filled:
        return _continue_flow(ctx)

    intent, handler = a["intent"], a["handler"]
    # "yes please" / "send the quote" right after we offered a quotation accepts the offer
    accepts_offer = (s.get("pending") == "quote_offer" and intent in (None, "affirm", "buy_system", "site_visit")
                     and not is_question(text) and bool(AGREES.match(text)))
    if intent is None and not accepts_offer:
        if s.get("flow") in ("purchase", "maintenance") and s.get("awaiting") in RESUME:
            return _not_understood_in_flow(ctx)
        return _not_understood(ctx)
    if accepts_offer:
        handler = "affirm"
    s["_unknown"] = 0
    if a["confidence"] < 0.4:  # answered, but unsure: let staff confirm it so the bot learns
        try:
            db.add_unmatched(phone, text, intent, a["confidence"])
        except Exception:
            log.exception("could not queue message for review")

    # 2. record what they want (purchase / enquiry / maintenance) with its priority
    _record_enquiry(ctx)

    # 3. answer
    awaiting_before = s.get("awaiting")
    reply = _HANDLERS.get(handler, _h_reply)(ctx)

    # 4. answered a side question in the middle of a flow: gently go back to it (unless this reply moved the
    #    flow on and already asks the next question itself)
    if (s.get("flow") and s.get("awaiting") and s.get("awaiting") == awaiting_before
            and handler not in ("purchase", "maintenance")):
        reply += "\n\n" + _ask(ctx, s["awaiting"], resumed=True)
    return reply


def _record_enquiry(ctx):
    a = ctx.a
    if a["category"] == "other":
        return
    try:
        db.upsert_enquiry(ctx.phone, ctx.s.get("slots", {}).get("name") or ctx.name, a["category"], a["intent"],
                          a["priority"], ctx.text, reasons=a["reasons"], confidence=a["confidence"])
    except Exception:
        log.exception("could not record enquiry")


# ---- not understood ---------------------------------------------------------------------------
def _not_understood(ctx):
    s, c = ctx.s, core()
    raw = ctx.text
    try:
        db.add_unmatched(ctx.phone, raw, ctx.a["guess"] or "", ctx.a["confidence"])
    except Exception:
        log.exception("could not store unmatched message")
    s["_unknown"] = s.get("_unknown", 0) + 1
    if len(raw) <= 2 or not re.search(r"[A-Za-z]", raw):
        return say(["👍 Anything else I can help with?", "Okay! Let me know if you need anything."], s)
    if s["_unknown"] == 1:
        _offer(s, UNSURE_MENU)
        return say(["Sorry{first_comma}, I didn't quite catch that. Are you looking to buy a solar system, "
                    "get a price, or do you need help with a system you already have?",
                    "I want to make sure I get this right{first_comma}. Could you tell me a little more? "
                    "For example, are you after a new system, a price, or help with an existing one?"], s,
                   **ctx.kw())
    # second miss in a row: stop guessing and give it to a person
    s["_unknown"] = 0
    c.sig(ctx.phone, "unanswered", c.clip(raw, 200))
    c.sig(ctx.phone, "needs_human", True)
    try:
        db.upsert_enquiry(ctx.phone, ctx.name, "enquiry", "unknown", "medium", raw,
                          summary=f'Question the bot could not answer: "{c.clip(raw, 200)}"', confidence=0)
    except Exception:
        log.exception("could not record enquiry")
    if ctx.is_web:
        return say(["I don't want to guess on that one{first_comma}. Please use *Request a quote* on this page "
                    "and leave your number, and a person will reply {promise}."], s, **ctx.kw())
    c.alert_team(ctx.phone, s.get("slots", {}).get("name") or ctx.name, "unanswered",
                 f'They asked: "{c.clip(raw, 300)}"', s)
    return say(["I don't want to guess on that one{first_comma}, so I've passed it to our team and a person will "
                "reply here {promise}. Is there anything else I can help with meanwhile?"], s, **ctx.kw())


# ---- simple handlers --------------------------------------------------------------------------
def _intent_replies(ctx):
    return (_intents.get(ctx.a["intent"]) or {}).get("replies") or []


def _h_reply(ctx):
    replies = _intent_replies(ctx)
    s = ctx.s
    if not replies:
        return _not_understood(ctx)
    text = say(replies, s, **ctx.kw(greet=greet_word(ctx.text)))
    name = ctx.a["intent"]
    if name.startswith("faq_"):
        topic = "payment" if any(w in name for w in ("pay", "deposit")) else name[4:]
        core().sig(ctx.phone, "faq", topic)
        if not s.get("_offered_quote") and random.random() < 0.4:
            s["_offered_quote"] = True
            _set_offer(s)
            text += "\n\n" + say(["If you'd like, I can arrange a free quotation for you.",
                                  "Would you like me to prepare a free quotation?"], s)
    return text


def _h_greeting(ctx):
    s = ctx.s
    reset_flow(s)
    _offer(s, MENU)
    return say(_intent_replies(ctx), s, **ctx.kw(greet=greet_word(ctx.text)))


def _h_thanks(ctx):
    return say(_intent_replies(ctx), ctx.s, **ctx.kw())


def _h_goodbye(ctx):
    reset_flow(ctx.s)
    return say(_intent_replies(ctx), ctx.s, **ctx.kw())


def _h_affirm(ctx):
    s = ctx.s
    if s.get("pending") == "quote_offer":
        s.pop("pending")
        ctx.a["intent"], ctx.a["category"] = "buy_system", "purchase"
        return _h_purchase(ctx)
    return say(_intent_replies(ctx), s, **ctx.kw())


def _h_deny(ctx):
    s = ctx.s
    s.pop("pending", None)
    if s.get("flow"):
        reset_flow(s)
        return say(["No problem{first_comma}, I've stopped that. Anything else I can help with?"], s, **ctx.kw())
    return say(_intent_replies(ctx), s, **ctx.kw())


def _h_contact(ctx):
    c = ctx.catalog
    s = ctx.s
    town = ctx.ents.get("location")
    asks_for_details = re.search(r"\b(address|where are you|located|office|shop|hours|open|close|phone|number|email|"
                                 r"contact|call|whatsapp)\b", nlu.clean(ctx.text))
    if town and not asks_for_details:
        known = [t.lower() for t in (c.get("towns") or [])]
        if town.lower() in known:
            _set_offer(s)
            return (f"Yes, we install in {town}. Would you like me to arrange a free site visit and a quotation?")
        return (f"I'm not sure we cover {town} yet, but I'll ask the team to confirm. What would you like solar "
                "for, so I can pass it on?")
    served = ""
    if town and town.lower() not in [t.lower() for t in (c.get("towns") or [])]:
        served = f"\n\nI'm not sure we cover {town} yet. A team member will confirm if you tell me your area."
    return (f"You can reach {c['company']} here:\n📞 {c['phone']}\n✉️ {c['email']}\n📍 {c['address']}\n"
            f"🕒 {c['hours']}{served}\n\nIs there something I can help with right now?")


def _h_packages(ctx):
    c, s = core(), ctx.s
    size = ctx.ents.get("size_kva")
    c.sig(ctx.phone, "viewed_packages")
    town = s.get("_town", "")
    pkgs = ctx.catalog["packages"]
    if size:
        p = min(pkgs, key=lambda p: abs((c.package_size(p) or p["daily_kwh"] / 3) - size))
        _set_offer(s)
        note = c.market_note(p, town)
        c.sig(ctx.phone, "package", p["name"])
        return (f"For a {size:g}kVA system, our closest is the *{p['name']}* at {c.price_text(p, town)}.\n\n"
                f"⚡ Powers: {p['powers']}\n📦 Includes: {p['includes']}"
                + (f"\n\n{note}" if note else "")
                + "\n\nWould you like me to arrange a free quotation and site visit?")
    lines = "\n".join(f"• *{p['name']}*: {c.price_text(p, town)}" for p in pkgs)
    _set_offer(s)
    return say(["Here are our current packages{first_comma}:\n\n" + lines.replace("{", "{{").replace("}", "}}")
                + "\n\nTell me what you want to run (fridge, TV, lights, pump...) and I'll suggest the right one, "
                  "or I can arrange a free quotation."], s, **ctx.kw())


def _h_sizing(ctx):
    c, s = core(), ctx.s
    e = ctx.ents
    if (e.get("size_kva") or e.get("budget") or (e.get("location") and e.get("appliances"))) \
            and not is_question(ctx.text):
        # a size, budget or town alongside the appliances means they are ready to buy, not just asking
        ctx.a["intent"], ctx.a["category"] = "buy_system", "purchase"
        return _h_purchase(ctx)
    kwh = e.get("kwh") or e.get("appliance_kwh")
    if not kwh:
        s["awaiting"] = "sizing_need"
        s["flow"] = "sizing"
        return say(["Happy to help you work that out{first_comma}. What would you like to keep running, "
                    "for example fridge, TV, lights, WiFi, a geyser or a borehole pump? "
                    "Or tell me your monthly ZESA units (kWh) if you have them."], s, **ctx.kw())
    return _sized(ctx, kwh)


def _sized(ctx, kwh):
    c, s = core(), ctx.s
    p = c.recommend(kwh)
    reset_flow(s)
    s["_recommended"] = p["name"]
    c.sig(ctx.phone, "sized_kwh", round(kwh, 1))
    c.sig(ctx.phone, "recommended", p["name"])
    town = s.get("_town", "")
    items = ctx.ents.get("appliances")
    basis = f"For {listing(items)} you'd use about {kwh:g} kWh a day" if items else f"For about {kwh:g} kWh a day"
    note = c.market_note(p, town)
    _set_offer(s)
    return (f"{basis}, so I'd suggest the *{p['name']}* ({c.price_text(p, town)}).\n\n⚡ It powers: {p['powers']}."
            + (f"\n\n{note}" if note else "")
            + "\n\nThat's an estimate: a technician confirms the final size on a free site visit. "
              "Would you like me to arrange a quotation?")


def _h_answer_team(ctx):
    """Answer from the training sheet, then collect the details the team needs and hand it over."""
    s = ctx.s
    replies = _intent_replies(ctx)
    if not replies:
        return _not_understood(ctx)
    text = say(replies, s, **ctx.kw())
    offer = s.get("pending")
    reset_flow(s)
    if offer:
        s["pending"] = offer
    s["flow"], s["awaiting"] = "details", "details"
    s["_detail_intent"], s["_detail_category"] = ctx.a["intent"], ctx.a["category"]
    _slots(s)["first_text"] = core().clip(ctx.text, 300)
    return text


def _h_answer_alert(ctx):
    """Answer from the training sheet and tell the team straight away (complaints, safety, escalations)."""
    c, s = core(), ctx.s
    text = say(_intent_replies(ctx), s, **ctx.kw()) if _intent_replies(ctx) else _not_understood(ctx)
    if not ctx.is_web:
        name = _slots(s).get("name") or ctx.name
        c.sig(ctx.phone, "needs_human", True)
        c.sig(ctx.phone, "unanswered", c.clip(ctx.text, 200))
        c.save_lead(ctx.phone, name, kind="callback", need=c.clip(ctx.text, 500))
        c.set_follow_up(ctx.phone, 0)
        c.alert_team(ctx.phone, name, "callback", f"[{ctx.a['priority'].upper()}] {c.clip(ctx.text, 300)}", s)
    return text


def _finish_details(ctx):
    """The customer sent the details we asked for: pass everything to the team."""
    c, s = core(), ctx.s
    slots = _slots(s)
    intent, category = s.pop("_detail_intent", ""), s.pop("_detail_category", "enquiry")
    meta = _intents.get(intent) or {}
    asked = slots.get("first_text", "")
    detail = c.clip(ctx.text, 500)
    summary = c.clip(f"{asked}  Details: {detail}", 500)
    name = slots.get("name") or ctx.name
    kind = {"maintenance": "maintenance", "purchase": "quote"}.get(category, "callback")
    if category == "purchase":
        c.sig(ctx.phone, "quote_requested")
        c.sig(ctx.phone, "need", c.clip(f"{asked} {detail}", 300))
        c.sig(ctx.phone, "quote_completed")
    elif category == "maintenance":
        c.sig(ctx.phone, "maintenance_issue", c.clip(f"{asked} {detail}", 300))
    else:
        c.sig(ctx.phone, "needs_human", True)
        c.sig(ctx.phone, "unanswered", c.clip(asked, 200))
    try:
        db.upsert_enquiry(ctx.phone, name, category, intent, meta.get("priority", "medium"), ctx.text,
                          summary=summary, reasons=["Sent the details we asked for"], confidence=1.0)
    except Exception:
        log.exception("could not record enquiry")
    if not ctx.is_web:
        c.save_lead(ctx.phone, name, kind=kind, need=c.clip(summary, 1000), location=ents_location(ctx))
        c.set_follow_up(ctx.phone, 0 if meta.get("priority") in ("high", "urgent") else 1)
        c.alert_team(ctx.phone, name, {"quote": "quote", "maintenance": "maintenance"}.get(kind, "callback"),
                     c.clip(summary, 400), s)
    first = first_name(name)
    reset_flow(s)
    extra = (" Please also leave your phone number with the *Request a quote* form on this page so they can reach you."
             if ctx.is_web else "")
    return say(["Thank you{first_comma}, I've passed this to our team and someone will contact you "
                "{promise}." + extra,
                "Thanks{first_comma}! That's with our team now, and they'll get back to you {promise}." + extra],
               s, **ctx.kw())


def ents_location(ctx):
    return ctx.ents.get("location") or ""


def _h_human(ctx):
    c, s = core(), ctx.s
    intent = ctx.a["intent"]
    slots = s.get("slots", {})
    name = slots.get("name") or ctx.name
    if ctx.is_web:
        return say(["Of course{first_comma}. Please use the *Request a quote* form on this page and leave your "
                    "number, or call {phone}."], s, **ctx.kw())
    if intent in ("talk_to_human", "complaint"):
        c.sig(ctx.phone, "callback")
    else:
        c.sig(ctx.phone, "needs_human", True)
        c.sig(ctx.phone, "unanswered", c.clip(ctx.text, 200))
    c.save_lead(ctx.phone, name, kind="callback", need=c.clip(ctx.text, 500))
    c.set_follow_up(ctx.phone, 0)
    c.alert_team(ctx.phone, name, "callback", f"{intent.replace('_', ' ').title()}: {c.clip(ctx.text, 300)}", s)
    replies = _intent_replies(ctx) or [
        "Sure{first_comma}. A member of our team will message you here {promise}. You can also call {phone}."]
    return say(replies, s, **ctx.kw())


# ---------------------------------------------------------------- multi-step flows
FLOW_TITLES = {"purchase": "buy", "maintenance": "maintenance"}


def reset_flow(s):
    for k in ("flow", "awaiting", "slots", "pending"):
        s.pop(k, None)


def _slots(s):
    return s.setdefault("slots", {})


def _add_detail(slots, key, value):
    value = value.strip()
    have = slots.get(key, "")
    if value and value.lower() not in have.lower():
        slots[key] = (have + "; " if have else "") + value


SYMPTOM = re.compile(
    r"\b(beep\w*|error|fault|code|alarm|flash\w*|blink\w*|trip\w*|overload\w*|not (?:charg\w*|work\w*|switch\w*|"
    r"hold\w*|turn\w*|giving|supply\w*)|no (?:power|output|electricity|lights?|charge)|stopp?\w*|dead|flat|drain\w*|"
    r"run\w* out|finish\w*|last|noise|hot|warm|blank|display|shut\w*|cut\w*|off|leak\w*|crack\w*|broken|damag\w*|"
    r"smok\w*|burn\w*|spark\w*|low|weak|slow|dim|only|fell|dropp\w*|keeps?|since|every|[ef]-?\d{1,3})\b")
CODE_RX = re.compile(r"\b(?:e|f|err|error)[ -]?\d{1,3}\b")


FAULT_WORDS = re.compile(r"\b(error|fault|beep\w*|alarm|code|flash\w*|blink\w*|smok\w*|burn\w*|spark\w*|broken|dead|"
                         r"not (?:charg\w*|work\w*|switch\w*|hold\w*|turn\w*|giving))\b")


def _specific_issue(text, a):
    """True when the message says what is wrong ('keeps beeping', 'batteries not charging'), not just that
    something is wrong ('my solar is malfunctioning')."""
    if a["priority"] == "urgent" or a["intent"] == "warranty_claim":
        return True
    t = nlu.clean(text)
    return bool(SYMPTOM.search(t)) or len(t.split()) >= 9


def _gather(ctx, issue_text=True):
    """Pull everything useful out of the message into the flow's slots. `issue_text` is False when the
    message was only the answer to another question (e.g. the town), so it is not mistaken for the problem."""
    s, e = ctx.s, ctx.ents
    slots = _slots(s)
    if e.get("location"):
        # keep a fuller address already given ("Borrowdale, Harare") rather than cutting it to the town
        if e["location"].lower() not in slots.get("location", "").lower():
            slots["location"] = e["location"]
        s["_town"] = e["location"]
    if e.get("phone"):
        slots["contact"] = e["phone"]
    if e.get("name"):
        slots["name"] = e["name"]
    kind = s.get("flow")
    if kind == "purchase":
        bits = []
        if e.get("size_kva"):
            bits.append(f"{e['size_kva']:g}kVA system")
            slots["size_kva"] = e["size_kva"]
        if e.get("appliances"):
            bits.append("to run " + listing(e["appliances"]))
        if e.get("kwh"):
            bits.append(f"about {e['kwh']:g} kWh/day")
        if e.get("budget"):
            bits.append(f"budget around USD {e['budget']:,}")
        if e.get("products") and not e.get("appliances"):
            bits.append(listing(e["products"]))
        if ctx.a["intent"] == "site_visit":
            bits.append("wants a site visit")
        if ctx.a["intent"] == "upgrade_system":
            bits.append("upgrade of an existing system: " + core().clip(ctx.text, 150))
        for p in ctx.catalog["packages"]:
            if p["name"].lower() in nlu.clean(ctx.text):
                bits.append(p["name"])
        for b in bits:
            _add_detail(slots, "need", b)
        if e.get("kwh") or e.get("appliance_kwh"):
            slots["kwh"] = e.get("kwh") or e["appliance_kwh"]
    elif kind == "maintenance":
        t = ctx.text
        code = CODE_RX.search(nlu.clean(t))
        if code and not slots.get("inverter_code"):
            slots["inverter_code"] = code.group(0).upper().replace(" ", "")   # said up front: do not ask again
        if ctx.a["intent"] == "warranty_claim":
            slots["no_tshoot"] = True   # a warranty claim goes straight to a person
        if nlu.SAFETY.search(nlu.clean(t)):
            slots["safety"] = True      # smoke, fire, sparks, shock: a person and safety advice, not DIY fixes
        if ctx.a["category"] == "maintenance":
            rank = {"low": 0, "medium": 1, "high": 2, "urgent": 3}
            if rank.get(ctx.a["priority"], 1) >= rank.get(slots.get("priority", "low"), 0):
                slots["priority"] = ctx.a["priority"]
        if not issue_text:
            pass
        elif ctx.a["intent"] == "maintenance_service":
            slots["routine"] = True
            _add_detail(slots, "issue", "Service / inspection requested: " + core().clip(t, 200))
        elif _specific_issue(t, ctx.a):
            _add_detail(slots, "issue", core().clip(t, 300))
        else:
            # "my solar is malfunctioning": remember it, but ask what is actually happening
            _add_detail(slots, "reported", core().clip(t, 300))


NO_CODE = "no error code or warning shown"


def _can_troubleshoot(s):
    """A plain fault gets first-line help from the bot; services, warranty claims and safety risks go to a person."""
    sl = _slots(s)
    return (s.get("flow") == "maintenance" and not sl.get("routine") and not sl.get("no_tshoot")
            and not sl.get("safety"))


def _missing(ctx):
    s = ctx.s
    slots = _slots(s)
    if s["flow"] == "maintenance" and slots.get("tshoot_result") == "fixed":
        return None   # solved: nothing more to collect
    if s["flow"] == "purchase":
        order = ["need", "location"]
    elif _can_troubleshoot(s):
        # like a technician: what does the inverter show, try two fixes, and only then collect where and who
        order = ["issue", "inverter_code", "tshoot", "location"]
    else:
        order = ["issue", "location"] + ([] if slots.get("routine") else ["inverter_code"])
    for slot in order:
        if slot == "tshoot":
            if not slots.get("tshoot_done"):
                return slot
        elif not slots.get(slot):
            return slot
    if ctx.is_web and not slots.get("contact"):
        return "contact"
    if not slots.get("name") and not first_name(ctx.name):
        return "name"
    if s["flow"] == "purchase" and not slots.get("confirmed"):
        return "confirm"   # read the request back before it reaches the sales team
    return None


ASKS = {
    "need": ["Happy to help with that{first_comma}! What would you like the system to run? For example fridge, "
             "TV, lights, a borehole pump, or the size if you know it (like 5kVA).",
             "Sure{first_comma}. So I can size it properly, what do you want to power: lights, fridge, TV, geyser, pump?"],
    "issue": ["Sorry to hear that{first_comma}, let's get it sorted. What is the system doing: beeping, an "
              "error code, no power, batteries not holding? Pick one below or describe it in your own words, and "
              "tell me roughly when it started.",
              "Oh no, that's frustrating{first_comma}. What are you seeing? Choose the closest problem below, or "
              "type what's happening (any beeping, error code, lights off...) and when it began."],
    "location": ["Which town or suburb is it in?", "Whereabouts are you, which town or suburb?",
                 "And which town or suburb are you in?"],
    "inverter_code": ["Does the inverter show an error code, beep or flash a warning light? If there's a code on "
                      "the display, type it (for example E05). Otherwise tap the option that fits."],
    "contact": ["What's the best phone number (WhatsApp if possible) for our team to reach you on?"],
    "name": ["And may I have your name please?", "Lastly, what name should I put on the request?"],
    "sizing_need": ["What would you like to keep running: fridge, TV, lights, geyser, pump? Or tell me your "
                    "monthly kWh if you know it."],
}
RESUME = {"details": "When you can, send me those details and I'll pass them to our team.",
          "need": "Back to your request: what would you like the system to run?",
          "issue": "Back to your problem: what exactly is happening with the system?",
          "location": "Back to your request: which town or suburb are you in?",
          "inverter_code": "Back to the inverter: does it show an error code, beep or flash a warning light?",
          "tshoot": "Back to the fix we were trying: did that help, or is it still the same?",
          "confirm": "Back to your request: shall I send it to our team?",
          "repeat": "Back to your quotation: is it for another system, or a change to the one you sent?",
          "change": "Back to your request: what would you like to change?",
          "contact": "Back to your request: what phone number can we reach you on?",
          "name": "Back to your request: what name should I use?",
          "sizing_need": "Back to sizing: what would you like to keep running?"}


def _ask(ctx, slot, resumed=False, filled=False):
    """The next question. `filled` means the customer just answered the previous one, so acknowledge it first
    (never say 'thanks' before they have given us anything)."""
    ctx.s["awaiting"] = slot
    _slot_ui(ctx, slot)
    if resumed:
        return RESUME[slot]
    if slot == "tshoot":
        return _tshoot_message(ctx)
    ack = ""
    if filled:
        ack = say(["Thanks.", "Got it.", "Okay, thanks.", "Noted."], ctx.s) + " "
    if slot == "confirm":
        return ack + _confirm_text(ctx)
    if slot == "change":
        return ("No problem. What would you like to change? For example the town, or what the system should run "
                "(like \"fridge, TV and a pump\" or \"10kVA\").")
    slots = _slots(ctx.s)
    result = slots.get("tshoot_result")
    if result in ("failed", "human") and not slots.get("handoff_said"):
        slots["handoff_said"] = True   # the fixes did not work: say so once, then collect what the technician needs
        ack = ("I'm sorry that didn't solve it. I'll hand this to one of our technicians, with everything we "
               "tried so they don't repeat it. " if result == "failed" else
               "Of course. I'll hand this to one of our technicians, with everything we've tried so far. ")
    return ack + say(ASKS[slot], ctx.s, **ctx.kw())


def _tshoot_text(slots):
    """What the customer described, for choosing the right troubleshooting guide."""
    code = slots.get("inverter_code", "")
    return " ".join(x for x in (slots.get("issue"), slots.get("reported"), "" if code == NO_CODE else code) if x)


def _tshoot_message(ctx):
    """The next fix to try, worded the way a technician would walk a customer through it."""
    s = ctx.s
    st = s.get("tshoot")
    if not st:
        st = s["tshoot"] = {"guide": troubleshoot.guide_for(_tshoot_text(_slots(s)))["id"], "step": 0}
    guide = troubleshoot.get(st["guide"])
    i = min(st["step"], len(guide["steps"]) - 1)
    intro = ("Before I send a technician, here's a quick fix to try.\n\n" if i == 0
             else "Thanks for trying that. One more thing to try:\n\n")
    step = re.sub(r"\s(\d)\) ", r"\n\1) ", guide["steps"][i])   # numbered checks on their own lines
    return intro + step + "\n\nDid that fix it? Tap below, or tell me what you see now."


NOT_PLACE_WORDS = {"no", "not", "yes", "just", "is", "are", "was", "were", "the", "my", "it", "its", "i", "and", "but",
                   "because", "when", "why", "what", "how", "have", "has", "do", "does", "did", "can", "will", "error",
                   "dead", "off", "on", "help", "please", "ok", "okay", "thanks", "thank", "you", "also", "still",
                   "since", "beeping", "working", "charging", "power", "inverter", "battery", "batteries", "panels",
                   "solar", "light", "lights", "fridge", "tv", "need", "want", "got", "had", "keeps", "stopped"}


def _looks_like_place(raw):
    """A short phrase made of place-like words: 'Borrowdale, Harare' yes, 'no error just dead' no."""
    if not _looks_like_answer(raw) or len(raw.split()) > 5:
        return False
    words = re.findall(r"[a-z']+", raw.lower())
    return bool(words) and not any(w in NOT_PLACE_WORDS for w in words)


def _looks_like_answer(raw):
    return "?" not in raw and len(raw.split()) <= 6 and len(re.findall(r"[A-Za-z]", raw)) >= 2


def _fill_awaiting(ctx):
    """If we asked a question, try to read this message as the answer. True when it was one."""
    s = ctx.s
    slot = s.get("awaiting")
    if not slot or not s.get("flow"):
        return False
    a, e, raw = ctx.a, ctx.ents, ctx.text
    slots = _slots(s)
    if slot == "tshoot":
        if ctx.tap in ("fixed", "notfixed"):
            v = "fixed" if ctx.tap == "fixed" else "not_fixed"
        else:
            # a safety risk or a request for a person overrides the fix-it steps
            if (nlu.SAFETY.search(nlu.clean(raw))
                    or a["intent"] in ("talk_to_human", "complaint", "greeting", "goodbye", "off_topic")):
                return False
            v = troubleshoot.verdict(raw)
            if v is None or (v == "fixed" and is_question(raw)):
                return False if is_question(raw) else "more"   # info or a bare "ok": keep it, ask whether it worked
        st = s.setdefault("tshoot", {"guide": troubleshoot.guide_for(_tshoot_text(slots))["id"], "step": 0})
        guide = troubleshoot.get(st["guide"])
        step = st.get("step", 0)
        if v == "fixed":
            slots["tshoot_result"] = "fixed"
            slots["fixed_by"] = f"{guide['title']}, step {step + 1}"
            return True
        slots.setdefault("tried", []).append(f"{guide['title']} step {step + 1}: \"{core().clip(raw, 80)}\"")
        st["step"] = step + 1
        if st["step"] >= troubleshoot.attempts(guide):
            slots["tshoot_done"] = True
            slots["tshoot_result"] = "failed"
        return True
    if slot == "repeat":
        if ctx.tap == "new_quote" or re.search(r"\b(another|new|second|different|other|extra|more)\b", raw, re.I):
            return "repeat_new"
        if ctx.tap == "edit_quote" or re.search(r"\b(change|update|edit|same|last|that one|first|previous)\b", raw, re.I):
            return "repeat_edit"
        if is_question(raw):
            return False
        return "repeat_new" if _apply_change(ctx) else False   # new details typed: treat it as another system
    if slot == "confirm":
        if ctx.tap == "send" or (not is_question(raw) and CONFIRMS.match(raw) and not DISAGREES.search(raw)):
            slots["confirmed"] = True
            return True
        if ctx.tap == "change" or DISAGREES.match(raw):
            return "change"
        if is_question(raw):
            return False
        return "edit" if _apply_change(ctx) else False   # new details typed straight away ("make it 10kVA")
    if slot == "change":
        if is_question(raw) or ctx.tap in ("send", "change"):
            if ctx.tap == "send":
                slots["confirmed"] = True
                return True
            return False
        return "edit" if _apply_change(ctx, force=True) else False
    if slot == "inverter_code":
        # "no" is a real answer here (no code, no beeps), not a refusal to continue
        if re.match(r"\s*(no|none|nope|nothing|not sure|dont know|don't know)\b", raw, re.I):
            slots["inverter_code"] = NO_CODE
            if len(raw.split()) >= 3:   # "no error just dead": the extra words describe the fault
                _add_detail(slots, "issue", core().clip(raw, 200))
            return True
        if is_question(raw) or not re.search(r"[A-Za-z0-9]", raw):
            return False
        slots["inverter_code"] = core().clip(raw, 200)
        return True
    if slot == "location" and e.get("location") and "?" not in raw and len(raw.split()) <= 8:
        slots["location"] = raw.strip(" .,!").title() if len(raw.split()) <= 4 else e["location"]
        s["_town"] = e["location"]
        return True
    # a clear greeting/thanks/etc. is not an answer; neither is a different, confident request
    # (a weak FAQ match that is not a question is still the details we asked for: "1HP Lorentz pump, 4 hours a day"
    # looks a little like the pump FAQ, but it is the customer's answer)
    faq = (a["intent"] or "").startswith("faq_")
    if a["intent"] in ("greeting", "thanks", "goodbye", "deny", "talk_to_human", "complaint", "price", "contact_info",
                       "bot_identity", "off_topic") or (faq and (slot != "details" or a["confidence"] >= 0.5
                                                                  or is_question(raw))):
        return False
    if slot == "details":
        # anything that is not a question is taken as the details we asked for
        confident_other = a["understood"] and a["confidence"] >= 0.6 and a["intent"] != "affirm"
        # "yes please send quote" answers the earlier offer; it is not the details we asked for
        agrees = (a["intent"] in ("affirm", "buy_system", "site_visit")
                  or re.match(r"\s*(yes|yeah|yep|ok|okay|sure|please|go ahead|ehe)\b", raw, re.I))
        # appliance / size / location / phone details are exactly what the question asked for
        has_details = any(e.get(k) for k in ("appliances", "kwh", "size_kva", "budget", "products", "location", "phone"))
        return ("?" not in raw and len(re.findall(r"[A-Za-z0-9]", raw)) >= 2
                and (has_details or not confident_other) and not agrees)
    if slot == "sizing_need":
        if is_question(raw):
            return False   # "will 5kVA run a borehole pump?" is a question, not the list we asked for
        kwh = e.get("kwh") or e.get("appliance_kwh")
        return bool(kwh)
    sure_of_other = a["understood"] and a["confidence"] >= 0.6 and a["intent"] not in (None, "affirm")
    if slot == "location":
        # a short, non-question reply to "where?" is the answer ("Borrowdale, Harare"), unless the model is
        # confident it is something else entirely or it is plainly not a place ("no error just dead")
        if _looks_like_place(raw) and (e.get("location") or not sure_of_other):
            place = raw.strip(" .,!").title()
        else:
            place = e.get("location", "")
        if not place:
            has_detail = any(e.get(k) for k in ("appliances", "kwh", "size_kva", "budget", "products", "phone"))
            if (not is_question(raw) and len(raw.split()) >= 2
                    and (s.get("flow") == "maintenance" or has_detail)
                    and a["intent"] not in ("greeting", "thanks", "goodbye", "deny", "talk_to_human", "complaint")):
                return "more"   # more detail about the request: keep it and ask for the town again
            return False
        slots["location"] = place
        s["_town"] = place
        return True
    if slot == "contact":
        phone = e.get("phone") or (raw if len(re.sub(r"\D", "", raw)) >= 7 else "")
        if not phone:
            return False
        slots["contact"] = phone.strip()
        return True
    if slot == "name":
        name = e.get("name") or (re.sub(r"[^A-Za-z' -]", "", raw).strip().title() if _looks_like_answer(raw) and not sure_of_other else "")
        if not name:
            return False
        slots["name"] = name
        return True
    if slot == "need":
        if re.search(r"\b(not sure|dont know|do not know|advise|any|you decide|help me choose|recommend)\b", nlu.clean(raw)):
            _add_detail(slots, "need", "Not sure: wants advice")
            return True
        if is_question(raw):
            return False   # "will 5kVA run a borehole pump?" is a question for us, not our answer
        if any(e.get(k) for k in ("appliances", "kwh", "size_kva", "budget", "products")):
            _gather(ctx)
            return True
        if len(raw.split()) >= 2 and "?" not in raw and a["intent"] in (None, "buy_system", "sizing"):
            _add_detail(slots, "need", core().clip(raw, 200))
            return True
        return False
    if slot == "issue":
        if "?" not in raw and a["intent"] not in ("affirm",) and (len(raw.split()) >= 2 or SYMPTOM.search(nlu.clean(raw))):
            _add_detail(slots, "issue", core().clip(raw, 300))
            return True
        return False
    return False


def _summary_so_far(ctx):
    """What the bot has understood of the current request, in one line."""
    s = ctx.s
    slots = _slots(s)
    if s.get("flow") == "purchase":
        bits = [slots["need"]] if slots.get("need") else []
        est = _estimate_line(ctx)
        if est:
            bits.append(f"closest package {est}")
        if slots.get("location"):
            bits.append(f"in {slots['location']}")
        return "your quotation request" + (": " + ", ".join(bits) if bits else "")
    if s.get("flow") == "maintenance":
        issue = "; ".join(x for x in (slots.get("reported"), slots.get("issue")) if x)
        code = slots.get("inverter_code")
        return ("your problem report" + (f": {core().clip(issue, 120)}" if issue else "")
                + (f" (inverter: {code})" if code and code != NO_CODE else ""))
    return ""


def _meta_reply(ctx):
    """'Did you understand?' / 'What did I say?': show what we understood, then ask the one thing still needed."""
    s = ctx.s
    so_far = _summary_so_far(ctx)
    slot = s.get("awaiting")
    if so_far and slot in RESUME:
        nxt = _ask(ctx, slot)
        return (say(["Yes{first_comma}, I'm following.", "Yes, I understood{first_comma}."], s, **ctx.kw())
                + f" So far I have {so_far}.\n\n" + nxt)
    last = s.get("_last_quote")
    _offer(s, MENU)
    if last:
        return (f"Yes{', ' + ctx.first if ctx.first else ''}, I'm here. Your last request was {last['summary']}, "
                "and it's with our team. What would you like to do next?")
    return say(["Yes, I'm here{first_comma}. Sorry if I missed something. What can I help you with?"], s, **ctx.kw())


def _price_line(ctx, size=None):
    c = core()
    p = None
    if size:
        sized = [p for p in ctx.catalog["packages"] if c.package_size(p)]
        p = min(sized, key=lambda p: abs(c.package_size(p) - size)) if sized else None
    p = p or _estimate(ctx)
    if not p:
        return None
    town = _slots(ctx.s).get("location", "")
    lead = (f"For {size:g}kVA, our closest package is the *{p['name']}* at" if size and c.package_size(p) != size
            else f"The *{p['name']}* is")
    return (f"{lead} {c.price_text(p, town)} installed. ⚡ It powers: {p['powers']}. The final price is confirmed "
            "on a free site visit.")


def _price_in_quote(ctx):
    """A price question during a quotation: answer it, keep any size they gave, and carry on with the request."""
    s, e = ctx.s, ctx.ents
    slots = _slots(s)
    slot = s.get("awaiting")
    size = e.get("size_kva")
    named = any(p["name"].lower() in nlu.clean(ctx.text) for p in ctx.catalog["packages"])
    if size or named or e.get("appliances") or e.get("kwh"):
        if slot in ("repeat", "need", "change", "confirm") or not slots.get("need"):
            for k in ("need", "kwh", "size_kva", "confirmed"):
                slots.pop(k, None)
            _gather(ctx)
            s.pop("awaiting", None)
        line = _price_line(ctx, size)
        if line:
            return line + "\n\n" + _advance(ctx)
    line = _price_line(ctx)
    if not line:
        pkgs = "\n".join(f"• *{p['name']}*: {core().price_text(p, slots.get('location', ''))}"
                         for p in ctx.catalog["packages"])
        line = "Here are our installed prices:\n\n" + pkgs
    return line + ("\n\n" + _ask(ctx, slot, resumed=True) if slot in RESUME else "")


def _not_understood_in_flow(ctx):
    """Unclear message mid-request: say so honestly and ask the same question again (with its buttons)."""
    s = ctx.s
    try:
        db.add_unmatched(ctx.phone, ctx.text, ctx.a["guess"] or "", ctx.a["confidence"])
    except Exception:
        log.exception("could not store unmatched message")
    s["_unknown"] = s.get("_unknown", 0) + 1
    if s["_unknown"] >= 3:   # three misses in a row: a person takes over
        s["_unknown"] = 0
        return _not_understood(ctx)
    sorry = say(["Sorry{first_comma}, I didn't quite get that.", "Sorry, I'm not sure I understood that."],
                s, **ctx.kw())
    so_far = _summary_so_far(ctx)
    return (sorry + (f" So far I have {so_far}." if so_far else "") + "\n\n"
            + _ask(ctx, s["awaiting"]))


def _keep_detail_and_reask(ctx):
    """The customer added detail instead of answering ('no error, it is just dead'): keep it, ask again."""
    s = ctx.s
    slots = _slots(s)
    _slot_ui(ctx, s.get("awaiting"))
    if s.get("awaiting") == "tshoot":
        if len(ctx.text.split()) >= 3:
            _add_detail(slots, "issue", core().clip(ctx.text, 300))
        return (say(["Thanks, noted.", "Okay, thanks.", "Got it."], s)
                + " Did that fix it, or is it still the same? Reply *fixed* or *still the same*.")
    if s.get("flow") == "maintenance":
        _add_detail(slots, "issue", core().clip(ctx.text, 300))
    else:
        _gather(ctx, issue_text=True)
    ack = say(["Got it, thanks.", "Okay, noted.", "Thanks, that helps."], s) + " "
    return ack + say(ASKS[s["awaiting"]], s, **ctx.kw())


def _continue_flow(ctx):
    s = ctx.s
    if s.get("flow") == "details":
        return _finish_details(ctx)
    if s.get("flow") == "sizing":
        kwh = ctx.ents.get("kwh") or ctx.ents.get("appliance_kwh")
        return _sized(ctx, kwh)
    answered = s.get("awaiting")
    _gather(ctx, issue_text=answered not in ("location", "contact", "name", "inverter_code", "tshoot"))
    s.pop("awaiting", None)
    return _advance(ctx, filled=True)


def _advance(ctx, filled=False):
    """Ask for the next missing detail, or finish the request when nothing is missing."""
    s = ctx.s
    slot = _missing(ctx)
    if slot:
        return _ask(ctx, slot, filled=filled)
    if s["flow"] == "purchase":
        return _finish_purchase(ctx)
    if _slots(s).get("tshoot_result") == "fixed":
        return _finish_resolved(ctx)
    return _finish_maintenance(ctx)


REPEAT_WINDOW = 3 * 24 * 3600   # a new quote request within 3 days of the last one: ask which it is


def _h_purchase(ctx):
    c, s = core(), ctx.s
    if s.get("flow") != "purchase":
        reset_flow(s)
        s["flow"] = "purchase"
        c.sig(ctx.phone, "quote_requested")
        last = s.get("_last_quote")
        if last and time.time() - last.get("at", 0) < REPEAT_WINDOW:
            _gather(ctx)
            s["awaiting"] = "repeat"
            _offer(s, REPEAT_BTNS)
            return (f"You already sent us a quotation request ({last['summary']}). Is this for another "
                    "system, or would you like to change that one?")
    _gather(ctx)
    return _advance(ctx)


def _h_maintenance(ctx):
    s = ctx.s
    fresh = s.get("flow") != "maintenance"
    if fresh:
        reset_flow(s)
        s["flow"] = "maintenance"
    _gather(ctx)
    prefix = ""
    risk = bool(_slots(s).get("safety"))
    if fresh and not risk and _missing(ctx) != "issue":
        # the next question is not the symptom question (which already says sorry), so be human first
        if ctx.a["intent"] == "maintenance_service":
            prefix = say(["Happy to arrange that{first_comma}.", "Of course{first_comma}, we can look at it."],
                         s, **ctx.kw()) + " "
        else:
            prefix = say(["Sorry to hear that{first_comma}.", "Sorry about the trouble{first_comma}.",
                          "That sounds frustrating{first_comma}, let's get it fixed."], s, **ctx.kw()) + " "
    if risk and not s.get("_safety_told"):
        s["_safety_told"] = True
        sheet = _intent_replies(ctx)
        prefix = ("⚠️ " + sheet[0] + "\n\n") if sheet else (
            "⚠️ For your safety: please switch off the inverter and the battery breaker, keep away from "
            "the equipment and do not touch it until our technician has checked it. If there is fire or "
            "smoke, leave the area and call the emergency services.\n\n")
        if not ctx.is_web:  # do not wait for the location: staff need to know now
            c = core()
            c.alert_team(ctx.phone, _slots(s).get("name") or ctx.name, "urgent", c.clip(ctx.text, 300), s)
    return prefix + _advance(ctx)


def _finish_purchase(ctx):
    c, s = core(), ctx.s
    slots = _slots(s)
    name = slots.get("name") or ctx.name
    need = slots.get("need") or "Wants a quotation"
    where = slots.get("location", "")
    rec = _estimate(ctx)
    est = _estimate_line(ctx)
    updated = bool(s.pop("_editing_quote", None))
    if rec:
        s["_recommended"] = rec["name"]
        c.sig(ctx.phone, "recommended", rec["name"])
    if slots.get("kwh"):
        c.sig(ctx.phone, "sized_kwh", round(slots["kwh"], 1))
    c.sig(ctx.phone, "need", c.clip(need, 300))
    if where:
        c.sig(ctx.phone, "location", where)
    if slots.get("contact"):
        c.sig(ctx.phone, "contact", slots["contact"])
    c.sig(ctx.phone, "quote_completed")
    c.sig(ctx.phone, "quote_step", "done")
    if name:
        db.set_contact_name(ctx.phone, name)
    c.save_lead(ctx.phone, name, location=where, need=c.clip(("UPDATED: " if updated else "") + need, 1000),
                suggested=s.get("_recommended", ""),
                contact=slots.get("contact"))
    c.set_follow_up(ctx.phone, 2)
    c.alert_team(ctx.phone, name, "quote", ("UPDATED request (replaces their earlier one)\n" if updated else "")
                 + f"Needs: {c.clip(need, 300)}" + (f"\nWhere: {where}" if where else "")
                 + (f"\nBot suggested: {s['_recommended']}" if s.get("_recommended") else ""), s)
    try:
        db.upsert_enquiry(ctx.phone, name, "purchase", "buy_system", ctx.a["priority"] if ctx.a["category"] == "purchase" else "high",
                          ctx.text, summary=f"{need}" + (f" in {where}" if where else ""),
                          reasons=ctx.a["reasons"] + ["Gave all details for a quotation"], confidence=ctx.a["confidence"])
    except Exception:
        log.exception("could not record enquiry")
    first = first_name(name)
    summary = f"{need}" + (f", in {where}" if where else "")
    suggestion = (f"\n\n💲 Your estimate: {est}. A technician confirms the size and final price on a free site "
                  "visit." if est else "")
    s["_last_quote"] = {"at": time.time(), "summary": summary,
                        "slots": {k: slots[k] for k in ("need", "location", "kwh", "size_kva", "name", "contact")
                                  if slots.get(k)}}
    reset_flow(s)
    s["_safety_told"] = False
    if ctx.is_web:
        return (f"✅ Thank you{', ' + first if first else ''}! I've saved your request ({summary}). Our team will "
                f"contact you {ctx.promise} with your quotation.{suggestion}")
    _offer(s, [opt("question", "Ask a question"), opt("fault", "Report a problem")])
    return (f"✅ Done{', ' + first if first else ''}! I've sent your {'updated ' if updated else ''}request to our sales team ({summary})."
            f"{suggestion}\n\nSomeone will message you here {ctx.promise} to book the visit.")


def _finish_maintenance(ctx):
    c, s = core(), ctx.s
    slots = _slots(s)
    name = slots.get("name") or ctx.name
    parts = []
    for x in (slots.get("reported"), slots.get("issue")):
        if x and x not in parts:
            parts.append(x)
    issue_shown = "; ".join(parts) or "Needs maintenance"   # what the customer told us, for their confirmation
    if slots.get("inverter_code"):
        parts.append("Inverter shows: " + slots["inverter_code"])
    failed = slots.get("tshoot_result") in ("failed", "human")
    if slots.get("tried"):
        parts.append("Bot fixes tried, no luck: " + " | ".join(slots["tried"]))
    issue = "; ".join(parts) or "Needs maintenance"
    where = slots.get("location", "")
    routine = bool(slots.get("routine"))
    priority = slots.get("priority") or ("medium" if routine else "high")
    c.sig(ctx.phone, "maintenance_issue", c.clip(issue, 300))
    if where:
        c.sig(ctx.phone, "location", where)
    if slots.get("contact"):
        c.sig(ctx.phone, "contact", slots["contact"])
    if name:
        db.set_contact_name(ctx.phone, name)
    c.save_lead(ctx.phone, name, kind="maintenance", location=where, need=c.clip(issue, 1000),
                contact=slots.get("contact"))
    c.set_follow_up(ctx.phone, 0 if priority in ("high", "urgent") else 1)
    c.alert_team(ctx.phone, name, "maintenance", f"[{priority.upper()}] {c.clip(issue, 300)}"
                 + (f"\nWhere: {where}" if where else ""), s)
    try:
        db.upsert_enquiry(ctx.phone, name, "maintenance", "maintenance_service" if routine else "maintenance_fault",
                          priority, ctx.text,
                          summary=issue + (f" ({where})" if where else ""), reasons=ctx.a["reasons"],
                          confidence=ctx.a["confidence"])
    except Exception:
        log.exception("could not record enquiry")
    first = first_name(name)
    reset_flow(s)
    s["_safety_told"] = False
    where_txt = f" in {where}" if where else ""
    hello = f"Thank you{', ' + first if first else ''}."
    if routine:
        what = issue.replace("Service / inspection requested: ", "")
        return (f"{hello} I've booked in your service request ({c.clip(what, 90)}){where_txt}, and our technical "
                f"team will contact you {ctx.promise} to arrange a visit.")
    how = {"urgent": "as URGENT", "high": "as a priority"}.get(priority, "")
    if failed:
        return (f"{hello} I've passed your report ({c.clip(issue_shown, 90)}){where_txt} to our technical team "
                f"{how}, together with the fixes we tried, so they won't repeat them. A person will contact you "
                f"{ctx.promise} to arrange a visit.".replace("  ", " ")
                + "\n\nUntil then, please don't open the inverter or touch any wiring. If you notice smoke, a "
                  f"burning smell or it gets very hot, switch it off and call us on {ctx.catalog['phone']}.")
    return (f"{hello} I've logged your report ({c.clip(issue_shown, 90)}){where_txt} {how}".replace("  ", " ").rstrip()
            + f", and our technical team will contact you {ctx.promise} to arrange a visit."
            "\n\nUntil then, please don't open the inverter or touch any wiring. If you notice smoke, a burning "
            f"smell or it gets very hot, switch it off and call us on {ctx.catalog['phone']}.")


def _finish_resolved(ctx):
    """The customer's own fix worked: log it as handled (no team alert) and wish them well."""
    c, s = core(), ctx.s
    slots = _slots(s)
    name = slots.get("name") or ctx.name
    parts = [x for x in (slots.get("reported"), slots.get("issue")) if x]
    if slots.get("inverter_code") and slots["inverter_code"] != NO_CODE:
        parts.append("Inverter shows: " + slots["inverter_code"])
    issue = "; ".join(dict.fromkeys(parts)) or "Fault reported"
    note = f"{issue} (solved by bot: {slots.get('fixed_by', 'troubleshooting')})"
    c.sig(ctx.phone, "maintenance_resolved", c.clip(note, 300))
    try:
        db.upsert_enquiry(ctx.phone, name, "maintenance", "maintenance_fault", "low", ctx.text, summary=note,
                          reasons=["Fixed by the customer with the bot's help"], confidence=ctx.a["confidence"])
        db.close_enquiry(ctx.phone, "maintenance")
    except Exception:
        log.exception("could not record resolved enquiry")
    first = first_name(name)
    reset_flow(s)
    s["_safety_told"] = False
    _offer(s, [opt("quote", "Get a quotation"), opt("question", "Ask a question")])
    return (f"Great news{', ' + first if first else ''}! 🎉 I'm glad that sorted it. If it happens again, message "
            f"me here. If you ever notice smoke, a burning smell or it gets very hot, switch it off and call us on "
            f"{ctx.catalog['phone']}. Anything else I can help with?")


# ---------------------------------------------------------------- buttons and lists
# Closed choices get buttons (up to 3, titles up to 20 characters) or a list (up to 10 rows, titles up to 24).
# Open information (what is wrong, the error code, the town, the name) is always typed. Every button is a
# shortcut for a phrase the bot already understands, so a typed answer works just as well.
def opt(key, title, description=""):
    o = {"id": "btn:" + key, "title": title}
    if description:
        o["description"] = description
    return o


def _offer(s, options, button=""):
    s["_ui"] = {"options": [o for o in options if o][:10], "button": button}


def _set_offer(s):
    s["pending"] = "quote_offer"
    _offer(s, OFFER_BTNS)


def _norm_title(t):
    return re.sub(r"[^a-z0-9]+", " ", (t or "").lower()).strip()


MENU = [opt("quote", "Get a quotation"), opt("fault", "Fix a problem"), opt("question", "Ask a question")]
UNSURE_MENU = [opt("quote", "Get a quotation"), opt("fault", "Fix a problem"), opt("human", "Talk to a person")]
OFFER_BTNS = [opt("yes_quote", "Yes, get a quote"), opt("no_thanks", "No, thanks")]
CODE_BTNS = [opt("code_none", "No code or beeps"), opt("code_beep", "It's beeping"), opt("code_blank", "Screen is blank")]
TSHOOT_BTNS = [opt("fixed", "✅ Fixed"), opt("notfixed", "❌ Still the same"), opt("tshoot_human", "👤 Talk to a person")]
REPEAT_BTNS = [opt("new_quote", "Another system"), opt("edit_quote", "Change that one")]
CONFIRM_BTNS = [opt("send", "✅ Send request"), opt("change", "✏️ Change details")]
FAULTS = [  # key, list title, description, what the customer would have typed
    ("charging", "Not charging", "Batteries or panels not charging", "my solar is not charging"),
    ("beep", "Beeping or error code", "Alarm, warning light or code", "my inverter is beeping and showing an error"),
    ("nopower", "No power or dead", "Nothing comes on", "no power, the system is dead"),
    ("trip", "Trips or switches off", "Shuts down under load", "the inverter keeps tripping and switching off"),
    ("drain", "Batteries drain fast", "Power runs out early", "the batteries drain too fast"),
    ("low", "Low solar output", "Panels produce little", "the panels are producing very little power"),
    ("appliance", "Geyser or pump issue", "Won't run on solar", "my geyser or pump is not working on solar"),
    ("other", "Something else", "Describe it in your own words", ""),
]
FAQ_ORDER = ("pay", "warrant", "batter", "night", "install", "site", "geyser", "borehole", "zesa", "last", "brand")
CONFIRMS = re.compile(r"\s*(yes|yeah|yep|ok|okay|sure|please|go ahead|ehe|send|correct|right|that'?s right|perfect|"
                      r"fine|confirm\w*)\b", re.I)
DISAGREES = re.compile(r"\b(no|not right|wrong|incorrect|change|edit|mistake|not correct)\b", re.I)


def _tap_text(tap, c):
    """What a tapped button means in words, so the rest of the bot treats it like a typed answer."""
    words = {"fixed": "fixed", "notfixed": "still the same", "code_none": "no",
             "code_beep": "it is beeping, I didn't see a code", "code_blank": "the display is blank",
             "yes_quote": "yes please", "no_thanks": "no thanks", "send": "yes, send it",
             "change": "I want to change the details", "quote": "I would like a quotation",
             "fault": "I have a problem with my system", "question": "I have a question",
             "human": "I want to talk to a person", "tshoot_human": "I want to talk to a person",
             "new_quote": "another system", "edit_quote": "change the last one"}
    words.update({"fault_" + k: text or "something else" for k, _, _, text in FAULTS})
    for p in c.CATALOG.get("packages", []):
        words[f"pkg_{p['id']}"] = f"I'd like the {p['name']} package"
    return words.get(tap) or tap.split(":")[-1].replace("_", " ")


def _force(ctx, intent):
    meta = _intents.get(intent) or {}
    ctx.a = dict(ctx.a, intent=intent, category=meta.get("category", "enquiry"), understood=True,
                 handler=meta.get("handler", "reply"), confidence=1.0)
    return ctx


def _faq_options():
    """The common questions as list rows (titles are the catalog FAQ titles, at most 24 characters)."""
    rows = [opt("prices", "💲 Prices & packages", "What each system costs and runs")]
    faqs = []
    for f in core().CATALOG.get("faqs", []):
        title = f.get("title") or ""
        slug = _faq_slug(re.sub(r"^[^\w]+", "", title).strip()) if title else ""
        if slug in _intents and 0 < len(title) <= 24:
            rank = next((i for i, w in enumerate(FAQ_ORDER) if w in slug), None)
            if rank is not None:
                faqs.append((rank, title, slug))
    for _, title, slug in sorted(faqs)[:8]:
        rows.append(opt("faq:" + slug, title))
    rows.append(opt("ask_own", "✍️ Type my own question"))
    return rows[:10]


def _direct_tap(ctx):
    """Buttons that start or steer a conversation directly. None: let the normal flow read the tap's words."""
    s, tap, slots = ctx.s, ctx.tap, _slots(ctx.s)
    if tap == "quote":
        reset_flow(s)
        return _h_purchase(_force(ctx, "buy_system"))
    if tap == "fault":
        reset_flow(s)
        s["flow"] = "maintenance"
        return _ask(ctx, "issue")
    if tap == "fault_other":
        if s.get("flow") != "maintenance":
            reset_flow(s)
            s["flow"] = "maintenance"
        s["awaiting"] = "issue"
        return say(["Okay{first_comma}, please tell me in your own words what the system is doing, and roughly "
                    "when it started."], s, **ctx.kw())
    if tap == "question":
        _offer(s, _faq_options(), "Choose a topic")
        return say(["Sure{first_comma}! Pick a topic below, or just type your question.",
                    "Of course{first_comma}. Choose a topic, or type your question in your own words."],
                   s, **ctx.kw())
    if tap == "ask_own":
        return say(["Go ahead{first_comma}, type your question and I'll answer it."], s, **ctx.kw())
    if tap == "prices":
        return _with_resume(ctx, _h_packages(_force(ctx, "price")))
    if tap.startswith("faq:"):
        slug = tap[4:]
        if slug not in _intents:
            return None
        text = _h_reply(_force(ctx, slug))
        if s.get("pending") != "quote_offer":
            _offer(s, [opt("quote", "Get a quotation"), opt("question", "Another question")])
        return _with_resume(ctx, text)
    if tap.startswith("pkg_"):
        p = next((p for p in ctx.catalog["packages"] if f"pkg_{p['id']}" == tap), None)
        if p is None:
            return None
        fresh = s.get("flow") != "purchase"
        if fresh:
            reset_flow(s)
            s["flow"] = "purchase"
            core().sig(ctx.phone, "quote_requested")
        core().sig(ctx.phone, "package", p["name"])
        slots = _slots(s)
        _add_detail(slots, "need", p["name"])
        if core().package_size(p):
            slots["size_kva"] = core().package_size(p)
        s.pop("awaiting", None)
        return _advance(ctx, filled=not fresh)
    if tap in ("human", "tshoot_human"):
        if tap == "tshoot_human" and s.get("flow") == "maintenance" and s.get("awaiting") == "tshoot":
            st = s.get("tshoot") or {}
            guide = troubleshoot.get(st.get("guide", "generic"))
            slots.setdefault("tried", []).append(f"{guide['title']}: asked for a person at step {st.get('step', 0) + 1}")
            slots["tshoot_done"], slots["tshoot_result"] = True, "human"
            s.pop("awaiting", None)
            return _advance(ctx)
        return _h_human(_force(ctx, "talk_to_human"))
    return None


def _with_resume(ctx, text):
    """A side answer in the middle of a request: answer, then go back to the question we were asking."""
    s = ctx.s
    if s.get("flow") in ("purchase", "maintenance") and s.get("awaiting"):
        return text + "\n\n" + _ask(ctx, s["awaiting"], resumed=True)
    return text


def _slot_ui(ctx, slot):
    """The buttons or list that go with a question (none for questions that need a typed answer)."""
    s = ctx.s
    if slot == "need":
        rows = [opt(f"pkg_{p['id']}", core().short(p["name"], 24),
                    core().short(f"{core().price_text(p, s.get('_town', ''))} · {p['powers']}", 72))
                for p in ctx.catalog["packages"]][:10]
        _offer(s, rows, "See packages")
    elif slot == "issue":
        _offer(s, [opt("fault_" + k, t, d) for k, t, d, _ in FAULTS], "Choose a problem")
    elif slot == "inverter_code":
        _offer(s, CODE_BTNS)
    elif slot == "tshoot":
        _offer(s, TSHOOT_BTNS)
    elif slot == "confirm":
        _offer(s, CONFIRM_BTNS)
    elif slot == "repeat":
        _offer(s, REPEAT_BTNS)
    else:
        s.pop("_ui", None)


def _estimate(ctx):
    """The package that fits what the customer asked for: one they named, the closest size, or sized from kWh."""
    c, slots = core(), _slots(ctx.s)
    pkgs = ctx.catalog["packages"]
    need = (slots.get("need") or "").lower()
    named = [p for p in pkgs if p["name"].lower() in need]
    if named:
        return named[-1]
    if slots.get("size_kva"):
        sized = [p for p in pkgs if c.package_size(p)]
        if sized:
            return min(sized, key=lambda p: abs(c.package_size(p) - slots["size_kva"]))
    if slots.get("kwh"):
        return c.recommend(slots["kwh"])
    return None


def _estimate_line(ctx):
    p = _estimate(ctx)
    if not p:
        return ""
    return f"*{p['name']}*, {core().price_text(p, _slots(ctx.s).get('location', ''))} installed"


def _confirm_text(ctx):
    c, s = core(), ctx.s
    slots = _slots(s)
    lines = [f"• *System:* {slots.get('need') or 'a quotation'}", f"• *Where:* {slots.get('location') or 'not given'}"]
    est = _estimate_line(ctx)
    if est:
        lines.append(f"• *Estimate:* {est} (final price after a free site visit)")
    if slots.get("name"):
        lines.append(f"• *Name:* {slots['name']}")
    if slots.get("contact"):
        lines.append(f"• *Phone:* {slots['contact']}")
    return ("Here's your quotation request:\n\n" + "\n".join(lines)
            + "\n\nShall I send it to our team, or would you like to change something?")


def _apply_change(ctx, force=False):
    """Update the quotation request from a correction. True when something changed."""
    s, e, raw = ctx.s, ctx.ents, ctx.text
    slots = _slots(s)
    names = [p["name"].lower() for p in ctx.catalog["packages"]]
    if any(e.get(k) for k in ("appliances", "kwh", "size_kva", "budget", "products")) \
            or any(n in nlu.clean(raw) for n in names):
        for k in ("need", "kwh", "size_kva"):
            slots.pop(k, None)
        _gather(ctx)
        return True
    if e.get("location"):
        slots["location"] = e["location"]
        s["_town"] = e["location"]
        return True
    if e.get("phone") or e.get("name"):
        _gather(ctx)
        return True
    if not force:
        return False
    if _looks_like_place(raw):
        slots["location"] = s["_town"] = raw.strip(" .,!").title()
        return True
    if len(raw.split()) >= 2:
        slots["need"] = core().clip(raw, 200)
        slots.pop("kwh", None)
        slots.pop("size_kva", None)
        return True
    return False


_HANDLERS = {
    "reply": _h_reply, "greeting": _h_greeting, "thanks": _h_thanks, "goodbye": _h_goodbye, "affirm": _h_affirm,
    "deny": _h_deny, "contact": _h_contact, "packages": _h_packages, "sizing": _h_sizing, "human": _h_human,
    "purchase": _h_purchase, "maintenance": _h_maintenance,
    "answer_team": _h_answer_team, "answer_alert": _h_answer_alert,
}
