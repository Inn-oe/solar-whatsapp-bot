"""Conversation brain: reads what the customer typed, understands it (nlu.py) and answers in plain text.

There are no menus or buttons. The customer writes however they like ("hi, my inverter keeps beeping"),
the language model works out what they want, and the bot talks it through like a person would:
asking one thing at a time, remembering what was already said, answering side questions without losing
its place, and handing over to the team when it is not sure.

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
from pathlib import Path

import db
import llm
import nlu

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


def is_question(text):
    return "?" in text or bool(QUESTION_START.match(text))


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
        self.ents = a["entities"]
        self.first = first_name(s.get("slots", {}).get("name") or name)
        c = core()
        self.catalog = c.CATALOG
        self.promise = c.reply_promise()

    def kw(self, **extra):
        return {"first": self.first, "company": self.catalog["company"], "phone": self.catalog["phone"],
                "hours": self.catalog["hours"], "promise": self.promise, **extra}


def respond(phone, name, text, s):
    """The reply text for one message. `s` is the saved conversation state (a dict that is mutated)."""
    reply = _respond(phone, name, text, s)
    s["_last_reply"] = (reply.get("text") if isinstance(reply, dict) else str(reply or ""))[:400]
    return reply


def _respond(phone, name, text, s):
    c = core()
    towns = [t for t in (c.CATALOG.get("towns") or []) if isinstance(t, str)]
    a = analyze(text, towns, context=s.get("_last_reply", ""))
    ctx = Ctx(phone, name, text, s, a)
    s.pop("offer", None)

    # 1. is the customer answering the question we just asked?
    filled = _fill_awaiting(ctx)
    if filled == "more":
        return _keep_detail_and_reask(ctx)
    if filled:
        return _continue_flow(ctx)

    intent, handler = a["intent"], a["handler"]
    # "yes please" / "send the quote" right after we offered a quotation accepts the offer
    accepts_offer = (s.get("pending") == "quote_offer" and intent in (None, "affirm", "buy_system", "site_visit")
                     and not is_question(text) and bool(AGREES.match(text)))
    if intent is None and not accepts_offer:
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
            s["pending"] = "quote_offer"
            text += "\n\n" + say(["If you'd like, I can arrange a free quotation for you.",
                                  "Would you like me to prepare a free quotation?"], s)
    return text


def _h_greeting(ctx):
    s = ctx.s
    reset_flow(s)
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
            s["pending"] = "quote_offer"
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
        s["pending"] = "quote_offer"
        note = c.market_note(p, town)
        c.sig(ctx.phone, "package", p["name"])
        return (f"For a {size:g}kVA system, our closest is the *{p['name']}* at {c.price_text(p, town)}.\n\n"
                f"⚡ Powers: {p['powers']}\n📦 Includes: {p['includes']}"
                + (f"\n\n{note}" if note else "")
                + "\n\nWould you like me to arrange a free quotation and site visit?")
    lines = "\n".join(f"• *{p['name']}*: {c.price_text(p, town)}" for p in pkgs)
    s["pending"] = "quote_offer"
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
    s["pending"] = "quote_offer"
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
    r"smok\w*|burn\w*|spark\w*|low|weak|slow|dim|only|fell|dropp\w*|keeps?|since|every)\b")


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


def _missing(ctx):
    s = ctx.s
    slots = _slots(s)
    order = ["need", "location"] if s["flow"] == "purchase" else ["issue", "location"]
    for slot in order:
        if not slots.get(slot):
            return slot
    if ctx.is_web and not slots.get("contact"):
        return "contact"
    if not slots.get("name") and not first_name(ctx.name):
        return "name"
    return None


ASKS = {
    "need": ["Happy to help with that{first_comma}! What would you like the system to run? For example fridge, "
             "TV, lights, a borehole pump, or the size if you know it (like 5kVA).",
             "Sure{first_comma}. So I can size it properly, what do you want to power: lights, fridge, TV, geyser, pump?"],
    "issue": ["Sorry to hear that{first_comma}, let's get it sorted. What is the system doing? For example "
              "beeping, an error code on the display, no power, or batteries not holding. And roughly when did "
              "it start?",
              "Oh no, that's frustrating{first_comma}. What are you seeing exactly? Any beeping, error code or "
              "warning light, lights off, or batteries draining fast? And when did it begin?"],
    "location": ["Which town or suburb is it in?", "Whereabouts are you, which town or suburb?",
                 "And which town or suburb are you in?"],
    "contact": ["What's the best phone number (WhatsApp if possible) for our team to reach you on?"],
    "name": ["And may I have your name please?", "Lastly, what name should I put on the request?"],
    "sizing_need": ["What would you like to keep running: fridge, TV, lights, geyser, pump? Or tell me your "
                    "monthly kWh if you know it."],
}
RESUME = {"details": "When you can, send me those details and I'll pass them to our team.",
          "need": "Back to your request: what would you like the system to run?",
          "issue": "Back to your problem: what exactly is happening with the system?",
          "location": "Back to your request: which town or suburb are you in?",
          "contact": "Back to your request: what phone number can we reach you on?",
          "name": "Back to your request: what name should I use?",
          "sizing_need": "Back to sizing: what would you like to keep running?"}


def _ask(ctx, slot, resumed=False, filled=False):
    """The next question. `filled` means the customer just answered the previous one, so acknowledge it first
    (never say 'thanks' before they have given us anything)."""
    ctx.s["awaiting"] = slot
    if resumed:
        return RESUME[slot]
    ack = ""
    if filled:
        ack = say(["Thanks.", "Got it.", "Okay, thanks.", "Noted."], ctx.s) + " "
    return ack + say(ASKS[slot], ctx.s, **ctx.kw())


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
            if ("?" not in raw and len(raw.split()) >= 2 and s.get("flow") in ("purchase", "maintenance")
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


def _keep_detail_and_reask(ctx):
    """The customer added detail instead of answering ('no error, it is just dead'): keep it, ask again."""
    s = ctx.s
    slots = _slots(s)
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
    _gather(ctx, issue_text=answered not in ("location", "contact", "name"))
    s.pop("awaiting", None)
    return _advance(ctx, filled=True)


def _advance(ctx, filled=False):
    """Ask for the next missing detail, or finish the request when nothing is missing."""
    s = ctx.s
    slot = _missing(ctx)
    if slot:
        return _ask(ctx, slot, filled=filled)
    return _finish_purchase(ctx) if s["flow"] == "purchase" else _finish_maintenance(ctx)


def _h_purchase(ctx):
    c, s = core(), ctx.s
    if s.get("flow") != "purchase":
        reset_flow(s)
        s["flow"] = "purchase"
        c.sig(ctx.phone, "quote_requested")
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
    if fresh and ctx.a["priority"] != "urgent" and _missing(ctx) != "issue":
        # the next question is not the symptom question (which already says sorry), so be human first
        if ctx.a["intent"] == "maintenance_service":
            prefix = say(["Happy to arrange that{first_comma}.", "Of course{first_comma}, we can look at it."],
                         s, **ctx.kw()) + " "
        else:
            prefix = say(["Sorry to hear that{first_comma}.", "Sorry about the trouble{first_comma}.",
                          "That sounds frustrating{first_comma}, let's get it fixed."], s, **ctx.kw()) + " "
    if ctx.a["priority"] == "urgent" and not s.get("_safety_told"):
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
    rec = None
    if slots.get("kwh"):
        rec = c.recommend(slots["kwh"])
        s["_recommended"] = rec["name"]
        c.sig(ctx.phone, "recommended", rec["name"])
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
    c.save_lead(ctx.phone, name, location=where, need=c.clip(need, 1000), suggested=s.get("_recommended", ""),
                contact=slots.get("contact"))
    c.set_follow_up(ctx.phone, 2)
    c.alert_team(ctx.phone, name, "quote", f"Needs: {c.clip(need, 300)}" + (f"\nWhere: {where}" if where else "")
                 + (f"\nBot suggested: {s['_recommended']}" if s.get("_recommended") else ""), s)
    try:
        db.upsert_enquiry(ctx.phone, name, "purchase", "buy_system", ctx.a["priority"] if ctx.a["category"] == "purchase" else "high",
                          ctx.text, summary=f"{need}" + (f" in {where}" if where else ""),
                          reasons=ctx.a["reasons"] + ["Gave all details for a quotation"], confidence=ctx.a["confidence"])
    except Exception:
        log.exception("could not record enquiry")
    first = first_name(name)
    summary = f"{need}" + (f", in {where}" if where else "")
    suggestion = (f"\n\nFrom what you've told me I'd suggest the *{rec['name']}*, but a technician will confirm the "
                  "size on a free site visit." if rec else "")
    reset_flow(s)
    s["_safety_told"] = False
    if ctx.is_web:
        return (f"✅ Thank you{', ' + first if first else ''}! I've saved your request ({summary}). Our team will "
                f"contact you {ctx.promise} with your quotation.{suggestion}")
    return (f"✅ Thank you{', ' + first if first else ''}! I've sent your request to our sales team ({summary}). "
            f"Someone will message you here {ctx.promise} with your quotation and arrange a free site visit."
            f"{suggestion}\n\nIf you have any other questions in the meantime, just ask.")


def _finish_maintenance(ctx):
    c, s = core(), ctx.s
    slots = _slots(s)
    name = slots.get("name") or ctx.name
    parts = []
    for x in (slots.get("reported"), slots.get("issue")):
        if x and x not in parts:
            parts.append(x)
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
    return (f"{hello} I've logged your report ({c.clip(issue, 90)}){where_txt} {how}".replace("  ", " ").rstrip()
            + f", and our technical team will contact you {ctx.promise} to arrange a visit."
            "\n\nUntil then, please don't open the inverter or touch any wiring. If you notice smoke, a burning "
            f"smell or it gets very hot, switch it off and call us on {ctx.catalog['phone']}.")


_HANDLERS = {
    "reply": _h_reply, "greeting": _h_greeting, "thanks": _h_thanks, "goodbye": _h_goodbye, "affirm": _h_affirm,
    "deny": _h_deny, "contact": _h_contact, "packages": _h_packages, "sizing": _h_sizing, "human": _h_human,
    "purchase": _h_purchase, "maintenance": _h_maintenance,
    "answer_team": _h_answer_team, "answer_alert": _h_answer_alert,
}
