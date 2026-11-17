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
    added = db.add_examples_bulk(rows, "seed")
    db.set_setting("nlu_seed_version", str(version))
    log.info("NLU seed: %d new example phrases", added)
    return True


QA_SHEET = Path(__file__).parent / "training" / "solar_customer_qa.csv"


def _import_sheet():
    """Load training/solar_customer_qa.csv (question/answer sheet) once per version of the file."""
    if not QA_SHEET.exists():
        return
    import qa_import
    digest = qa_import.file_hash(QA_SHEET)
    if db.get_setting("nlu_qa_hash") == digest:
        return
    qa_import.import_csv(QA_SHEET)
    db.set_setting("nlu_qa_hash", digest)


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
    if os.getenv("NLU_BACKEND", "tfidf").lower() == "transformer":
        _upgrade_to_transformer(examples)
        info = dict(info, note="The transformer model is loading in the background. The quick model answers "
                               "until it is ready; check this page again in a few minutes.")
    return info


_upgrade_lock = threading.Lock()


def _upgrade_to_transformer(examples):
    if not _upgrade_lock.acquire(blocking=False):
        return  # one is already running; it will pick up the phrases at its next run

    def run():
        try:
            nlu.engine.train(examples, backend="transformer")
        except Exception:
            log.exception("transformer training failed; staying on the quick model")
        finally:
            _upgrade_lock.release()

    threading.Thread(target=run, daemon=True, name="nlu-transformer").start()


def start(catalog):
    """Called once at start-up."""
    try:
        seed(catalog)
        _import_sheet()
        return retrain()
    except Exception:
        log.exception("NLU start-up failed; the bot will hand every message to the team")
        return {"trained": False}


# ---------------------------------------------------------------- understanding
def analyze(text, towns=None, existing_customer=False):
    """Everything the model knows about one message (also used by the 'test a phrase' tool)."""
    res = nlu.engine.predict(text)
    ents = nlu.extract_entities(text, towns)
    t = nlu.clean(text)
    intent = res["intent"] if nlu.engine.understood(res) else None
    # safety words mean a fault report, unless the person is clearly talking about buying something
    category_now = (_intents.get(intent) or {}).get("category")
    if nlu.SAFETY.search(t) and category_now not in ("maintenance", "purchase") and "maintenance_fault" in _intents:
        intent = "maintenance_fault"
    # figures or appliances with no clear intent: treat it as a sizing / buying question
    if intent is None and (ents.get("appliances") or ents.get("kwh")) and "sizing" in _intents:
        intent = "sizing"
    meta = _intents.get(intent) or {}
    category = meta.get("category", "enquiry")
    priority, reasons = nlu.decide_priority(meta.get("priority", "medium"), category, text, ents, existing_customer)
    return {"intent": intent, "guess": res["intent"], "confidence": round(res["confidence"], 3),
            "similarity": round(res["similarity"], 3), "ranked": res["ranked"], "matched": res["matched"],
            "category": category, "priority": priority, "reasons": reasons, "entities": ents,
            "handler": meta.get("handler", "reply"), "understood": intent is not None}


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
    c = core()
    towns = [t for t in (c.CATALOG.get("towns") or []) if isinstance(t, str)]
    a = analyze(text, towns)
    ctx = Ctx(phone, name, text, s, a)
    s.pop("offer", None)

    # 1. is the customer answering the question we just asked?
    filled = _fill_awaiting(ctx)
    if filled:
        return _continue_flow(ctx)

    intent, handler = a["intent"], a["handler"]
    if intent is None:
        return _not_understood(ctx)
    s["_unknown"] = 0
    if a["confidence"] < 0.4:  # answered, but unsure: let staff confirm it so the bot learns
        try:
            db.add_unmatched(phone, text, intent, a["confidence"])
        except Exception:
            log.exception("could not queue message for review")

    # 2. record what they want (purchase / enquiry / maintenance) with its priority
    _record_enquiry(ctx)

    # 3. answer
    reply = _HANDLERS.get(handler, _h_reply)(ctx)

    # 4. answered a side question in the middle of a flow: gently go back to it
    if s.get("flow") and s.get("awaiting") and handler not in ("purchase", "maintenance"):
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
    if e.get("size_kva") or e.get("budget") or (e.get("location") and e.get("appliances")):
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
    reset_flow(s)
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


def _gather(ctx):
    """Pull everything useful out of the message into the flow's slots."""
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
        if (ctx.a["intent"] in ("maintenance_fault", "warranty_claim") or ctx.a["priority"] == "urgent"
                or len(t.split()) >= 4):
            _add_detail(slots, "issue", core().clip(t, 300))
        elif ctx.a["intent"] == "maintenance_service":
            _add_detail(slots, "issue", "Routine service / inspection")


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
    "issue": ["I'm sorry you're having trouble{first_comma}. What exactly is happening? For example any beeping, "
              "error code or lights on the inverter, and since when?",
              "Sorry to hear that{first_comma}. Can you describe the problem? Any error code, beeping or warning "
              "light, and when did it start?"],
    "location": ["Thanks! Whereabouts is it (town or suburb)?", "Got it. Which town or suburb are you in?",
                 "And where is the installation, which town or suburb?"],
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


def _ask(ctx, slot, resumed=False):
    ctx.s["awaiting"] = slot
    if resumed:
        return RESUME[slot]
    return say(ASKS[slot], ctx.s, **ctx.kw())


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
    if a["intent"] in ("greeting", "thanks", "goodbye", "deny", "talk_to_human", "complaint", "price", "contact_info",
                       "bot_identity", "off_topic") or (a["intent"] or "").startswith("faq_"):
        return False
    if slot == "details":
        # anything that is not a question is taken as the details we asked for
        confident_other = a["understood"] and a["confidence"] >= 0.6 and a["intent"] != "affirm"
        return "?" not in raw and len(re.findall(r"[A-Za-z0-9]", raw)) >= 2 and not confident_other
    if slot == "sizing_need":
        kwh = e.get("kwh") or e.get("appliance_kwh")
        return bool(kwh)
    sure_of_other = a["understood"] and a["confidence"] >= 0.6 and a["intent"] not in (None, "affirm")
    if slot == "location":
        # a short, non-question reply to "where?" is the answer ("Borrowdale, Harare"), unless the model is
        # confident it is something else entirely
        if _looks_like_answer(raw) and (e.get("location") or not sure_of_other):
            place = raw.strip(" .,!").title()
        else:
            place = e.get("location", "")
        if not place:
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
        if any(e.get(k) for k in ("appliances", "kwh", "size_kva", "budget", "products")):
            _gather(ctx)
            return True
        if len(raw.split()) >= 2 and "?" not in raw and a["intent"] in (None, "buy_system", "sizing"):
            _add_detail(slots, "need", core().clip(raw, 200))
            return True
        return False
    if slot == "issue":
        if len(raw.split()) >= 2 and "?" not in raw and a["intent"] not in ("affirm",):
            _add_detail(slots, "issue", core().clip(raw, 300))
            return True
        return False
    return False


def _continue_flow(ctx):
    s = ctx.s
    if s.get("flow") == "details":
        return _finish_details(ctx)
    if s.get("flow") == "sizing":
        kwh = ctx.ents.get("kwh") or ctx.ents.get("appliance_kwh")
        return _sized(ctx, kwh)
    _gather(ctx)
    s.pop("awaiting", None)
    return _advance(ctx)


def _advance(ctx):
    """Ask for the next missing detail, or finish the request when nothing is missing."""
    s = ctx.s
    slot = _missing(ctx)
    if slot:
        return _ask(ctx, slot)
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
    if s.get("flow") != "maintenance":
        reset_flow(s)
        s["flow"] = "maintenance"
    _gather(ctx)
    prefix = ""
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
    issue = slots.get("issue") or "Needs maintenance"
    where = slots.get("location", "")
    priority = ctx.a["priority"] if ctx.a["category"] == "maintenance" else "high"
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
        db.upsert_enquiry(ctx.phone, name, "maintenance", ctx.a["intent"] or "maintenance_fault", priority, ctx.text,
                          summary=issue + (f" ({where})" if where else ""), reasons=ctx.a["reasons"],
                          confidence=ctx.a["confidence"])
    except Exception:
        log.exception("could not record enquiry")
    first = first_name(name)
    reset_flow(s)
    s["_safety_told"] = False
    if priority == "urgent":
        lead = "I've marked this as URGENT"
    elif priority == "high":
        lead = "I've marked this as high priority"
    else:
        lead = "I've logged this"
    where_txt = f" in {where}" if where else ""
    return (f"Thank you{', ' + first if first else ''}. {lead} and our technical team will contact you "
            f"{ctx.promise} to arrange a visit{where_txt}. If it gets worse, call us on {ctx.catalog['phone']}.")


_HANDLERS = {
    "reply": _h_reply, "greeting": _h_greeting, "thanks": _h_thanks, "goodbye": _h_goodbye, "affirm": _h_affirm,
    "deny": _h_deny, "contact": _h_contact, "packages": _h_packages, "sizing": _h_sizing, "human": _h_human,
    "purchase": _h_purchase, "maintenance": _h_maintenance,
    "answer_team": _h_answer_team, "answer_alert": _h_answer_alert,
}
