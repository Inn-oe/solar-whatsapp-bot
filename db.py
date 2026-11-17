"""Storage for leads, chat logs and the editable catalog.

SQLite file by default; set DATABASE_URL (PostgreSQL) to keep data across restarts on Render.
"""
import datetime as dt
import json
import os

from sqlalchemy import (Boolean, Column, DateTime, Float, Integer, String, Text, UniqueConstraint, create_engine, desc,
                        func, select)
from sqlalchemy.orm import declarative_base, sessionmaker

_url = os.getenv("DATABASE_URL", "sqlite:///data.db")
if _url.startswith("postgres://"):
    _url = "postgresql://" + _url[len("postgres://"):]

engine = create_engine(_url, pool_pre_ping=True)
Session = sessionmaker(bind=engine, expire_on_commit=False)
Base = declarative_base()

STATUSES = ("new", "contacted", "won", "lost")


def _now():
    return dt.datetime.now(dt.timezone.utc)


class Lead(Base):
    __tablename__ = "leads"
    id = Column(Integer, primary_key=True)
    created = Column(DateTime(timezone=True), default=_now, index=True)
    source = Column(String(20))              # whatsapp | web
    kind = Column(String(20), default="quote")  # quote | callback
    name = Column(String(200), default="")
    phone = Column(String(80), default="")
    location = Column(String(200), default="")
    need = Column(Text, default="")
    suggested = Column(String(200), default="")
    status = Column(String(20), default="new")
    notes = Column(Text, default="")


class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True)
    created = Column(DateTime(timezone=True), default=_now, index=True)
    who = Column(String(80), index=True)     # phone number, or "web:<session>"
    direction = Column(String(3))            # in | out
    sender = Column(String(10), default='')  # customer | bot | staff
    text = Column(Text)


class Contact(Base):
    """One row per person who has chatted: what they did, where they are in the sale, and notes."""
    __tablename__ = "contacts"
    who = Column(String(80), primary_key=True)
    name = Column(String(200), default="")
    first_seen = Column(DateTime(timezone=True), default=_now)
    last_inbound = Column(DateTime(timezone=True))
    last_outbound = Column(DateTime(timezone=True))
    inbound_count = Column(Integer, default=0)
    signals = Column(Text, default="{}")      # JSON of things the person did in the chat
    stage = Column(String(20), default="new")
    notes = Column(Text, default="")
    follow_up = Column(String(10), default="")  # YYYY-MM-DD
    assigned = Column(String(100), default="")
    bot_paused = Column(Boolean, default=False)


class Setting(Base):
    __tablename__ = "settings"
    key = Column(String(50), primary_key=True)
    value = Column(Text)


class ChatState(Base):
    """Where each person is in a conversation (e.g. halfway through a quote), so restarts don't lose it."""
    __tablename__ = "chat_state"
    who = Column(String(80), primary_key=True)
    data = Column(Text, default="{}")
    updated = Column(DateTime(timezone=True), default=_now)


class MarketPrice(Base):
    """One advertised price for a solar system size, read from a public web page."""
    __tablename__ = "market_prices"
    id = Column(Integer, primary_key=True)
    seen = Column(DateTime(timezone=True), default=_now, index=True)
    source = Column(String(120), index=True)   # name of the seller / site
    url = Column(String(500))
    location = Column(String(80), default="")  # town the source covers, "" = national
    size_kva = Column(String(8))               # kept as text ("3.2") so SQLite and Postgres agree
    price = Column(Integer)                    # USD
    kind = Column(String(12), default="")      # installed | kit | "" (not stated)
    snippet = Column(String(300), default="")  # the text the price was read from


class NluIntent(Base):
    """A kind of message the bot understands, with its category, priority and the replies it may give."""
    __tablename__ = "nlu_intents"
    name = Column(String(60), primary_key=True)
    category = Column(String(20), default="enquiry")   # purchase | enquiry | maintenance | other
    priority = Column(String(10), default="medium")    # low | medium | high | urgent
    handler = Column(String(20), default="reply")      # reply | purchase | maintenance | packages | sizing | ...
    replies = Column(Text, default="[]")               # JSON list of reply variants
    enabled = Column(Boolean, default=True)
    updated = Column(DateTime(timezone=True), default=_now)


class NluExample(Base):
    """One example phrase for an intent. These are what the language model is trained on."""
    __tablename__ = "nlu_examples"
    __table_args__ = (UniqueConstraint("intent", "text", name="uq_nlu_example"),)
    id = Column(Integer, primary_key=True)
    intent = Column(String(60), index=True)
    text = Column(String(500))
    source = Column(String(12), default="seed")        # seed | staff | chat
    created = Column(DateTime(timezone=True), default=_now)


class Enquiry(Base):
    """What a customer wants, one open row per person and category (purchase / enquiry / maintenance)."""
    __tablename__ = "enquiries"
    id = Column(Integer, primary_key=True)
    who = Column(String(80), index=True)
    name = Column(String(200), default="")
    category = Column(String(20), index=True)
    intent = Column(String(60), default="")
    priority = Column(String(10), default="medium")
    reasons = Column(Text, default="[]")               # JSON list: why this priority
    status = Column(String(12), default="open", index=True)  # open | handled
    summary = Column(Text, default="")
    last_text = Column(Text, default="")
    message_count = Column(Integer, default=1)
    confidence = Column(Float, default=0.0)
    created = Column(DateTime(timezone=True), default=_now)
    updated = Column(DateTime(timezone=True), default=_now, index=True)
    handled_by = Column(String(100), default="")


class Unmatched(Base):
    """Messages the model was unsure about: staff label them and they become training examples."""
    __tablename__ = "nlu_unmatched"
    id = Column(Integer, primary_key=True)
    who = Column(String(80), default="")
    text = Column(String(500))
    guess = Column(String(60), default="")
    confidence = Column(Float, default=0.0)
    status = Column(String(12), default="new", index=True)  # new | labelled | ignored
    label = Column(String(60), default="")
    created = Column(DateTime(timezone=True), default=_now)


Base.metadata.create_all(engine)


def replace_market_prices(source, rows):
    """Swap one source's prices for what was just read (so removed adverts disappear)."""
    with Session() as s:
        s.query(MarketPrice).filter(MarketPrice.source == source).delete()
        for r in rows:
            s.add(MarketPrice(source=source[:120], url=r["url"][:500], location=r.get("location", "")[:80],
                              size_kva=str(r["size_kva"]), price=int(r["price"]), kind=r.get("kind", "")[:12],
                              snippet=r.get("snippet", "")[:300]))
        s.commit()


def market_prices(max_age_days=45):
    """Prices read recently; older ones are ignored as out of date."""
    cutoff = _now() - dt.timedelta(days=max_age_days)
    with Session() as s:
        rows = s.scalars(select(MarketPrice).where(MarketPrice.seen >= cutoff)).all()
        return [{"source": r.source, "url": r.url, "location": r.location or "", "size_kva": float(r.size_kva),
                 "price": r.price, "kind": r.kind or "", "snippet": r.snippet or "", "seen": _iso(r.seen)}
                for r in rows]


def drop_market_source(source):
    with Session() as s:
        s.query(MarketPrice).filter(MarketPrice.source == source).delete()
        s.commit()


STATE_MAX_AGE = dt.timedelta(hours=24)  # an unfinished conversation older than this starts fresh


def get_state(who):
    with Session() as s:
        row = s.get(ChatState, who[:80])
        if row is None:
            return {}
        updated = row.updated if (row.updated is None or row.updated.tzinfo) else row.updated.replace(tzinfo=dt.timezone.utc)
        try:
            data = json.loads(row.data or "{}")
        except ValueError:
            return {}
        if updated and _now() - updated > STATE_MAX_AGE:
            # keep only long-lived memory (alert timestamps), drop the half-finished flow
            return {k: v for k, v in data.items() if k.startswith("_")}
        return data


def save_state(who, data):
    with Session() as s:
        row = s.get(ChatState, who[:80])
        if row is None:
            row = ChatState(who=who[:80])
            s.add(row)
        row.data = json.dumps(data)
        row.updated = _now()
        s.commit()


def _iso(d):
    """ISO time with a UTC offset (SQLite returns naive values, which browsers would read as local time)."""
    if not d:
        return None
    return (d if d.tzinfo else d.replace(tzinfo=dt.timezone.utc)).isoformat()


def _lead(r):
    return {"id": r.id, "created": _iso(r.created), "source": r.source, "kind": r.kind,
            "name": r.name, "phone": r.phone, "location": r.location, "need": r.need,
            "suggested": r.suggested, "status": r.status, "notes": r.notes}


def add_lead(source, phone, name="", location="", need="", suggested="", kind="quote"):
    with Session() as s:
        row = Lead(source=source, kind=kind, phone=phone[:80], name=name[:200],
                   location=location[:200], need=need[:2000], suggested=suggested[:200])
        s.add(row)
        s.commit()
        return _lead(row)


def list_leads(status=None, limit=500):
    with Session() as s:
        q = select(Lead).order_by(desc(Lead.created)).limit(limit)
        if status in STATUSES:
            q = q.where(Lead.status == status)
        return [_lead(r) for r in s.scalars(q)]


def update_lead(lead_id, status=None, notes=None):
    with Session() as s:
        row = s.get(Lead, lead_id)
        if not row:
            return None
        if status in STATUSES:
            row.status = status
        if notes is not None:
            row.notes = notes[:2000]
        s.commit()
        return _lead(row)


def add_message(who, direction, text, sender=None):
    sender = sender or ("customer" if direction == "in" else "bot")
    with Session() as s:
        s.add(Message(who=who[:80], direction=direction, sender=sender, text=(text or "")[:2000]))
        s.commit()


# ---------------------------------------------------------------- contacts (potential clients)
CONTACT_FIELDS = ("stage", "notes", "follow_up", "assigned", "bot_paused")


def contact_dict(c):
    try:
        signals = json.loads(c.signals or "{}")
    except ValueError:
        signals = {}
    return {"who": c.who, "name": c.name or "", "first_seen": c.first_seen, "last_inbound": c.last_inbound,
            "last_outbound": c.last_outbound, "inbound_count": c.inbound_count or 0, "signals": signals,
            "stage": c.stage or "new", "notes": c.notes or "", "follow_up": c.follow_up or "",
            "assigned": c.assigned or "", "bot_paused": bool(c.bot_paused)}


def touch_contact(who, name="", direction="in"):
    """Create the contact on first message and update counters/timestamps."""
    with Session() as s:
        c = s.get(Contact, who[:80])
        if c is None:
            c = Contact(who=who[:80], name=(name or "")[:200], signals="{}", stage="new", inbound_count=0)
            s.add(c)
        if name and not c.name:
            c.name = name[:200]
        if direction == "in":
            c.last_inbound = _now()
            c.inbound_count = (c.inbound_count or 0) + 1
        else:
            c.last_outbound = _now()
        s.commit()
        return contact_dict(c)


def add_signal(who, key, value=True):
    """Record something the person did. 'faq' collects a list of topics."""
    with Session() as s:
        c = s.get(Contact, who[:80])
        if c is None:
            c = Contact(who=who[:80], signals="{}", stage="new", inbound_count=0)
            s.add(c)
        data = json.loads(c.signals or "{}")
        if key == "faq":
            topics = data.get("faq", [])
            if value not in topics:
                topics.append(value)
            data["faq"] = topics
        elif key == "unanswered":  # the last few questions the bot could not answer
            data["unanswered"] = (data.get("unanswered", []) + [value])[-5:]
        else:
            data[key] = value
        c.signals = json.dumps(data)
        s.commit()


def set_contact_name(who, name):
    with Session() as s:
        c = s.get(Contact, who[:80])
        if c is not None and name:
            c.name = name[:200]
            s.commit()


def get_contact(who):
    with Session() as s:
        c = s.get(Contact, who[:80])
        return contact_dict(c) if c else None


def list_contacts(limit=500):
    with Session() as s:
        rows = s.scalars(select(Contact).order_by(desc(Contact.last_inbound)).limit(limit))
        return [contact_dict(c) for c in rows]


def update_contact(who, **changes):
    with Session() as s:
        c = s.get(Contact, who[:80])
        if c is None:
            return None
        for k, v in changes.items():
            if k in CONTACT_FIELDS and v is not None:
                setattr(c, k, v)
        s.commit()
        return contact_dict(c)


def leads_for(who, phone=""):
    with Session() as s:
        q = select(Lead).order_by(desc(Lead.created))
        rows = [r for r in s.scalars(q.limit(500)) if (phone and r.phone == phone) or r.phone == who]
        return [_lead(r) for r in rows]


def conversations(limit=100):
    with Session() as s:
        groups = s.execute(
            select(Message.who, func.max(Message.created).label("last"), func.count(Message.id))
            .group_by(Message.who).order_by(desc("last")).limit(limit)).all()
        out = []
        for who, last, count in groups:
            preview = s.scalars(select(Message.text).where(Message.who == who)
                                .order_by(desc(Message.created)).limit(1)).first()
            out.append({"who": who, "last": _iso(last), "count": count, "preview": (preview or "")[:80]})
        return out


def thread(who, limit=300):
    with Session() as s:
        rows = s.scalars(select(Message).where(Message.who == who)
                         .order_by(desc(Message.created)).limit(limit)).all()
        return [{"created": _iso(r.created), "direction": r.direction, "sender": r.sender or "",
                 "text": r.text} for r in reversed(rows)]


# ---------------------------------------------------------------- language model data
def _intent(r):
    try:
        replies = json.loads(r.replies or "[]")
    except ValueError:
        replies = []
    return {"name": r.name, "category": r.category, "priority": r.priority, "handler": r.handler,
            "replies": replies, "enabled": bool(r.enabled)}


def list_intents():
    with Session() as s:
        return {r.name: _intent(r) for r in s.scalars(select(NluIntent))}


def save_intent(name, category="enquiry", priority="medium", handler="reply", replies=None, enabled=True):
    with Session() as s:
        row = s.get(NluIntent, name) or NluIntent(name=name)
        row.category, row.priority, row.handler = category, priority, handler
        row.replies = json.dumps(replies or [])
        row.enabled = enabled
        row.updated = _now()
        s.add(row)
        s.commit()
        return _intent(row)


def delete_intent(name):
    with Session() as s:
        s.query(NluExample).filter(NluExample.intent == name).delete()
        s.query(NluIntent).filter(NluIntent.name == name).delete()
        s.commit()


def list_examples(intent=None):
    with Session() as s:
        q = select(NluExample).order_by(NluExample.id)
        if intent:
            q = q.where(NluExample.intent == intent)
        return [{"id": r.id, "intent": r.intent, "text": r.text, "source": r.source} for r in s.scalars(q)]


def add_example(intent, text, source="staff"):
    """Add an example phrase unless it is already there. Returns True when a row was added."""
    text = " ".join(str(text or "").split())[:500]
    if not text:
        return False
    with Session() as s:
        if s.scalar(select(NluExample.id).where(NluExample.intent == intent, NluExample.text == text)):
            return False
        s.add(NluExample(intent=intent, text=text, source=source))
        s.commit()
        return True


def add_examples_bulk(rows, source="seed"):
    """rows: [(intent, text)]. Adds the ones that are not stored yet in one go. Returns how many were added."""
    with Session() as s:
        have = {(i, t) for i, t in s.execute(select(NluExample.intent, NluExample.text))}
        added = 0
        for intent, text in rows:
            text = " ".join(str(text or "").split())[:500]
            if text and (intent, text) not in have:
                have.add((intent, text))
                s.add(NluExample(intent=intent, text=text, source=source))
                added += 1
        s.commit()
        return added


def delete_example(example_id):
    with Session() as s:
        s.query(NluExample).filter(NluExample.id == example_id).delete()
        s.commit()


def add_unmatched(who, text, guess="", confidence=0.0):
    text = " ".join(str(text or "").split())[:500]
    if not text:
        return
    with Session() as s:
        if s.scalar(select(Unmatched.id).where(Unmatched.text == text, Unmatched.status == "new")):
            return
        s.add(Unmatched(who=who[:80], text=text, guess=guess or "", confidence=float(confidence)))
        s.commit()


def list_unmatched(status="new", limit=200):
    with Session() as s:
        rows = s.scalars(select(Unmatched).where(Unmatched.status == status)
                         .order_by(desc(Unmatched.created)).limit(limit))
        return [{"id": r.id, "who": r.who, "text": r.text, "guess": r.guess, "confidence": r.confidence,
                 "created": _iso(r.created)} for r in rows]


def label_unmatched(item_id, intent):
    """Staff says which intent a message was ('ignore' dismisses it). The text becomes a training example."""
    with Session() as s:
        row = s.get(Unmatched, item_id)
        if row is None:
            return None
        row.status, row.label = ("ignored", "") if intent == "ignore" else ("labelled", intent)
        text = row.text
        s.commit()
    if intent != "ignore":
        add_example(intent, text, "chat")
    return True


# ---------------------------------------------------------------- enquiries
PRIORITY_ORDER = {"low": 0, "medium": 1, "high": 2, "urgent": 3}


def _enquiry(r):
    try:
        reasons = json.loads(r.reasons or "[]")
    except ValueError:
        reasons = []
    return {"id": r.id, "who": r.who, "name": r.name, "category": r.category, "intent": r.intent,
            "priority": r.priority, "reasons": reasons, "status": r.status, "summary": r.summary,
            "last_text": r.last_text, "message_count": r.message_count, "created": _iso(r.created),
            "updated": _iso(r.updated)}


def _open_enquiry(s, who, category):
    return s.scalar(select(Enquiry).where(Enquiry.who == who[:80], Enquiry.category == category,
                                          Enquiry.status == "open").order_by(desc(Enquiry.updated)))


def upsert_enquiry(who, name, category, intent, priority, text, summary="", reasons=None, confidence=0.0):
    """One open enquiry per person and category: later messages update it and can only raise its priority."""
    with Session() as s:
        row = _open_enquiry(s, who, category)
        if row is None:
            row = Enquiry(who=who[:80], category=category, message_count=0, priority=priority, reasons="[]")
            s.add(row)
        if PRIORITY_ORDER.get(priority, 1) > PRIORITY_ORDER.get(row.priority or "low", 0):
            row.priority = priority
        old = json.loads(row.reasons or "[]")
        row.reasons = json.dumps(list(dict.fromkeys(old + (reasons or [])))[:6])
        row.intent, row.name = intent, (name or row.name or "")[:200]
        row.last_text = (text or "")[:500]
        if summary:
            row.summary = summary[:500]
        elif not row.summary:
            row.summary = (text or "")[:200]
        row.message_count = (row.message_count or 0) + 1
        row.confidence = float(confidence)
        row.updated = _now()
        s.commit()
        return _enquiry(row)


def set_enquiry_summary(who, category, summary):
    with Session() as s:
        row = _open_enquiry(s, who, category)
        if row:
            row.summary = summary[:500]
            s.commit()


def get_setting(key):
    with Session() as s:
        row = s.get(Setting, key)
        return row.value if row else None


def set_setting(key, value):
    with Session() as s:
        row = s.get(Setting, key)
        if row:
            row.value = value
        else:
            s.add(Setting(key=key, value=value))
        s.commit()
