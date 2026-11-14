"""Storage for leads, chat logs and the editable catalog.

SQLite file by default; set DATABASE_URL (PostgreSQL) to keep data across restarts on Render.
"""
import datetime as dt
import json
import os

from sqlalchemy import Boolean, Column, DateTime, Integer, String, Text, create_engine, desc, func, select
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


Base.metadata.create_all(engine)


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
