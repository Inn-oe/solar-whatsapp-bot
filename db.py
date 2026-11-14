"""Storage for leads, chat logs and the editable catalog.

SQLite file by default; set DATABASE_URL (PostgreSQL) to keep data across restarts on Render.
"""
import datetime as dt
import os

from sqlalchemy import Column, DateTime, Integer, String, Text, create_engine, desc, func, select
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
    text = Column(Text)


class Setting(Base):
    __tablename__ = "settings"
    key = Column(String(50), primary_key=True)
    value = Column(Text)


Base.metadata.create_all(engine)


def _iso(d):
    return d.isoformat() if d else None


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


def add_message(who, direction, text):
    with Session() as s:
        s.add(Message(who=who[:80], direction=direction, text=(text or "")[:2000]))
        s.commit()


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
        return [{"created": _iso(r.created), "direction": r.direction, "text": r.text}
                for r in reversed(rows)]


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
