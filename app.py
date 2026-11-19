"""Solar sales assistant: WhatsApp bot (Meta Cloud API) + customer web page + admin page.

Environment variables:
  VERIFY_TOKEN       secret typed into Meta's webhook settings
  WHATSAPP_TOKEN     WhatsApp access token
  PHONE_NUMBER_ID    WhatsApp phone number ID
  APP_SECRET         optional Meta app secret (enables webhook signature checks)
  ADMIN_PASSWORD     password for /admin (admin is disabled when unset)
  SECRET_KEY         signs the admin login cookie
  DATABASE_URL       PostgreSQL URL to keep leads/chats across restarts (SQLite otherwise)
  PUBLIC_WHATSAPP    optional digits-only WhatsApp number, shows a "Chat on WhatsApp" button
  ALERT_NUMBERS      optional staff WhatsApp numbers (digits, comma-separated) that get new-lead alerts
  ALERT_TEMPLATE     optional approved template name with one body variable, used when a staff
                     member has not messaged the bot in 24 hours (WhatsApp blocks free text then)
  ALERT_TEMPLATE_LANG  template language code (default en_US)
  INTERNAL_KEY       shared secret that lets the business platform send staff replies through the bot
"""
import datetime as dt
import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import threading
import time
from datetime import timedelta
from functools import wraps
from pathlib import Path

import requests
from flask import Flask, Response, jsonify, request, send_from_directory, session

import brain
import components
import db
import leads
import llm
import market
import nlu
import quote as quote_mod

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("solar-bot")

BASE = Path(__file__).parent
WEB = BASE / "web"

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY") or secrets.token_hex(32)
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=bool(os.getenv("RENDER")),
    PERMANENT_SESSION_LIFETIME=timedelta(hours=8),
)

VERIFY_TOKEN = os.getenv("VERIFY_TOKEN", "")
WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN", "")
PHONE_NUMBER_ID = os.getenv("PHONE_NUMBER_ID", "")
APP_SECRET = os.getenv("APP_SECRET", "")
GRAPH_VERSION = os.getenv("GRAPH_VERSION", "v21.0")
PUBLIC_WHATSAPP = re.sub(r"\D", "", os.getenv("PUBLIC_WHATSAPP", ""))
ALERT_NUMBERS = [n for n in (re.sub(r"\D", "", x) for x in os.getenv("ALERT_NUMBERS", "").split(",")) if n]
ALERT_TEMPLATE = os.getenv("ALERT_TEMPLATE", "").strip()
ALERT_TEMPLATE_LANG = os.getenv("ALERT_TEMPLATE_LANG", "en_US").strip()
INTERNAL_KEY = os.getenv("INTERNAL_KEY", "")


# ---------------------------------------------------------------- catalog
def validate_catalog(c):
    """Return an error message, or None when the catalog is usable."""
    if not isinstance(c, dict):
        return "The catalog must be a JSON object."
    for k in ("company", "currency", "phone", "email", "address", "hours"):
        if not isinstance(c.get(k), str):
            return f'"{k}" must be text.'
    pk = c.get("packages")
    if not isinstance(pk, list) or not pk:
        return '"packages" must be a non-empty list.'
    seen = set()
    for p in pk:
        if not isinstance(p, dict):
            return "Every package must be an object."
        for k in ("id", "name", "powers", "includes"):
            if not isinstance(p.get(k), str) or not p[k]:
                return f'Every package needs text for "{k}".'
        for k in ("price", "daily_kwh"):
            if not isinstance(p.get(k), (int, float)) or isinstance(p.get(k), bool):
                return f'Package "{p.get("name")}": "{k}" must be a number.'
        if p["id"] in seen:
            return f'Duplicate package id "{p["id"]}".'
        seen.add(p["id"])
    if "utc_offset" in c and (not isinstance(c["utc_offset"], (int, float)) or isinstance(c["utc_offset"], bool)):
        return '"utc_offset" must be a number of hours, e.g. 2.'
    if "response_time" in c and not isinstance(c["response_time"], str):
        return '"response_time" must be text, e.g. "1 hour".'
    oh = c.get("open_hours")
    if oh is not None:
        if not isinstance(oh, dict):
            return '"open_hours" must map day numbers (0 = Monday) to ["08:00", "17:00"].'
        for day, span in oh.items():
            if (day not in {"0", "1", "2", "3", "4", "5", "6"} or not isinstance(span, list) or len(span) != 2
                    or not all(isinstance(x, str) and re.fullmatch(r"\d{2}:\d{2}", x) for x in span)
                    or span[0] >= span[1]):
                return '"open_hours" must map day numbers (0 = Monday) to ["08:00", "17:00"].'
    if not isinstance(c.get("faqs"), list):
        return '"faqs" must be a list.'
    for f in c["faqs"]:
        if not (isinstance(f, dict) and isinstance(f.get("answer"), str)
                and isinstance(f.get("keys"), list) and f["keys"] and all(isinstance(k, str) for k in f["keys"])):
            return 'Every FAQ needs "keys" (list of words) and "answer" (text).'
        if "title" in f and not (isinstance(f["title"], str) and 0 < len(f["title"]) <= 24):
            return f'FAQ title "{f.get("title")}" must be 1-24 characters (it is shown on a WhatsApp button).'
    if "market_prices" in c and not isinstance(c["market_prices"], bool):
        return '"market_prices" must be true or false.'
    for p in pk:
        if "size_kva" in p and (not isinstance(p["size_kva"], (int, float)) or isinstance(p["size_kva"], bool)):
            return f'Package "{p.get("name")}": "size_kva" must be a number.'
    if "towns" in c and not (isinstance(c["towns"], list) and all(isinstance(x, str) and x for x in c["towns"])):
        return '"towns" must be a list of town names.'
    return None


def load_catalog():
    cat = json.loads((BASE / "catalog.json").read_text(encoding="utf-8"))
    saved = db.get_setting("catalog")
    if saved:
        try:
            data = json.loads(saved)
            if validate_catalog(data) is None:
                cat = data
        except ValueError:
            log.error("saved catalog is not valid JSON; using catalog.json")
    return cat


CATALOG = load_catalog()

# Conversation state lives in the database (db.get_state / db.save_state) so restarts don't lose it.
SEEN_IDS: dict[str, float] = {}
HITS: dict[str, list] = {}


def package_size(p):
    """System size in kVA: the package's "size_kva", or the number in its name ("Home 3kW")."""
    if isinstance(p.get("size_kva"), (int, float)) and not isinstance(p.get("size_kva"), bool):
        return float(p["size_kva"])
    m = re.search(r"(?i)(\d{1,2}(?:\.\d)?)\s?k(?:va|w)(?![a-z])", p.get("name", ""))
    return float(m.group(1)) if m else None


def market_range(p, town=""):
    """Current advertised price range for a package's size, or None (no data / switched off)."""
    if not CATALOG.get("market_prices", True) or CATALOG.get("currency", "").upper() not in ("USD", "US$", "$"):
        return None
    size = package_size(p)
    if not size:
        return None
    try:
        r = market.range_for(size, town)
    except Exception:
        log.exception("market range failed")
        return None
    return r if r and r["low"] < r["high"] else None


def price_text(p, town=""):
    """What the bot shows as a package's price: the live market range, else the catalog price."""
    r = market_range(p, town)
    return f"{CATALOG['currency']} {r['low']:,}–{r['high']:,}" if r else money(p["price"])


def market_note(p, town=""):
    r = market_range(p, town)
    if not r:
        return ""
    when = ""
    try:
        when = ", checked " + dt.datetime.fromisoformat(market.summary()["refreshed"]).strftime("%d %b").lstrip("0")
    except (TypeError, ValueError):
        pass
    return (f"📊 *Market check:* {r['size']:g}kVA systems in {r['where']} are advertised at "
            f"{CATALOG['currency']} {r['low']:,}–{r['high']:,} (typical {r['median']:,}), from {r['count']} "
            f"current prices at {r['sources']} sellers{when}. Your exact price comes with your quotation.")


def live_price(size, town=""):
    """The current advertised market price for a system size ({"median","size","count","sources","when"}), or None."""
    if not CATALOG.get("market_prices", True) or CATALOG.get("currency", "").upper() not in ("USD", "US$", "$"):
        return None
    try:
        r = market.range_for(size, town)
        if not r:
            return None
        when = ""
        try:
            when = dt.datetime.fromisoformat(market.summary()["refreshed"]).strftime("%d %b").lstrip("0")
        except (TypeError, ValueError):
            pass
        return {"median": r["median"], "size": r["size"], "count": r["count"], "sources": r["sources"], "when": when}
    except Exception:
        log.exception("live price failed")
        return None


def live_parts(size):
    if not CATALOG.get("market_prices", True):
        return None
    try:
        return components.live_parts(size) or None
    except Exception:
        log.exception("live parts failed")
        return None


def components_loop():
    time.sleep(40)
    while True:
        try:
            last = db.get_setting("component_prices")
            stale = True
            if last:
                when = json.loads(last).get("refreshed")
                stale = not when or dt.datetime.now(dt.timezone.utc) - dt.datetime.fromisoformat(when) > timedelta(hours=24)
            if stale:
                components.refresh()
        except Exception:
            log.exception("component refresh failed")
        time.sleep(1800)


def test_quote_reply():
    """A sample 5kVA quotation with its PDF and buttons, exactly as a customer would receive it."""
    q = quote_mod.build(5, live_price(5), live_parts(5))
    ref = "TEST-" + dt.date.today().strftime("%y%m%d")
    token = save_quote(q, "Test Client", "Harare", ref)
    text = (f"🧪 TEST quotation ({q['size_kva']:g}kVA, {quote_mod.money(q['total'])}). If you can open the PDF above "
            "and these buttons work, WhatsApp delivery is fine.")
    return R(text, [{"id": "btn:visit", "title": "Book site visit"}, {"id": "btn:question", "title": "Ask a question"},
                    {"id": "btn:fault", "title": "Report a problem"}],
             document={"token": token, "title": "Download quotation (PDF)", "caption": f"Quotation {ref}",
                       "short_text": text})


def save_quote(q, client="", where="", ref=""):
    """Store a quotation so its PDF can be fetched (web link) or regenerated (WhatsApp). Returns the secret token."""
    token = secrets.token_urlsafe(12)
    db.set_setting("quote:" + token, json.dumps({"q": q, "client": client, "where": where, "ref": ref,
                                                  "made": dt.date.today().isoformat()}))
    return token


def quote_pdf(token):
    raw = db.get_setting("quote:" + re.sub(r"[^A-Za-z0-9_-]", "", token or ""))
    if not raw:
        return None, None
    d = json.loads(raw)
    c = CATALOG
    pdf = quote_mod.pdf_bytes(d["q"], c["company"], {k: c.get(k, "") for k in ("phone", "email", "address")},
                              d.get("client", ""), d.get("where", ""), d.get("ref", ""))
    return pdf, f"GieBee-Quotation-{d.get('ref') or 'quote'}.pdf"


def money(n):
    return f"{CATALOG['currency']} {n:,.0f}"


def clip(s, n):
    return (s or "").strip()[:n]


def client_ip():
    return (request.headers.get("X-Forwarded-For", "").split(",")[0].strip()
            or request.remote_addr or "?")


def limited(key, limit, window=60):
    now = time.time()
    if len(HITS) > 5000:
        HITS.clear()
    q = [t for t in HITS.get(key, []) if now - t < window]
    if len(q) >= limit:
        HITS[key] = q
        return True
    q.append(now)
    HITS[key] = q
    return False


# ---------------------------------------------------------------- replies
DAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
DEFAULT_OPEN_HOURS = {"0": ["08:00", "17:00"], "1": ["08:00", "17:00"], "2": ["08:00", "17:00"],
                      "3": ["08:00", "17:00"], "4": ["08:00", "17:00"], "5": ["08:00", "13:00"]}


def local_now():
    """Business local time (naive). utc_offset defaults to 2 (Zimbabwe / South Africa)."""
    offset = CATALOG.get("utc_offset", 2)
    return (dt.datetime.now(dt.timezone.utc) + timedelta(hours=offset)).replace(tzinfo=None)


def _hours(day):
    return (CATALOG.get("open_hours") or DEFAULT_OPEN_HOURS).get(str(day))


def is_open(now=None):
    now = now or local_now()
    h = _hours(now.weekday())
    return bool(h) and h[0] <= now.strftime("%H:%M") < h[1]


def reply_promise(now=None):
    """When a person will answer, e.g. 'within 1 hour' or 'when we open tomorrow at 08:00'."""
    now = now or local_now()
    if is_open(now):
        return f"within {CATALOG.get('response_time') or '1 hour'}"
    for i in range(8):
        d = now + timedelta(days=i)
        h = _hours(d.weekday())
        if not h or (i == 0 and now.strftime("%H:%M") >= h[0]):
            continue
        when = "today" if i == 0 else "tomorrow" if i == 1 else f"on {DAYS[d.weekday()]}"
        return f"when we open {when} at {h[0]}"
    return "as soon as possible"


def first_name(name):
    parts = re.sub(r"[^\w' -]", "", name or "").split()
    return parts[0].capitalize()[:30] if parts else ""


# A reply is text, optionally with buttons (up to 3) or list rows (up to 10) for the question it ends with.
# Options are {"id": "btn:<key>", "title": ..., "description": ...}; the customer can always type instead.
def R(text, options=(), button="", document=None):
    return {"text": text, "options": [dict(o) for o in (options or [])][:10], "button": button or "",
            "document": document or None}


def as_reply(r):
    return r if isinstance(r, dict) else R(r)


def reply_log_text(r):
    return as_reply(r)["text"]


DEFAULT_TOWNS = ["Harare", "Chitungwiza", "Bulawayo", "Mutare", "Gweru", "Masvingo", "Kwekwe", "Kadoma",
                 "Marondera"]


def short(s, n):
    s = str(s or "")
    return s if len(s) <= n else s[:n - 1].rstrip() + "…"


def recommend(daily_kwh):
    for p in sorted(CATALOG["packages"], key=lambda p: p["daily_kwh"]):
        if p["daily_kwh"] >= daily_kwh:
            return p
    return max(CATALOG["packages"], key=lambda p: p["daily_kwh"])


def sig(who, key, value=True):
    """Remember something the person did in the chat (used to score potential clients)."""
    try:
        db.add_signal(who, key, value)
    except Exception:  # never let lead tracking break a reply
        log.exception("could not record signal %s", key)


# ---------------------------------------------------------------- staff alerts
ALERT_TITLES = {"quote": "New quotation request", "callback": "Client wants to talk to a person",
                "unanswered": "Question the bot could not answer", "paused": "Client wrote (bot paused)",
                "maintenance": "Maintenance request", "urgent": "URGENT: safety risk reported"}
ALERT_GAP = {"quote": 0, "callback": 600, "unanswered": 3600, "paused": 1800, "maintenance": 0, "urgent": 0}  # seconds between repeats


def send_template(to, text):
    """Fallback for alerts outside WhatsApp's 24-hour window: an approved template with one variable."""
    if not (ALERT_TEMPLATE and WHATSAPP_TOKEN and PHONE_NUMBER_ID):
        return False
    one_line = re.sub(r"\s+", " ", text)[:900]  # template variables may not contain new lines
    r = requests.post(
        f"https://graph.facebook.com/{GRAPH_VERSION}/{PHONE_NUMBER_ID}/messages",
        headers={"Authorization": f"Bearer {WHATSAPP_TOKEN}"},
        json={"messaging_product": "whatsapp", "to": to, "type": "template",
              "template": {"name": ALERT_TEMPLATE, "language": {"code": ALERT_TEMPLATE_LANG},
                           "components": [{"type": "body", "parameters": [{"type": "text", "text": one_line}]}]}},
        timeout=15,
    )
    if r.status_code >= 300:
        log.error("alert template failed %s: %s", r.status_code, r.text[:300])
        return False
    return True


def _deliver_alert(body, skip):
    for n in ALERT_NUMBERS:
        if n == skip:
            continue
        try:
            if not send_text(n, body) and not send_template(n, body):
                log.warning("alert to %s not delivered (staff must message the bot within 24h, "
                            "or set ALERT_TEMPLATE)", n)
        except Exception:
            log.exception("alert to %s failed", n)


def alert_team(who, name, kind, detail, state):
    """Tell staff on WhatsApp about something that needs a person. Never blocks the customer's reply."""
    if not ALERT_NUMBERS or who.startswith("web:"):
        return
    sent = state.setdefault("_alerts", {})
    if time.time() - sent.get(kind, 0) < ALERT_GAP.get(kind, 600):
        return
    sent[kind] = time.time()
    shown = name or "Unknown name"
    body = (f"🔔 *{ALERT_TITLES.get(kind, kind)}*\n{shown} (+{who})\n{clip(detail, 500)}\n\n"
            f"Reply to them: https://wa.me/{who}")
    threading.Thread(target=_deliver_alert, args=(body, who), daemon=True).start()


def market_summary_text():
    """What staff get when they send 'market' to the bot: current advertised ranges by size."""
    m = market.summary()
    if not m["sizes"]:
        return "No market prices collected yet. They refresh automatically every day."
    lines = ["📊 *Market prices (advertised, USD)*\n"]
    for size, e in m["sizes"].items():
        a = e["all"]
        if a["count"] < 2:
            continue
        towns = ", ".join(f"{t} {st['low']:,}–{st['high']:,}" for t, st in e["towns"].items()
                          if st["count"] >= market.MIN_PRICES)
        lines.append(f"*{size:g}kVA*: {a['low']:,}–{a['high']:,} (typical {a['median']:,}; {a['count']} prices, "
                     f"{a['sources']} sellers)" + (f"\n   {towns}" if towns else ""))
    try:
        lines.append("\nChecked " + dt.datetime.fromisoformat(m["refreshed"]).strftime("%d %b %H:%M UTC"))
    except (TypeError, ValueError):
        pass
    return "\n".join(lines)


def staff_summary():
    """What a staff member gets when they send 'leads' to the bot: who to contact today."""
    rows = []
    for c in db.list_contacts():
        if c["who"] in ALERT_NUMBERS or c["stage"] in ("won", "lost"):
            continue
        a = leads.analyse(c)
        rows.append((not a["follow_up_due"], -a["score"], c, a))
    rows.sort(key=lambda r: r[:2])
    if not rows:
        return "No open leads right now."
    lines = [f"*Open leads: {len(rows)}* (most urgent first)\n"]
    icon = {"hot": "🔥", "warm": "🌤", "cold": "❄️"}
    for _, _, c, a in rows[:8]:
        label = c["name"] or ("Website visitor" if c["who"].startswith("web:") else "+" + c["who"])
        due = " · follow-up due" if a["follow_up_due"] else ""
        lines.append(f"{icon[a['temperature']]} *{label}* ({a['score']}){due}\n{a['next_action'].split(' Note:')[0]}")
    lines.append("\nFull list: the WhatsApp tab in the business platform.")
    return "\n\n".join(lines)


def set_follow_up(who, days):
    """Schedule a follow-up unless staff already set one."""
    try:
        c = db.get_contact(who)
        if c and not c["follow_up"]:
            db.update_contact(who, follow_up=(local_now().date() + timedelta(days=days)).isoformat())
    except Exception:
        log.exception("could not set follow-up for %s", who)


def save_lead(phone, name, **data):
    source = "web" if phone.startswith("web:") else "whatsapp"
    shown_phone = data.pop("contact", None) or ("" if source == "web" else phone)
    lead = db.add_lead(source, shown_phone, name or "", **data)
    log.info("LEAD #%s %s %s", lead["id"], source, lead["kind"])


# ---------------------------------------------------------------- bot brain
def handle(phone, name, text):
    """Return the reply (dict from R()) for one incoming message. `phone` is a number or "web:<id>".

    The language model (nlu.py / brain.py) works out what the person wants. Where they are in the
    conversation is loaded from and saved to the database."""
    try:
        s = db.get_state(phone)
    except Exception:
        log.exception("could not load chat state")
        s = {}
    try:
        r = brain.respond(phone, name, text, s)
        return R(r["text"], r.get("options"), r.get("button"), r.get("document")) if isinstance(r, dict) else R(r)
    except Exception:
        log.exception("brain failed")
        return R("Sorry, something went wrong on my side. A member of our team will follow up with you.")
    finally:
        try:
            db.save_state(phone, s)
        except Exception:
            log.exception("could not save chat state")


# ---------------------------------------------------------------- WhatsApp I/O
def send_text(to, body):
    if not (WHATSAPP_TOKEN and PHONE_NUMBER_ID):
        log.warning("WHATSAPP_TOKEN/PHONE_NUMBER_ID not set; would send to %s", to)
        return False
    url = f"https://graph.facebook.com/{GRAPH_VERSION}/{PHONE_NUMBER_ID}/messages"
    r = requests.post(
        url,
        headers={"Authorization": f"Bearer {WHATSAPP_TOKEN}"},
        json={"messaging_product": "whatsapp", "to": to, "type": "text",
              "text": {"body": body, "preview_url": False}},
        timeout=15,
    )
    if r.status_code >= 300:
        log.error("send failed %s: %s", r.status_code, r.text[:300])
        return False
    return True


BODY_LIMIT = 1024   # WhatsApp: interactive message body
BUTTON_TITLE = 20   # reply button title
ROW_TITLE, ROW_DESC, LIST_LABEL = 24, 72, 20


def interactive_payload(text, options, button=""):
    """The WhatsApp "interactive" object for a reply: buttons for up to 3 short choices, a list otherwise.
    Returns (lead_text, interactive): lead_text is sent first as plain text when the body would be too long."""
    options = [o for o in options if o.get("id") and o.get("title")][:10]
    if not options:
        return text, None
    lead, body = "", text.strip()
    if len(body) > BODY_LIMIT:
        paras = body.split("\n\n")
        lead, body = "\n\n".join(paras[:-1]), paras[-1]
        if len(body) > BODY_LIMIT or len(lead) > 4000:
            lead, body = text.strip(), "Please choose an option, or type your answer:"
    use_buttons = (len(options) <= 3 and all(len(o["title"]) <= BUTTON_TITLE for o in options)
                   and not any(o.get("description") for o in options))
    if use_buttons:
        action = {"buttons": [{"type": "reply", "reply": {"id": o["id"][:256], "title": o["title"][:BUTTON_TITLE]}}
                              for o in options]}
        return lead, {"type": "button", "body": {"text": body}, "action": action}
    rows = []
    for o in options:
        row = {"id": o["id"][:200], "title": short(o["title"], ROW_TITLE)}
        if o.get("description"):
            row["description"] = short(o["description"], ROW_DESC)
        rows.append(row)
    label = short(button or "Choose", LIST_LABEL)
    return lead, {"type": "list", "body": {"text": body},
                  "action": {"button": label, "sections": [{"title": "Options", "rows": rows}]}}


def send_interactive(to, interactive):
    if not (WHATSAPP_TOKEN and PHONE_NUMBER_ID):
        log.warning("WHATSAPP_TOKEN/PHONE_NUMBER_ID not set; would send buttons to %s", to)
        return False
    r = requests.post(
        f"https://graph.facebook.com/{GRAPH_VERSION}/{PHONE_NUMBER_ID}/messages",
        headers={"Authorization": f"Bearer {WHATSAPP_TOKEN}"},
        json={"messaging_product": "whatsapp", "to": to, "type": "interactive", "interactive": interactive},
        timeout=15,
    )
    if r.status_code >= 300:
        log.error("interactive send failed %s: %s", r.status_code, r.text[:300])
        return False
    return True


def send_document(to, pdf, filename, caption=""):
    """Upload a PDF to WhatsApp and send it as a document. True when WhatsApp accepted it."""
    if not (WHATSAPP_TOKEN and PHONE_NUMBER_ID):
        log.warning("WHATSAPP_TOKEN/PHONE_NUMBER_ID not set; would send %s to %s", filename, to)
        return False
    base = f"https://graph.facebook.com/{GRAPH_VERSION}/{PHONE_NUMBER_ID}"
    auth = {"Authorization": f"Bearer {WHATSAPP_TOKEN}"}
    up = requests.post(base + "/media", headers=auth, data={"messaging_product": "whatsapp", "type": "application/pdf"},
                       files={"file": (filename, pdf, "application/pdf")}, timeout=30)
    if up.status_code >= 300 or not up.json().get("id"):
        log.error("media upload failed %s: %s", up.status_code, up.text[:300])
        return False
    r = requests.post(base + "/messages", headers=auth, timeout=15,
                      json={"messaging_product": "whatsapp", "to": to, "type": "document",
                            "document": {"id": up.json()["id"], "filename": filename, "caption": caption[:900]}})
    if r.status_code >= 300:
        log.error("document send failed %s: %s", r.status_code, r.text[:300])
        return False
    return True


def send_reply(to, reply):
    """Send a bot reply: with buttons or a list when it offers choices, plain text otherwise. If WhatsApp
    refuses the buttons, the text still goes out with the choices written in, so the customer can type one."""
    r = as_reply(reply)
    doc = r.get("document")
    if doc:
        try:
            pdf, name = quote_pdf(doc["token"])
            if pdf and send_document(to, pdf, name, doc.get("caption", "")):
                r = dict(r, text=doc.get("short_text") or r["text"])   # the PDF has the itemised list
        except Exception:
            log.exception("could not send quotation PDF")   # the full itemised text still goes out below
    if not r["options"]:
        return send_text(to, r["text"])
    lead, interactive = interactive_payload(r["text"], r["options"], r["button"])
    try:
        if (not lead or send_text(to, lead)) and send_interactive(to, interactive):
            return True
    except Exception:
        log.exception("interactive send failed")
    choices = " · ".join(re.sub(r"^\W+", "", o["title"]) for o in r["options"])
    return send_text(to, (r["text"] if not lead else interactive["body"]["text"]) + f"\n\n(Reply with: {choices})")


def valid_signature(req):
    if not APP_SECRET:
        return True
    sig = req.headers.get("X-Hub-Signature-256", "")
    expected = "sha256=" + hmac.new(APP_SECRET.encode(), req.get_data(), hashlib.sha256).hexdigest()
    return hmac.compare_digest(sig, expected)


@app.get("/webhook")
def verify():
    if (request.args.get("hub.mode") == "subscribe" and VERIFY_TOKEN
            and hmac.compare_digest(request.args.get("hub.verify_token", ""), VERIFY_TOKEN)):
        return request.args.get("hub.challenge", ""), 200
    return "Forbidden", 403


@app.post("/webhook")
def incoming():
    if not valid_signature(request):
        return "Bad signature", 403
    data = request.get_json(silent=True) or {}
    now = time.time()
    for k in [k for k, v in SEEN_IDS.items() if now - v > 3600]:
        SEEN_IDS.pop(k, None)

    for entry in data.get("entry", []):
        for change in entry.get("changes", []):
            value = change.get("value", {})
            names = {c.get("wa_id"): c.get("profile", {}).get("name", "")
                     for c in value.get("contacts", [])}
            for m in value.get("messages", []):
                mid = m.get("id")
                if mid in SEEN_IDS:
                    continue
                SEEN_IDS[mid] = now
                phone = m.get("from")
                kind = m.get("type")
                try:
                    shown = None  # what the client saw (button title), for the chat history
                    if kind == "text":
                        text = m["text"]["body"]
                    elif kind == "interactive":
                        i = m["interactive"]
                        picked = i.get("button_reply") or i.get("list_reply") or {}
                        shown = picked.get("title") or picked.get("id", "")
                        text = picked.get("id") or shown   # "btn:<key>": the bot reads the button, not its title
                    elif kind == "button":  # quick-reply button on a template message
                        text = m.get("button", {}).get("payload") or m.get("button", {}).get("text", "")
                    else:
                        text = ""
                    if phone in ALERT_NUMBERS and text.strip().lower() in {"market", "market prices"}:
                        send_text(phone, market_summary_text())
                        continue
                    if phone in ALERT_NUMBERS and text.strip().lower() in {"test quote", "testquote", "/testquote"}:
                        send_reply(phone, test_quote_reply())   # lets staff check the PDF and buttons on a real phone
                        continue
                    if phone in ALERT_NUMBERS and text.strip().lower() in {"leads", "report", "today"}:
                        # staff asking for today's list; this also keeps their 24-hour alert window open
                        send_text(phone, staff_summary())
                        continue
                    contact = db.touch_contact(phone, names.get(phone, ""), "in")
                    db.add_message(phone, "in", shown or text or f"[{kind} message]")
                    if contact["bot_paused"]:
                        # a staff member is handling this chat: no bot reply, but tell them the client wrote
                        st = db.get_state(phone)
                        alert_team(phone, contact["name"], "paused", f'They wrote: "{clip(text or kind, 300)}"', st)
                        db.save_state(phone, st)
                        continue
                    if text:
                        reply = handle(phone, names.get(phone, ""), text)
                    else:
                        reply = R("Thanks! I can only read typed messages here, not photos, voice notes or files. "
                                  "Could you type it out for me? If it's urgent, say \"call me\" and a person will contact you.")
                    if send_reply(phone, reply):
                        db.add_message(phone, "out", reply_log_text(reply))
                        db.touch_contact(phone, direction="out")
                except Exception:  # one bad message must not fail the webhook
                    log.exception("handler error")
    return "ok", 200


# ---------------------------------------------------------------- customer web page
@app.after_request
def headers(resp):
    resp.headers.setdefault("X-Content-Type-Options", "nosniff")
    resp.headers.setdefault("Referrer-Policy", "same-origin")
    if request.path.startswith("/admin"):
        resp.headers["Cache-Control"] = "no-store"
        resp.headers["X-Frame-Options"] = "DENY"
    return resp


@app.get("/")
def home():
    return send_from_directory(WEB, "index.html")


@app.get("/_health")
def health():
    return "ok"


@app.get("/api/catalog")
def api_catalog():
    c = CATALOG
    packages = [dict(p, price_text=price_text(p), market=bool(market_range(p))) for p in c["packages"]]
    return jsonify({k: c[k] for k in ("company", "currency", "phone", "email", "address", "hours")}
                   | {"packages": packages, "whatsapp": PUBLIC_WHATSAPP})


@app.get("/quote/<token>.pdf")
def quote_download(token):
    pdf, name = quote_pdf(token)
    if not pdf:
        return "Quotation not found.", 404
    return Response(pdf, mimetype="application/pdf", headers={"Content-Disposition": f'inline; filename="{name}"',
                                                              "Cache-Control": "private, max-age=3600"})


@app.post("/api/chat")
def api_chat():
    if limited("chat:" + client_ip(), 30):
        return jsonify(error="Too many messages. Please wait a minute."), 429
    d = request.get_json(silent=True) or {}
    sid = re.sub(r"[^A-Za-z0-9_-]", "", str(d.get("session", "")))[:40]
    text = clip(str(d.get("text", "")), 500)
    if not sid or not text:
        return jsonify(error="Missing message."), 400
    shown = clip(str(d.get("shown") or ""), 100)   # the title of a tapped button, for the chat history
    who = "web:" + sid
    db.touch_contact(who, "", "in")
    reply = handle(who, "", text)
    db.add_message(who, "in", shown or text)
    db.add_message(who, "out", reply_log_text(reply))
    db.touch_contact(who, direction="out")
    doc = reply.get("document")
    return jsonify(reply=reply["text"], options=[{"id": o["id"], "title": o["title"]} for o in reply["options"]],
                   document=({"url": f"/quote/{doc['token']}.pdf", "title": doc.get("title", "Download quotation (PDF)")}
                             if doc else None))


@app.post("/api/lead")
def api_lead():
    if limited("lead:" + client_ip(), 5, 600):
        return jsonify(error="Too many requests. Please try again later."), 429
    d = request.get_json(silent=True) or {}
    name, phone = clip(str(d.get("name", "")), 200), clip(str(d.get("phone", "")), 40)
    if not name or len(re.sub(r"\D", "", phone)) < 7:
        return jsonify(error="Please enter your name and a valid phone number."), 400
    db.add_lead("web", phone, name, clip(str(d.get("location", "")), 200),
                clip(str(d.get("need", "")), 1000), clip(str(d.get("package", "")), 200))
    return jsonify(ok=True)


@app.get("/privacy")
def privacy():
    c = CATALOG
    return (
        f"<!doctype html><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>Privacy Policy - {c['company']}</title>"
        "<body style='font-family:system-ui,sans-serif;max-width:700px;margin:2rem auto;padding:0 1rem;line-height:1.6'>"
        f"<h1>Privacy Policy</h1><p><b>{c['company']}</b> operates a WhatsApp and web assistant that answers "
        "questions about solar products and prepares quotations.</p>"
        "<h2>What we collect</h2><p>Your WhatsApp number, profile name and the messages you send us, "
        "plus the name, phone number, location and requirements you give when requesting a quotation.</p>"
        "<h2>How we use it</h2><p>Only to reply to you, prepare quotations and contact you about your enquiry. "
        "We do not sell your data.</p>"
        "<h2>Sharing and retention</h2><p>WhatsApp messages are processed through Meta's WhatsApp Business Platform. "
        "We keep enquiry details only as long as needed to serve you.</p>"
        f"<h2>Your rights</h2><p>To see or delete your data, contact us at {c['email']} or {c['phone']}.</p>"
        "</body>"
    )


# ---------------------------------------------------------------- admin
def admin_required(fn):
    @wraps(fn)
    def wrapper(*a, **k):
        if not session.get("admin"):
            return jsonify(error="Please log in."), 401
        if request.method != "GET" and request.headers.get("X-Requested-With") != "fetch":
            return jsonify(error="Bad request."), 400
        return fn(*a, **k)
    return wrapper


@app.get("/admin")
def admin_page():
    return send_from_directory(WEB, "admin.html")


@app.post("/admin/login")
def admin_login():
    password = os.getenv("ADMIN_PASSWORD", "")
    if not password:
        return jsonify(error="Admin is disabled: set ADMIN_PASSWORD on the server."), 503
    if limited("login:" + client_ip(), 5, 600):
        return jsonify(error="Too many attempts. Try again in 10 minutes."), 429
    d = request.get_json(silent=True) or {}
    if hmac.compare_digest(str(d.get("password", "")).encode(), password.encode()):
        session.clear()
        session["admin"] = True
        session.permanent = True
        return jsonify(ok=True)
    return jsonify(error="Wrong password."), 401


@app.post("/admin/logout")
def admin_logout():
    session.clear()
    return jsonify(ok=True)


@app.get("/admin/api/me")
def admin_me():
    return jsonify(loggedIn=bool(session.get("admin")), enabled=bool(os.getenv("ADMIN_PASSWORD")))


@app.get("/admin/api/leads")
@admin_required
def admin_leads():
    return jsonify(leads=db.list_leads(request.args.get("status")))


@app.patch("/admin/api/leads/<int:lead_id>")
@admin_required
def admin_lead_update(lead_id):
    d = request.get_json(silent=True) or {}
    row = db.update_lead(lead_id, d.get("status"), d.get("notes"))
    return (jsonify(row), 200) if row else (jsonify(error="Not found."), 404)


@app.get("/admin/api/conversations")
@admin_required
def admin_conversations():
    return jsonify(conversations=db.conversations())


@app.get("/admin/api/conversations/<path:who>")
@admin_required
def admin_thread(who):
    return jsonify(messages=db.thread(who))


def _client_json(c):
    a = leads.analyse(c)
    out = {k: v for k, v in c.items()}
    for k in ("first_seen", "last_inbound", "last_outbound"):
        out[k] = db._iso(c[k])
    return out | a


@app.get("/admin/api/clients")
@admin_required
def admin_clients():
    rows = [_client_json(c) for c in db.list_contacts()]
    rows.sort(key=lambda r: (not r["follow_up_due"], r["stage"] in ("won", "lost"), -r["score"]))
    return jsonify(clients=rows, stages=list(leads.STAGES))


@app.get("/admin/api/clients/<who>")
@admin_required
def admin_client(who):
    c = db.get_contact(who)
    if not c:
        return jsonify(error="Not found."), 404
    return jsonify(client=_client_json(c), messages=db.thread(who),
                   requests=db.leads_for(who, c["signals"].get("contact", "")))


@app.patch("/admin/api/clients/<who>")
@admin_required
def admin_client_update(who):
    d = request.get_json(silent=True) or {}
    changes = {}
    if "stage" in d:
        if d["stage"] not in leads.STAGES:
            return jsonify(error="Unknown stage."), 400
        changes["stage"] = d["stage"]
    if "follow_up" in d:
        f = str(d["follow_up"] or "")
        if f and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", f):
            return jsonify(error="Follow-up date must be YYYY-MM-DD."), 400
        changes["follow_up"] = f
    if "notes" in d:
        changes["notes"] = clip(str(d["notes"] or ""), 2000)
    if "assigned" in d:
        changes["assigned"] = clip(str(d["assigned"] or ""), 100)
    if "bot_paused" in d:
        changes["bot_paused"] = bool(d["bot_paused"])
    c = db.update_contact(who, **changes)
    return (jsonify(client=_client_json(c)), 200) if c else (jsonify(error="Not found."), 404)


def staff_reply(who, text):
    """Send a staff member's message to a client. Returns (json, status)."""
    c = db.get_contact(who)
    if not c:
        return {"error": "Not found."}, 404
    a = leads.analyse(c)
    if not a["is_whatsapp"]:
        return {"error": "Website visitors cannot be messaged from here. Use the phone number they gave."}, 400
    if not a["window_open"]:
        return {"error": "Their last message is over 24 hours old. WhatsApp only allows an approved "
                         "template message now. Call them, or wait for them to write again."}, 400
    text = clip(str(text or ""), 1000)
    if not text:
        return {"error": "Type a message first."}, 400
    if not send_text(who, text):
        return {"error": "WhatsApp did not accept the message. Check WHATSAPP_TOKEN and PHONE_NUMBER_ID "
                         "in Render."}, 502
    db.add_message(who, "out", text, "staff")
    db.touch_contact(who, direction="out")
    db.add_signal(who, "needs_human", False)  # a person has now answered
    if c["stage"] == "new":
        db.update_contact(who, stage="contacted")
    return {"ok": True}, 200


@app.post("/admin/api/clients/<who>/reply")
@admin_required
def admin_client_reply(who):
    body, status = staff_reply(who, (request.get_json(silent=True) or {}).get("text", ""))
    return jsonify(body), status


@app.post("/internal/reply")
def internal_reply():
    """Staff replies sent from the business platform (server to server, shared INTERNAL_KEY)."""
    if not INTERNAL_KEY:
        return jsonify(error="Replies from the platform are not set up (INTERNAL_KEY missing on the bot)."), 503
    if not _internal_ok():
        return jsonify(error="Forbidden."), 403
    d = request.get_json(silent=True) or {}
    body, status = staff_reply(str(d.get("who", ""))[:80], d.get("text", ""))
    if status == 200:
        log.info("staff reply to %s by %s", d.get("who"), clip(str(d.get("by", "")), 100))
    return jsonify(body), status


@app.get("/admin/api/market")
@admin_required
def admin_market():
    st = market.status()
    sources = [dict(src, **st.get(src["name"], {})) for src in market.get_sources()]
    prices = sorted(db.market_prices(), key=lambda r: (r["size_kva"], r["price"]))
    return jsonify(summary=market.summary(), sources=sources, prices=prices,
                   enabled=bool(CATALOG.get("market_prices", True)), min_prices=market.MIN_PRICES)


@app.post("/admin/api/market/refresh")
@admin_required
def admin_market_refresh():
    if limited("market-refresh", 3, 3600):
        return jsonify(error="Refreshed recently. To be fair to the sites we read, wait a while."), 429
    threading.Thread(target=market.refresh, daemon=True).start()
    return jsonify(ok=True, message="Refresh started. It takes about a minute.")


@app.put("/admin/api/market/sources")
@admin_required
def admin_market_sources():
    data = request.get_json(silent=True)
    err = market.validate_sources(data)
    if err:
        return jsonify(error=err), 400
    market.set_sources([{"name": x["name"].strip()[:120], "url": x["url"].strip()[:500],
                         "location": x.get("location", "").strip()[:80]} for x in data])
    market._cache.clear()
    return jsonify(ok=True)


@app.get("/admin/api/catalog")
@admin_required
def admin_catalog_get():
    return jsonify(catalog=CATALOG)


@app.put("/admin/api/catalog")
@admin_required
def admin_catalog_put():
    data = request.get_json(silent=True)
    err = validate_catalog(data)
    if err:
        return jsonify(error=err), 400
    db.set_setting("catalog", json.dumps(data, ensure_ascii=False))
    CATALOG.clear()
    CATALOG.update(data)
    return jsonify(ok=True)


# ---------------------------------------------------------------- language model (called by the business platform)
def _internal_ok():
    key = request.headers.get("X-Internal-Key", "")
    return bool(INTERNAL_KEY) and hmac.compare_digest(key.encode(), INTERNAL_KEY.encode())


_train_lock = threading.Lock()
_train_state = {"running": False, "error": "", "finished_at": None}


def _train_status():
    return dict(nlu.engine.info, training=_train_state["running"], train_error=_train_state["error"],
                trained_at=_train_state["finished_at"], llm=llm.status())


def _train_in_background():
    try:
        brain.retrain()
        _train_state["error"] = ""
    except Exception as exc:
        log.exception("training failed")
        _train_state["error"] = f"Training failed ({type(exc).__name__}). The previous model is still answering."
    finally:
        _train_state["finished_at"] = int(time.time())
        _train_state["running"] = False
        _train_lock.release()


@app.get("/internal/nlu/status")
def nlu_status():
    if not _internal_ok():
        return jsonify(error="Forbidden."), 403
    return jsonify(_train_status())


@app.post("/internal/nlu/train")
def nlu_train():
    """Rebuild the model from the example phrases in the database (after staff edit them).

    Training takes ~40 s on a small server, longer than the platform's web requests may wait, so it runs in the
    background: this answers at once with training=true and the platform polls /internal/nlu/status until it is
    false. Send {"wait": true} to train synchronously instead (tests, scripts)."""
    if not _internal_ok():
        return jsonify(error="Forbidden."), 403
    if (request.get_json(silent=True) or {}).get("wait"):
        with _train_lock:
            return jsonify(brain.retrain())
    if _train_lock.acquire(blocking=False):
        _train_state.update(running=True, error="")
        threading.Thread(target=_train_in_background, daemon=True, name="nlu-train").start()
    return jsonify(_train_status())


@app.post("/internal/nlu/test")
def nlu_test():
    """What would the bot make of this message? Used by the platform's 'Test a phrase' box."""
    if not _internal_ok():
        return jsonify(error="Forbidden."), 403
    text = clip(str((request.get_json(silent=True) or {}).get("text", "")), 500)
    if not text:
        return jsonify(error="Type a message to test."), 400
    a = brain.analyze(text, [t for t in (CATALOG.get("towns") or []) if isinstance(t, str)])
    return jsonify(a)


brain.start(CATALOG)


if os.getenv("MARKET_AUTO", "1") != "0":
    threading.Thread(target=market.loop, daemon=True, name="market-refresh").start()
    threading.Thread(target=components_loop, daemon=True, name="parts-refresh").start()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5001)), debug=False)
