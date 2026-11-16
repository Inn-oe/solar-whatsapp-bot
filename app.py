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
from flask import Flask, jsonify, request, send_from_directory, session

import db
import leads

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
                and isinstance(f.get("keys"), list) and all(isinstance(k, str) for k in f["keys"])):
            return 'Every FAQ needs "keys" (list of words) and "answer" (text).'
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


def menu_text(first=""):
    c = CATALOG
    hello = f"Hi {first} 👋\n" if first else ""
    return (
        f"{hello}Welcome to *{c['company']}* ☀️\n"
        "How can I help? Reply with a number:\n\n"
        "1️⃣ Solar packages & prices\n"
        "2️⃣ Size a system for my home or business\n"
        "3️⃣ Get a free quotation\n"
        "4️⃣ Talk to a person\n"
        "5️⃣ Contact & hours\n\n"
        "You can also just ask a question, e.g. _warranty_ or _how long is installation_."
    )


def packages_text():
    lines = ["*Our solar packages*\n"]
    for p in CATALOG["packages"]:
        lines.append(f"{p['id']}. *{p['name']}* - {money(p['price'])}\n   {p['powers']}")
    lines.append("\nReply with a package number for details, or *menu* to go back.")
    return "\n".join(lines)


def package_detail(pid):
    for p in CATALOG["packages"]:
        if p["id"] == pid:
            return (
                f"*{p['name']}* - {money(p['price'])}\n\n"
                f"Powers: {p['powers']}\n\n"
                f"Includes: {p['includes']}\n\n"
                "Reply *quote* to request a free quotation, or *menu* to go back."
            )
    return None


def contact_text():
    c = CATALOG
    return (
        f"*{c['company']}*\n📞 {c['phone']}\n✉️ {c['email']}\n📍 {c['address']}\n"
        f"🕒 {c['hours']}\n\nReply *menu* for the main menu."
    )


def recommend(daily_kwh):
    for p in sorted(CATALOG["packages"], key=lambda p: p["daily_kwh"]):
        if p["daily_kwh"] >= daily_kwh:
            return p
    return max(CATALOG["packages"], key=lambda p: p["daily_kwh"])


def faq_answer(text):
    """Return (topic, answer) for the first matching FAQ, or None."""
    t = text.lower()
    for f in CATALOG["faqs"]:
        if any(k.lower() in t for k in f["keys"]):
            keys = [k.lower() for k in f["keys"]]
            topic = "payment" if PAYMENT_WORDS & set(keys) else keys[0]
            return topic, f["answer"] + "\n\nReply *menu* for more options."
    return None


def sig(who, key, value=True):
    """Remember something the person did in the chat (used to score potential clients)."""
    try:
        db.add_signal(who, key, value)
    except Exception:  # never let lead tracking break a reply
        log.exception("could not record signal %s", key)


PAYMENT_WORDS = {"pay", "payment", "finance", "instalment", "installment", "deposit"}
SMALL_TALK = {"ok", "okay", "k", "thanks", "thank you", "thank u", "thanx", "thx", "yes", "no", "sure", "cool",
              "fine", "alright", "noted", "great", "nice", "👍", "🙏", "good", "bye", "tatenda", "ndatenda"}

# ---------------------------------------------------------------- staff alerts
ALERT_TITLES = {"quote": "New quotation request", "callback": "Client wants to talk to a person",
                "unanswered": "Question the bot could not answer", "paused": "Client wrote (bot paused)"}
ALERT_GAP = {"quote": 0, "callback": 600, "unanswered": 3600, "paused": 1800}  # seconds between repeats


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


def reset(s):
    """Back to the main menu, keeping long-lived memory (keys starting with '_')."""
    keep = {k: v for k, v in s.items() if k.startswith("_")}
    s.clear()
    s.update(keep)
    s["state"] = "menu"


def save_lead(phone, name, **data):
    source = "web" if phone.startswith("web:") else "whatsapp"
    shown_phone = data.pop("contact", None) or ("" if source == "web" else phone)
    lead = db.add_lead(source, shown_phone, name or "", **data)
    log.info("LEAD #%s %s %s", lead["id"], source, lead["kind"])


# ---------------------------------------------------------------- bot brain
def handle(phone, name, text):
    """Return the reply for one incoming message. `phone` is a number or "web:<id>".

    Where the person is in the conversation is loaded from and saved to the database."""
    try:
        s = db.get_state(phone)
    except Exception:
        log.exception("could not load chat state")
        s = {}
    s.setdefault("state", "menu")
    try:
        return _handle(phone, name, text, s)
    finally:
        try:
            db.save_state(phone, s)
        except Exception:
            log.exception("could not save chat state")


def _handle(phone, name, text, s):
    raw = text.strip()
    t = raw.lower()
    is_web = phone.startswith("web:")
    first = first_name(s.get("lead_name") or name)

    if t in {"menu", "hi", "hello", "hey", "start", "0", "hie", "hesi", "hi there", "good morning",
             "good afternoon", "good evening", "mhoro", "makadii", "sawubona"}:
        reset(s)
        return menu_text(first)

    state = s["state"]

    # --- multi-step flows
    if state == "size_kwh":
        nums = re.findall(r"\d+(?:\.\d+)?", t.replace(",", ""))
        if not nums:
            return "Please reply with a number, e.g. *10* (kWh per day), or *menu* to cancel."
        value = float(nums[0])
        daily = value / 30 if value > 100 else value  # a large number is a monthly figure
        p = recommend(daily)
        s["state"] = "menu"
        s["recommended"] = p["name"]
        sig(phone, "sized_kwh", round(daily, 1))
        sig(phone, "recommended", p["name"])
        return (
            f"For about {daily:.0f} kWh/day, I'd suggest the *{p['name']}* "
            f"({money(p['price'])}).\n\nIt powers: {p['powers']}.\n\n"
            "This is an estimate; a technician confirms the final size at a site visit.\n"
            "Reply *quote* for a free quotation or *menu* to go back."
        )

    if state == "quote_name":
        s["lead_name"] = clip(raw, 200)
        s["state"] = "quote_location"
        db.set_contact_name(phone, s["lead_name"])
        sig(phone, "quote_step", "location")
        hello = first_name(s["lead_name"])
        return f"Thanks{', ' + hello if hello else ''}! What *town or suburb* is the installation for?"

    if state == "quote_location":
        s["lead_location"] = clip(raw, 200)
        sig(phone, "location", s["lead_location"])
        sig(phone, "quote_step", "contact" if is_web else "need")
        if is_web:
            s["state"] = "quote_contact"
            return "What *phone number* (WhatsApp if possible) can we reach you on?"
        s["state"] = "quote_need"
        return "Which package or what do you want to power? (e.g. *Home 3kW*, or *fridge, TV and lights*)"

    if state == "quote_contact":
        digits = re.sub(r"\D", "", raw)
        if len(digits) < 7:
            return "That doesn't look like a phone number. Please type it with the country code, e.g. *+263 77 123 4567*."
        s["lead_contact"] = clip(raw, 40)
        s["state"] = "quote_need"
        sig(phone, "contact", s["lead_contact"])
        sig(phone, "quote_step", "need")
        return "Which package or what do you want to power? (e.g. *Home 3kW*, or *fridge, TV and lights*)"

    if state == "quote_need":
        sig(phone, "need", clip(raw, 300))
        sig(phone, "quote_completed")
        sig(phone, "quote_step", "done")
        lead_name = s.get("lead_name") or name
        where = s.get("lead_location", "")
        save_lead(phone, lead_name, location=where, need=clip(raw, 1000),
                  suggested=s.get("recommended", ""), contact=s.get("lead_contact"))
        set_follow_up(phone, 2)
        alert_team(phone, lead_name, "quote",
                   f"Needs: {clip(raw, 300)}" + (f"\nWhere: {where}" if where else "")
                   + (f"\nBot suggested: {s['recommended']}" if s.get("recommended") else ""), s)
        promise = reply_promise()
        reset(s)
        if is_web:
            return (f"✅ Thank you{', ' + first if first else ''}! Your request is saved. "
                    f"Our team will contact you {promise} with your quotation.\n\nReply *menu* for more options.")
        return (f"✅ Thank you{', ' + first if first else ''}! Your request is saved.\n\n"
                f"One of our team will message you here {promise} with your quotation, "
                "and arrange a free site visit to confirm the size.\n\nReply *menu* for more options.")

    # --- a package number right after the package list shows its details
    if t.isdigit() and s.pop("last", None) == "packages":
        d = package_detail(t)
        if d:
            s["last"] = "packages"
            name = next((p["name"] for p in CATALOG["packages"] if p["id"] == t), t)
            sig(phone, "package", name)
            return d
    s.pop("last", None)

    # --- top-level commands
    if t in {"1", "packages", "prices", "price"}:
        s["state"] = "menu"
        s["last"] = "packages"
        sig(phone, "viewed_packages")
        return packages_text()

    if t in {"2", "size", "sizing"}:
        s["state"] = "size_kwh"
        return (
            "Let's size your system ⚡\nHow much electricity do you use?\n"
            "Reply with kWh *per day* (e.g. *10*) or *per month* from your bill (e.g. *300*)."
        )

    if t in {"3", "quote", "quotation"}:
        s["state"] = "quote_name"
        sig(phone, "quote_requested")
        sig(phone, "quote_step", "name")
        return "Great, let's get you a free quotation. What is your *full name*?"

    if t in {"4", "human", "agent", "person", "call"}:
        if is_web:
            return (f"Please use the *Request a quote* form on this page and leave your number, "
                    f"or call {CATALOG['phone']}.\n\nReply *menu* for more options.")
        sig(phone, "callback")
        save_lead(phone, name, kind="callback", need="Asked to speak to a person")
        set_follow_up(phone, 0)
        alert_team(phone, s.get("lead_name") or name, "callback", "Asked to speak to a person.", s)
        return (f"Sure{', ' + first if first else ''}. A member of our team will message you here {reply_promise()}. "
                f"You can also call {CATALOG['phone']}.\n\nReply *menu* for more options.")

    if t in {"5", "contact", "hours", "address", "location"}:
        return contact_text()

    faq = faq_answer(t)
    if faq:
        sig(phone, "faq", faq[0])
        return faq[1]

    if t.strip(" .!") in SMALL_TALK or len(t) <= 2:
        return "👍 Reply *menu* any time to see packages, size a system or get a quote."

    # Something the bot can't answer: hand it to a person instead of looping on the menu.
    sig(phone, "unanswered", clip(raw, 200))
    sig(phone, "needs_human", True)
    promise = reply_promise()
    if is_web:
        return (f"Good question. I can't answer that one myself. Please use *Request a quote* on this page "
                f"and leave your number, and a person will reply {promise}.\n\nOr reply *menu* for options.")
    alert_team(phone, s.get("lead_name") or name, "unanswered", f'They asked: "{clip(raw, 300)}"', s)
    return (f"Good question{', ' + first if first else ''}. I've passed it to our team, and a person will "
            f"reply here {promise}.\n\nMeanwhile, reply *menu* to see packages, size a system or get a quote.")


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
                    if kind == "text":
                        text = m["text"]["body"]
                    elif kind == "interactive":
                        i = m["interactive"]
                        text = (i.get("button_reply") or i.get("list_reply") or {}).get("id", "")
                    else:
                        text = ""
                    if phone in ALERT_NUMBERS and text.strip().lower() in {"leads", "report", "today"}:
                        # staff asking for today's list; this also keeps their 24-hour alert window open
                        send_text(phone, staff_summary())
                        continue
                    contact = db.touch_contact(phone, names.get(phone, ""), "in")
                    db.add_message(phone, "in", text or f"[{kind} message]")
                    if contact["bot_paused"]:
                        # a staff member is handling this chat: no bot reply, but tell them the client wrote
                        st = db.get_state(phone)
                        alert_team(phone, contact["name"], "paused", f'They wrote: "{clip(text or kind, 300)}"', st)
                        db.save_state(phone, st)
                        continue
                    if text:
                        reply = handle(phone, names.get(phone, ""), text)
                    else:
                        reply = "I can only read text messages for now. Reply *menu* to see options."
                    if send_text(phone, reply):
                        db.add_message(phone, "out", reply)
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
    return jsonify({k: c[k] for k in ("company", "currency", "phone", "email", "address", "hours")}
                   | {"packages": c["packages"], "whatsapp": PUBLIC_WHATSAPP})


@app.post("/api/chat")
def api_chat():
    if limited("chat:" + client_ip(), 30):
        return jsonify(error="Too many messages. Please wait a minute."), 429
    d = request.get_json(silent=True) or {}
    sid = re.sub(r"[^A-Za-z0-9_-]", "", str(d.get("session", "")))[:40]
    text = clip(str(d.get("text", "")), 500)
    if not sid or not text:
        return jsonify(error="Missing message."), 400
    who = "web:" + sid
    db.touch_contact(who, "", "in")
    reply = handle(who, "", text)
    db.add_message(who, "in", text)
    db.add_message(who, "out", reply)
    db.touch_contact(who, direction="out")
    return jsonify(reply=reply)


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
    if not hmac.compare_digest(request.headers.get("X-Internal-Key", "").encode(), INTERNAL_KEY.encode()):
        return jsonify(error="Forbidden."), 403
    d = request.get_json(silent=True) or {}
    body, status = staff_reply(str(d.get("who", ""))[:80], d.get("text", ""))
    if status == 200:
        log.info("staff reply to %s by %s", d.get("who"), clip(str(d.get("by", "")), 100))
    return jsonify(body), status


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


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5001)), debug=False)
