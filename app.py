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
"""
import hashlib
import hmac
import json
import logging
import os
import re
import secrets
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

# Conversation state per user (memory only; resets on restart).
SESSIONS: dict[str, dict] = {}
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
def menu_text():
    c = CATALOG
    return (
        f"Welcome to *{c['company']}* ☀️\n"
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


def save_lead(phone, name, **data):
    source = "web" if phone.startswith("web:") else "whatsapp"
    shown_phone = data.pop("contact", None) or ("" if source == "web" else phone)
    lead = db.add_lead(source, shown_phone, name or "", **data)
    log.info("LEAD #%s %s %s", lead["id"], source, lead["kind"])


# ---------------------------------------------------------------- bot brain
def handle(phone, name, text):
    """Return the reply for one incoming message. `phone` is a number or "web:<id>"."""
    if len(SESSIONS) > 5000:
        SESSIONS.clear()
    raw = text.strip()
    t = raw.lower()
    is_web = phone.startswith("web:")
    s = SESSIONS.setdefault(phone, {"state": "menu"})

    if t in {"menu", "hi", "hello", "hey", "start", "0", "hie", "hesi"}:
        s.clear()
        s["state"] = "menu"
        return menu_text()

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
        return "Thanks! What *town or suburb* is the installation for?"

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
        save_lead(phone, s.get("lead_name") or name, location=s.get("lead_location", ""),
                  need=clip(raw, 1000), suggested=s.get("recommended", ""),
                  contact=s.get("lead_contact"))
        s.clear()
        s["state"] = "menu"
        return ("✅ Thank you! Our team will contact you shortly with your quotation.\n\n"
                "Reply *menu* for more options.")

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
        return (f"Okay, a team member will message you soon. You can also call {CATALOG['phone']}.\n\n"
                "Reply *menu* for more options.")

    if t in {"5", "contact", "hours", "address", "location"}:
        return contact_text()

    faq = faq_answer(t)
    if faq:
        sig(phone, "faq", faq[0])
        return faq[1]
    return "Sorry, I didn't quite get that. Reply with a number from the menu:\n\n" + menu_text()


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
                    contact = db.touch_contact(phone, names.get(phone, ""), "in")
                    db.add_message(phone, "in", text or f"[{kind} message]")
                    if contact["bot_paused"]:
                        continue  # a staff member is handling this chat
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


@app.post("/admin/api/clients/<who>/reply")
@admin_required
def admin_client_reply(who):
    c = db.get_contact(who)
    if not c:
        return jsonify(error="Not found."), 404
    a = leads.analyse(c)
    if not a["is_whatsapp"]:
        return jsonify(error="Website visitors cannot be messaged from here. Use the phone number they gave."), 400
    if not a["window_open"]:
        return jsonify(error="Their last message is over 24 hours old. WhatsApp only allows an approved "
                             "template message now (create one in WhatsApp Manager)."), 400
    text = clip(str((request.get_json(silent=True) or {}).get("text", "")), 1000)
    if not text:
        return jsonify(error="Type a message first."), 400
    if not send_text(who, text):
        return jsonify(error="WhatsApp did not accept the message. Check WHATSAPP_TOKEN and PHONE_NUMBER_ID "
                             "in Render."), 502
    db.add_message(who, "out", text, "staff")
    db.touch_contact(who, direction="out")
    if c["stage"] == "new":
        db.update_contact(who, stage="contacted")
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


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5001)), debug=False)
