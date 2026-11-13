"""WhatsApp sales chatbot for a solar company (Meta WhatsApp Cloud API webhook).

Environment variables:
  VERIFY_TOKEN       secret you type into Meta's webhook settings
  WHATSAPP_TOKEN     access token (temporary or system-user token)
  PHONE_NUMBER_ID    the WhatsApp phone number ID (test number works)
  APP_SECRET         optional; Meta app secret, enables signature checks
  GRAPH_VERSION      optional, default v21.0
"""
import hashlib
import hmac
import json
import logging
import os
import re
import time
from pathlib import Path

import requests
from flask import Flask, jsonify, request

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("solar-bot")

app = Flask(__name__)
CATALOG_PATH = Path(__file__).with_name("catalog.json")
CATALOG = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))

VERIFY_TOKEN = os.getenv("VERIFY_TOKEN", "")
WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN", "")
PHONE_NUMBER_ID = os.getenv("PHONE_NUMBER_ID", "")
APP_SECRET = os.getenv("APP_SECRET", "")
GRAPH_VERSION = os.getenv("GRAPH_VERSION", "v21.0")

# Conversation state per phone number (in memory: resets on restart, fine for a free tier).
SESSIONS: dict[str, dict] = {}
SEEN_IDS: dict[str, float] = {}  # message id -> time, to ignore Meta's retries


def money(n):
    return f"{CATALOG['currency']} {n:,.0f}"


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
    for p in CATALOG["packages"]:
        if p["daily_kwh"] >= daily_kwh:
            return p
    return CATALOG["packages"][-1]


def faq_answer(text):
    t = text.lower()
    for f in CATALOG["faqs"]:
        if any(k in t for k in f["keys"]):
            return f["answer"] + "\n\nReply *menu* for more options."
    return None


def log_lead(phone, name, data):
    """Leads go to the logs (Render dashboard -> Logs). Replace with a DB/ERP call later."""
    log.info("LEAD %s", json.dumps({"phone": phone, "name": name, **data}, ensure_ascii=False))


# ---------------------------------------------------------------- bot brain
def handle(phone, name, text):
    """Return the reply text for one incoming message."""
    raw = text.strip()
    t = raw.lower()
    s = SESSIONS.setdefault(phone, {"state": "menu"})

    if t in {"menu", "hi", "hello", "hey", "start", "0", "hie", "hesi"}:
        s.clear()
        s["state"] = "menu"
        return menu_text()

    state = s["state"]

    # --- multi-step flows first
    if state == "size_kwh":
        nums = re.findall(r"\d+(?:\.\d+)?", t.replace(",", ""))
        if not nums:
            return "Please reply with a number, e.g. *10* (kWh per day), or *menu* to cancel."
        value = float(nums[0])
        # A large number is almost certainly a monthly figure
        daily = value / 30 if value > 100 else value
        p = recommend(daily)
        s["state"] = "menu"
        s["recommended"] = p["name"]
        return (
            f"For about {daily:.0f} kWh/day, I'd suggest the *{p['name']}* "
            f"({money(p['price'])}).\n\nIt powers: {p['powers']}.\n\n"
            "This is an estimate; a technician confirms the final size at a site visit.\n"
            "Reply *quote* for a free quotation or *menu* to go back."
        )

    if state == "quote_name":
        s["lead_name"] = raw
        s["state"] = "quote_location"
        return "Thanks! What *town or suburb* is the installation for?"

    if state == "quote_location":
        s["lead_location"] = raw
        s["state"] = "quote_need"
        return "Which package or what do you want to power? (e.g. *Home 3kW*, or *fridge, TV and lights*)"

    if state == "quote_need":
        log_lead(phone, s.get("lead_name") or name, {
            "location": s.get("lead_location"),
            "need": raw,
            "suggested": s.get("recommended"),
        })
        s.clear()
        s["state"] = "menu"
        return (
            "✅ Thank you! Our team will contact you on this number shortly with your quotation.\n\n"
            "Reply *menu* for more options."
        )

    # --- a package number right after the package list shows its details
    if t.isdigit() and s.pop("last", None) == "packages":
        d = package_detail(t)
        if d:
            s["last"] = "packages"
            return d
    s.pop("last", None)

    # --- top-level commands
    if t in {"1", "packages", "prices", "price"}:
        s["state"] = "menu"
        s["last"] = "packages"
        return packages_text()

    if t in {"2", "size", "sizing"}:
        s["state"] = "size_kwh"
        return (
            "Let's size your system ⚡\nHow much electricity do you use?\n"
            "Reply with kWh *per day* (e.g. *10*) or *per month* from your bill (e.g. *300*)."
        )

    if t in {"3", "quote", "quotation"}:
        s["state"] = "quote_name"
        return "Great, let's get you a free quotation. What is your *full name*?"

    if t in {"4", "human", "agent", "person", "call"}:
        log_lead(phone, name, {"request": "wants a person"})
        return (
            f"Okay, a team member will message you soon. You can also call {CATALOG['phone']}.\n\n"
            "Reply *menu* for more options."
        )

    if t in {"5", "contact", "hours", "address", "location"}:
        return contact_text()

    return faq_answer(t) or (
        "Sorry, I didn't quite get that. " + "Reply with a number from the menu:\n\n" + menu_text()
    )


# ---------------------------------------------------------------- WhatsApp I/O
def send_text(to, body):
    if not (WHATSAPP_TOKEN and PHONE_NUMBER_ID):
        log.warning("WHATSAPP_TOKEN/PHONE_NUMBER_ID not set; would send to %s: %s", to, body[:80])
        return
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


def valid_signature(req):
    if not APP_SECRET:
        return True
    sig = req.headers.get("X-Hub-Signature-256", "")
    expected = "sha256=" + hmac.new(APP_SECRET.encode(), req.get_data(), hashlib.sha256).hexdigest()
    return hmac.compare_digest(sig, expected)


@app.get("/")
def home():
    return jsonify(status="ok", service="solar-whatsapp-bot")


@app.get("/privacy")
def privacy():
    c = CATALOG
    return (
        f"<!doctype html><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>Privacy Policy - {c['company']}</title>"
        "<body style='font-family:system-ui,sans-serif;max-width:700px;margin:2rem auto;padding:0 1rem;line-height:1.6'>"
        f"<h1>Privacy Policy</h1><p><b>{c['company']}</b> operates a WhatsApp assistant that answers questions "
        "about solar products and prepares quotations.</p>"
        "<h2>What we collect</h2><p>Your WhatsApp number, profile name and the messages you send us, "
        "including the name, location and requirements you give when requesting a quotation.</p>"
        "<h2>How we use it</h2><p>Only to reply to you, prepare quotations and contact you about your enquiry. "
        "We do not sell your data.</p>"
        "<h2>Sharing and retention</h2><p>Messages are processed through Meta's WhatsApp Business Platform. "
        "We keep enquiry details only as long as needed to serve you.</p>"
        f"<h2>Your rights</h2><p>To see or delete your data, contact us at {c['email']} or {c['phone']}.</p>"
        "</body>"
    )


@app.get("/_health")
def health():
    return "ok"


@app.get("/webhook")
def verify():
    if (request.args.get("hub.mode") == "subscribe"
            and VERIFY_TOKEN
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
                if kind == "text":
                    text = m["text"]["body"]
                elif kind == "interactive":
                    i = m["interactive"]
                    text = (i.get("button_reply") or i.get("list_reply") or {}).get("id", "")
                else:
                    send_text(phone, "I can only read text messages for now. Reply *menu* to see options.")
                    continue
                try:
                    send_text(phone, handle(phone, names.get(phone, ""), text))
                except Exception:  # never let one bad message fail the webhook
                    log.exception("handler error")
    return "ok", 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5001)), debug=False)
