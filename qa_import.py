"""Load a question-and-answer training sheet (CSV) into the bot's intents and example phrases.

Columns used: category, intent, customer_question, recommended_answer, response_mode, required_fields, risk.
Each row's question becomes an example phrase. Rows that mean the same as one of the bot's built-in intents
(price, quote, fault, human, contact ...) are merged into it, so the bot's multi-step conversations keep working;
every other CSV intent becomes its own intent called qa_<intent>, answering with the sheet's recommended answer.

Safe to run again: it only adds missing intents and phrases, and never overwrites what staff edited.
    python qa_import.py training/solar_customer_qa.csv
"""
import csv
import hashlib
import logging
import re
import sys
from pathlib import Path

import db

log = logging.getLogger("solar-bot.qa")

# CSV intent -> built-in intent with the same meaning (only the example phrases are added)
MERGE = {
    "price_general": "price", "price_3kva": "price", "price_5kva": "price", "price_8kva": "price",
    "price_10kva": "price", "quote": "buy_system", "shona_quote": "buy_system", "site_visit": "site_visit",
    "upgrade": "upgrade_system", "appliance_list": "sizing",
    "not_charging": "maintenance_fault", "inverter_fault": "maintenance_fault", "shutdown": "maintenance_fault",
    "low_generation": "maintenance_fault", "battery_draining": "maintenance_fault", "shona_fault": "maintenance_fault",
    "technician": "maintenance_fault", "maintenance": "maintenance_service", "warranty_claim": "warranty_claim",
    "complaint": "complaint", "human": "talk_to_human", "contact": "contact_info", "location": "contact_info",
    "opening_hours": "contact_info", "service_areas": "contact_info",
}
# extra ways to say the same thing, added on top of the sheet's phrases (sheet intent -> phrases)
EXTRA_PHRASES = {
    "overheating": ["my battery is swollen", "i got a shock from my inverter", "the inverter is smoking",
                    "there is smoke coming from my solar system", "my inverter smells like burning",
                    "the battery is bulging", "the inverter is making sparks", "something is on fire in my solar setup"],
    "water_damage": ["water is leaking onto my inverter", "the inverter got wet in the storm", "rain got into my battery box",
                     "my solar room is flooded", "water got into the inverter"],
}
# questions that are only a greeting: left to the greeting intent
SKIP_QUESTIONS = {"hello", "hi"}

PURCHASE_HANDOFF = {"bulk", "business", "farm"}     # start the quotation conversation
EMERGENCY = {"overheating", "water_damage"}          # safety advice first, urgent
TEAM_MODES = {"lookup", "policy_lookup", "regulatory_lookup", "quote_lookup", "authenticated_lookup",
              "schedule_lookup", "calculation", "technical_review", "quote", "workflow", "sales_handoff",
              "troubleshooting", "privacy_workflow"}


ASKS_FOR_INFO = re.compile(r"(please (?:share|tell|send|list|note)|share your|tell me|send me|list your|which (?:branch|brand|package|product)|what appliances)", re.I)


def describe(csv_category, intent, mode, risk, required="", answer=""):
    """-> (category, priority, handler) for a CSV intent that becomes its own bot intent."""
    if intent in EMERGENCY or risk == "critical" or mode == "emergency":
        return "maintenance", "urgent", "maintenance"
    if intent in PURCHASE_HANDOFF:
        return "purchase", "high", "purchase"
    if mode == "refuse":
        return "enquiry", "low", "reply"
    if intent == "refund":
        return "enquiry", "high", "answer_team"
    if intent == "payment_confirmation":
        return "enquiry", "high", "answer_team"
    if mode in ("escalate", "handoff"):
        return "enquiry", "high", "answer_alert"
    if mode in ("clarify",):
        return "other", "low", "reply"
    if mode in TEAM_MODES:
        # collect details only when the sheet's answer actually asks the customer for something
        handler = "answer_team" if (required and ASKS_FOR_INFO.search(answer)) else "reply"
    else:
        handler = "reply"          # knowledge, company_check
    cat, pri = "enquiry", "low"
    if csv_category == "Sales":
        cat, pri = ("purchase", "high") if mode in ("quote", "workflow", "sales_handoff", "quote_lookup") else ("purchase", "medium")
        if intent in ("invoice", "returns", "payment_methods", "warranty"):
            cat, pri = "enquiry", "medium" if intent != "payment_methods" else "low"
    elif csv_category == "Sizing":
        cat, pri = ("enquiry", "low") if mode == "knowledge" else ("purchase", "medium")
    elif csv_category == "Equipment":
        cat, pri = "enquiry", ("medium" if mode == "technical_review" else "low")
    elif csv_category == "Installation":
        cat, pri = ("enquiry", "medium") if intent in ("installation_time", "permits") else ("enquiry", "low")
        if risk == "high":
            cat, pri = "enquiry", "high"
    elif csv_category == "Support":
        if intent in ("panel_cleaning",):
            cat, pri = "maintenance", "low"
        elif intent == "status":
            cat, pri = "enquiry", "medium"
        else:
            cat, pri = "maintenance", "high"
    elif csv_category == "Privacy":
        cat, pri = "enquiry", "low"
    elif csv_category == "Language":
        cat, pri = ("purchase", "medium") if intent == "shona" else ("other", "low")
    return cat, pri, handler


def import_csv(path, existing=None):
    """Add the sheet's intents and phrases. Returns (new_intents, new_phrases)."""
    existing = db.list_intents() if existing is None else existing
    rows = list(csv.DictReader(Path(path).open(encoding="utf-8-sig", newline="")))
    by_intent = {}
    for r in rows:
        by_intent.setdefault(r["intent"].strip(), []).append(r)
    before = dict(existing)
    examples, created = [], 0
    for intent, group in by_intent.items():
        first = group[0]
        target = MERGE.get(intent)
        if target is None:
            target = f"qa_{intent}"
            if target not in existing:
                category, priority, handler = describe(first["category"], intent, first["response_mode"], first["risk"],
                                                         first["required_fields"], first["recommended_answer"])
                answer = " ".join(first["recommended_answer"].split())
                db.save_intent(target, category, priority, handler, [answer])
                existing[target] = {"name": target}
                created += 1
        elif target not in existing:
            log.warning("built-in intent %s is missing; skipping %s", target, intent)
            continue
        if intent not in MERGE and target in before and before[target].get("handler") == "answer_team":
            # earlier versions asked for details on answers that never ask for any; just answer instead
            category, priority, handler = describe(first["category"], intent, first["response_mode"], first["risk"],
                                                   first["required_fields"], first["recommended_answer"])
            if handler == "reply":
                old = before[target]
                db.save_intent(target, old["category"], old["priority"], "reply", old["replies"], old["enabled"])
        for r in group:
            q = " ".join(r["customer_question"].split())
            if q.lower().strip(" ?!.") not in SKIP_QUESTIONS:
                examples.append((target, q))
        examples += [(target, q) for q in EXTRA_PHRASES.get(intent, [])]
    added = db.add_examples_bulk(examples, "csv")
    log.info("QA sheet: %d new intents, %d new phrases", created, added)
    return created, added


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(import_csv(sys.argv[1] if len(sys.argv) > 1 else "training/solar_customer_qa.csv"))
