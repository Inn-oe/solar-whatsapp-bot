"""Itemised quotations in GieBee Engineering's own format, learned from its past quotations (quote_book.json).

A quotation lists the inverter, battery, panels, protection kit, mounting, cabling and labour, each with a
quantity and unit price taken from the median of GieBee's real quotations for that system size, and a total.
Transport is added after the site visit. Rebuild the price book with training/build_quote_book.py.
"""
import datetime as dt
import json
import re
from pathlib import Path

BOOK_FILE = Path(__file__).parent / "quote_book.json"
_book = None


def book():
    global _book
    if _book is None:
        _book = json.loads(BOOK_FILE.read_text(encoding="utf-8"))
    return _book


def sizes():
    return [s["size_kva"] for s in book()["systems"]]


def template(size_kva):
    """The learned system closest to the size asked for (on a tie, the larger one)."""
    systems = book()["systems"]
    return min(systems, key=lambda s: (abs(s["size_kva"] - size_kva), -s["size_kva"]))


def build(size_kva):
    """-> {"size_kva", "lines": [{item, qty, unit, amount}], "total", "quotes", "transport"} or None."""
    if not size_kva:
        return None
    t = template(float(size_kva))
    lines = [dict(l, amount=round(l["qty"] * l["unit"], 2)) for l in t["lines"]]
    return {"size_kva": t["size_kva"], "asked_kva": float(size_kva), "lines": lines,
            "total": round(sum(l["amount"] for l in lines), 2), "quotes": t["quotes"], "transport": t["transport"]}


def money(x):
    return f"{book()['currency']} {x:,.2f}".replace(".00", "")


def reference(who, when=None):
    when = when or dt.datetime.now(dt.timezone.utc)
    digits = re.sub(r"\D", "", who or "")[-4:] or "0000"
    return f"WA-{when:%y%m%d}-{digits}"


def as_text(q, company, client="", where="", ref=""):
    """The quotation as a WhatsApp message (bold with *, one line per item)."""
    head = [f"🧾 *QUOTATION* · {company}"]
    meta = [x for x in (f"Ref {ref}" if ref else "", dt.date.today().strftime("%d %b %Y")) if x]
    head.append(" · ".join(meta))
    who = ", ".join(x for x in (client, where) if x)
    if who:
        head.append(f"For: {who}")
    rows = [f"{i}. {l['item']}: {l['qty']:g} × {money(l['unit'])} = *{money(l['amount'])}*"
            for i, l in enumerate(q["lines"], 1)]
    note = ""
    if abs(q["asked_kva"] - q["size_kva"]) >= 0.25:
        note = f"\n(You asked for {q['asked_kva']:g}kVA; our closest standard system is {q['size_kva']:g}kVA.)"
    b = book()
    return ("\n".join(head) + "\n\n" + f"*{q['size_kva']:g}kVA solar system, installed*\n" + "\n".join(rows)
            + f"\n\n*Total: {money(q['total'])}*{note}\n\n🚚 {b['transport_note']}\n💳 {b['payment_note']}\n"
            "Prices are from our recent quotations; a technician confirms the system and final price on a free "
            "site visit.")


def as_summary(q):
    """One line for staff alerts and the lead record."""
    return (f"{q['size_kva']:g}kVA quote {money(q['total'])}: "
            + "; ".join(f"{l['qty']:g}x {l['item']}" for l in q["lines"]))
