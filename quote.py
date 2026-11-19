"""Itemised quotations in GieBee Engineering's own format, learned from its past quotations (quote_book.json).

A quotation lists the inverter, battery, panels, protection kit, mounting, cabling and labour, each with a
quantity and unit price taken from the median of GieBee's real quotations for that system size, and a total.
Transport is added after the site visit. Rebuild the price book with training/build_quote_book.py.
"""
import datetime as dt
import hashlib
import io
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


LIVE_BAND = (0.6, 1.6)   # a live market price outside this band of the book price is ignored as bad data


def _step(x):
    return round(x / 5) * 5 if x >= 50 else round(x * 2) / 2


def _live_unit(item, parts):
    """The live price for one quotation line from the parts feed, or None when the feed has nothing for it."""
    low = item.lower()
    if "inverter" in low and parts.get("inverter"):
        return parts["inverter"]
    m = re.search(r"(\d{2})\s?v\s*(\d{2,4})\s?ah", low)
    if "batter" in low and m and parts.get("battery_kwh"):
        return parts["battery_kwh"] * int(m.group(1)) * int(m.group(2)) / 1000
    m = re.search(r"(\d{3})\s?w", low)
    if "panel" in low and m and parts.get("panel_w"):
        return parts["panel_w"] * int(m.group(1))
    return None


def _apply_parts(lines, parts, basis):
    """Use a part's live price when it is within the sanity band of our own; always record the comparison."""
    out, used, compare = [], [], []
    for l in lines:
        live = _live_unit(l["item"], parts)
        if live is None:
            out.append(l)
            continue
        compare.append({"item": l["item"], "ours": l["unit"], "market": round(live, 2)})
        if LIVE_BAND[0] <= live / l["unit"] <= LIVE_BAND[1]:
            u = _step(live)
            out.append(dict(l, unit=u, amount=round(l["qty"] * u, 2)))
            used.append(l["item"])
        else:
            out.append(l)
    if used:
        basis += f"; live supplier prices (checked {parts.get('when', 'recently')}) for " + ", ".join(used)
    return out, basis, compare


def build(size_kva, live=None, parts=None):
    """-> {"size_kva", "lines": [{item, qty, unit, amount}], "total", "quotes", "transport", "basis"} or None.

    `live` is the current advertised market price for this size, {"median", "count", "sources", "when"}. Every
    line is scaled by the same factor so the total follows the market; without it (or when it looks wrong) the
    prices are GieBee's own from its past quotations."""
    if not size_kva:
        return None
    t = template(float(size_kva))
    base = [dict(l, amount=round(l["qty"] * l["unit"], 2)) for l in t["lines"]]
    base_total = sum(l["amount"] for l in base)
    lines, basis = base, "GieBee's recent quotations"
    if live and live.get("median") and abs(live.get("size", t["size_kva"]) - t["size_kva"]) < 0.25:
        f = live["median"] / base_total
        if LIVE_BAND[0] <= f <= LIVE_BAND[1]:
            lines = []
            for l in t["lines"]:
                u = _step(l["unit"] * f)
                lines.append(dict(l, unit=u, amount=round(l["qty"] * u, 2)))
            basis = (f"current Zimbabwe market prices ({live.get('count', '')} adverts at {live.get('sources', '')} "
                     f"sellers, checked {live.get('when', 'recently')})")
    compare = []
    if parts:
        lines, basis, compare = _apply_parts(lines, parts, basis)
    return {"size_kva": t["size_kva"], "asked_kva": float(size_kva), "lines": lines,
            "total": round(sum(l["amount"] for l in lines), 2), "quotes": t["quotes"], "transport": t["transport"],
            "basis": basis, "compare": compare}


def money(x):
    return f"{book()['currency']} {x:,.2f}".replace(".00", "")


def reference(who, when=None):
    when = when or dt.datetime.now(dt.timezone.utc)
    digits = re.sub(r"\D", "", who or "")[-4:] or hashlib.sha1((who or "").encode()).hexdigest()[:4]
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
            f"Prices follow {q.get('basis', 'our recent quotations')}; a technician confirms the system and final "
            "price on a free site visit.")


def as_summary(q):
    """One line for staff alerts and the lead record."""
    return (f"{q['size_kva']:g}kVA quote {money(q['total'])}: "
            + "; ".join(f"{l['qty']:g}x {l['item']}" for l in q["lines"]))


def pdf_bytes(q, company, contact=None, client="", where="", ref=""):
    """The quotation as a PDF (bytes), laid out like a company quotation: header, itemised table, total, terms."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    contact = contact or {}
    b = book()
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm,
                            bottomMargin=16 * mm, title=f"Quotation {ref}".strip(), author=company)
    ss = getSampleStyleSheet()
    small = ParagraphStyle("s", parent=ss["Normal"], fontSize=9, leading=12, textColor=colors.HexColor("#444444"))
    body = ParagraphStyle("b", parent=ss["Normal"], fontSize=10, leading=13)
    h1 = ParagraphStyle("h", parent=ss["Title"], fontSize=20, alignment=0, spaceAfter=2, textColor=colors.HexColor("#0b3d2e"))
    today = dt.date.today()
    valid = today + dt.timedelta(days=14)
    esc = lambda s: (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    flow = [Paragraph(esc(company), h1),
            Paragraph(" · ".join(esc(x) for x in (contact.get("address"), contact.get("phone"), contact.get("email")) if x), small),
            Spacer(1, 6 * mm), Paragraph("<b>QUOTATION</b>", ParagraphStyle("q", parent=body, fontSize=14))]
    meta = [["Reference", ref or "-"], ["Date", today.strftime("%d %b %Y")], ["Valid until", valid.strftime("%d %b %Y")]]
    if client:
        meta.append(["Prepared for", client])
    if where:
        meta.append(["Site", where])
    mt = Table([[Paragraph(esc(a), small), Paragraph(esc(c), body)] for a, c in meta], colWidths=[32 * mm, 120 * mm])
    mt.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("BOTTOMPADDING", (0, 0), (-1, -1), 1)]))
    flow += [Spacer(1, 3 * mm), mt, Spacer(1, 6 * mm),
             Paragraph(f"<b>{q['size_kva']:g} kVA solar system, supplied and installed</b>", body), Spacer(1, 2 * mm)]
    rows = [["#", "Item", "Qty", f"Unit ({b['currency']})", f"Amount ({b['currency']})"]]
    for i, l in enumerate(q["lines"], 1):
        rows.append([str(i), Paragraph(esc(l["item"]), body), f"{l['qty']:g}", f"{l['unit']:,.2f}", f"{l['amount']:,.2f}"])
    rows.append(["", "", "", Paragraph("<b>TOTAL</b>", body), Paragraph(f"<b>{q['total']:,.2f}</b>", body)])
    t = Table(rows, colWidths=[9 * mm, 80 * mm, 15 * mm, 30 * mm, 33 * mm], repeatRows=1)
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0b3d2e")),
                           ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("FONTSIZE", (0, 0), (-1, -1), 9),
                           ("ALIGN", (2, 0), (-1, -1), "RIGHT"), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                           ("LINEBELOW", (0, 1), (-1, -2), 0.25, colors.HexColor("#cccccc")),
                           ("LINEABOVE", (0, -1), (-1, -1), 1, colors.HexColor("#0b3d2e")),
                           ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]))
    notes = [b["transport_note"], b["payment_note"],
             f"Prices follow {q.get('basis', 'our recent quotations')}.",
             "A technician confirms the system size and final price on a free site visit. This quotation is valid for 14 days."]
    if abs(q["asked_kva"] - q["size_kva"]) >= 0.25:
        notes.insert(0, f"You asked for {q['asked_kva']:g} kVA; our closest standard system is {q['size_kva']:g} kVA.")
    flow += [t, Spacer(1, 6 * mm), Paragraph("<b>Terms and notes</b>", body)] + [Paragraph("• " + esc(n), small) for n in notes]
    doc.build(flow)
    return buf.getvalue()
