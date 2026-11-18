"""Rebuild quote_book.json from GieBee's quotation line items (read-only).

Usage:
  ERP_DATABASE_URL=postgresql://... python training/build_quote_book.py          # from the ERP database
  python training/build_quote_book.py items.csv                                  # from an export with columns
                                                                                 # quotation_id,description,quantity,unit_price

Each quotation with exactly one hybrid inverter and some panels is a solar system quote. Lines are sorted into
inverter, battery, panels, protection kit, mounting, cabling, combiner, labour and transport; for every inverter
size seen in at least MIN_QUOTES quotations the book keeps the median unit price and typical quantity of each part
that appears in at least half of them. No customer data is read or written.
"""
import csv
import json
import os
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

MIN_QUOTES = 2
OUT = Path(__file__).resolve().parent.parent / "quote_book.json"
ORDER = ["inverter", "battery", "panel", "protection", "mounting", "cabling", "combiner", "labour"]


def classify(d):
    d = re.sub(r"\s+", " ", (d or "").lower().strip())
    if "inverter" in d and "air" not in d:
        return "inverter"
    for cls, rx in (("battery", r"batter"), ("panel", r"\bpanels?\b(?!.*mount)"), ("protection", r"protection"),
                    ("other", r"tank"), ("mounting", r"mount|structure|solar stand"), ("combiner", r"combiner"),
                    ("labour", r"labour|installation"), ("transport", r"transport"),
                    ("cabling", r"cabl|wiring|accessor")):
        if re.search(rx, d):
            return cls
    return "other"


def kva(d):
    m = re.search(r"(\d+(?:\.\d+)?)\s*k(?:va|w)", (d or "").lower())
    return float(m.group(1)) if m else None


def tidy(d):
    d = re.sub(r"\s+", " ", (d or "").strip())
    d = re.sub(r"(?i)(\d+)\s*watts?", r"\1W", d)
    d = re.sub(r"(?i)(\d+(?:\.\d+)?)\s*kva", r"\1kVA", d)
    d = re.sub(r"(?i)(\d+(?:\.\d+)?)\s*kwh", r"\1kWh", d)
    d = re.sub(r"(?i)(\d+(?:\.\d+)?)\s*v\b", r"\1V", d)
    d = re.sub(r"(?i)(\d+)\s*ah\b", r"\1Ah", d)
    return d[:1].upper() + d[1:]


def build(rows):
    """rows: iterable of (quotation_id, description, quantity, unit_price) -> book dict."""
    by_quote = defaultdict(list)
    for qid, desc, qty, price in rows:
        if price is None or qty is None:
            continue
        by_quote[qid].append((classify(desc), desc, float(qty), float(price)))
    sized = defaultdict(list)
    for lines in by_quote.values():
        inv = [l for l in lines if l[0] == "inverter"]
        if len(inv) == 1 and any(l[0] == "panel" for l in lines) and kva(inv[0][1]):
            sized[kva(inv[0][1])].append(lines)
    systems = []
    for size, quotes in sorted(sized.items()):
        if len(quotes) < MIN_QUOTES:
            continue
        parts, transport = [], []
        for cls in ORDER:
            found = [[l for l in q if l[0] == cls] for q in quotes]
            found = [f for f in found if f]
            if len(found) * 2 < len(quotes):
                continue
            descs = Counter(tidy(f[0][1]) for f in found)
            parts.append({"item": descs.most_common(1)[0][0],
                          "qty": round(statistics.median(f[0][2] for f in found)),
                          "unit": round(statistics.median(f[0][3] for f in found), 2)})
        for q in quotes:
            transport += [l[3] for l in q if l[0] == "transport"]
        systems.append({"size_kva": size, "quotes": len(quotes), "lines": parts,
                        "transport": round(statistics.median(transport)) if transport else 80})
    return systems


def main():
    if len(sys.argv) > 1:
        with open(sys.argv[1], newline="", encoding="utf-8-sig") as f:
            rows = [(r["quotation_id"], r["description"], r["quantity"], r["unit_price"]) for r in csv.DictReader(f)]
    else:
        import psycopg2
        with psycopg2.connect(os.environ["ERP_DATABASE_URL"]) as con, con.cursor() as cur:
            cur.execute("set transaction read only")
            cur.execute("select quotation_id, description, quantity, unit_price from quotation_items")
            rows = cur.fetchall()
    old = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    old["systems"] = build(rows)
    OUT.write_text(json.dumps(old, indent=1), encoding="utf-8")
    print(f"{len(old['systems'])} system sizes written to {OUT}")


if __name__ == "__main__":
    main()
