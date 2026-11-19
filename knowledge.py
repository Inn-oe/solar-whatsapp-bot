"""Technician knowledge: inverter fault codes by brand (exact lookup, so "F56" or "error 08" always match), plus short
Zimbabwe regulation facts. Nothing here is guessed: a code that is not in data/faults.json is reported as unknown
so the bot hands it to a technician instead of inventing a meaning."""
import json
import re
from pathlib import Path

_FILE = Path(__file__).parent / "data" / "faults.json"
_data = None

F_RE = re.compile(r"(?i)\bf\s?(\d{2})\b")


def data():
    global _data
    if _data is None:
        _data = json.loads(_FILE.read_text(encoding="utf-8"))
    return _data


def brand_of(text):
    """The fault-table key for a brand named in the text, or ''."""
    t = (text or "").lower()
    best = ""
    for key, tab in data()["tables"].items():
        for a in tab["aliases"]:
            if re.search(r"(?<![a-z])" + re.escape(a) + r"(?![a-z])", t):
                return key
    return best


def code_of(text):
    """A fault code written in the text ('F23', 'error 08', 'fault code 2') as (normalised, had_f_prefix), or None."""
    t = text or ""
    m = F_RE.search(t)
    if m:
        return "F" + m.group(1), True
    m = re.search(r"(?i)\b(?:e|err)\s?-?(\d{1,3})\b", t)
    if m:
        return m.group(1), False
    m = re.search(r"(?i)\b(?:fault|error|err|warning|code|alarm)\s*(?:code|no\.?|number)?\s*[:#-]?\s*(?:e\s?)?(\d{1,3})\b", t)
    if m:
        return m.group(1), False
    return None


def _norm(table_key, code):
    codes = data()["tables"][table_key]["codes"]
    for cand in (code, code.lstrip("F"), code.lstrip("0"), code.zfill(2), "F" + code.zfill(2)):
        if cand in codes:
            return cand
    return None


def lookup(text, brand_hint=""):
    """-> {"state": "found"|"unknown_code"|"need_brand", "brand", "code", ...}; None if the text has no code."""
    c = code_of(text)
    if not c:
        return None
    code, f_prefixed = c
    brand = brand_of(text) or brand_of(brand_hint)
    if not brand:
        if f_prefixed:
            brand = "deye"       # F-prefixed two-digit codes are the Deye / Sunsynk style
        else:
            return {"state": "need_brand", "code": code}
    tab = data()["tables"][brand]
    key = _norm(brand, code)
    if not key:
        return {"state": "unknown_code", "brand": brand, "code": code, "table": tab["name"]}
    meaning, severity, check = tab["codes"][key]
    return {"state": "found", "brand": brand, "code": key, "table": tab["name"], "meaning": meaning,
            "severity": severity, "check": check, "source": tab["source"]}


def explain(res):
    """Customer-facing wording for a lookup result, or '' when there is nothing safe to say."""
    if not res:
        return ""
    if res["state"] == "need_brand":
        return ("Error codes mean different things on different inverters. Which brand is it "
                "(Deye, Sunsynk, Growatt, Axpert/Mecer/Kodak, Victron or another)? It's printed on the front.")
    if res["state"] == "unknown_code":
        return (f"I don't have code {res['code']} for that inverter in my list, and I won't guess what it means with "
                "a fault. I'll pass it to a technician with the code.")
    sev = {"stop": "⚠️ This one needs a technician. Please don't keep resetting it.", "restart": "",
           "info": "", "check": ""}[res["severity"]]
    return (f"*{res['brand'].title() if res['brand'] != 'voltronic' else 'Axpert-type'} fault {res['code']}: "
            f"{res['meaning']}.*\n{res['check']}" + (f"\n\n{sev}" if sev else ""))


# ---------------------------------------------------------------- regulation facts (sources checked)
REGULATION_FACTS = [
    "Net metering in Zimbabwe: the Electricity (Net Metering) Regulations (SI 86 of 2018) were amended by SI 38 of "
    "2022, which raised the limit for net metering from 100 kW to 5 MW. Net metering is applied for through ZETDC.",
    "Licensing: under SI 125 and 126 of 2026, commercial and captive power plants up to 10 MW no longer need a full "
    "ZERA generation licence. Customers with a large system should confirm the exact requirement with ZERA.",
    "Grid-tied systems must be registered and use certified inverters; the installation must follow electrical "
    "safety standards (isolate AC and DC before work, earthing, surge protection).",
]
