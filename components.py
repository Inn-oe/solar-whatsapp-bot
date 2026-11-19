"""Live prices of individual parts (inverters, batteries, panels) from supplier shop pages.

Reads category pages politely (robots.txt obeyed, one request per page, same rules as market.py), keeps each
product's name and price, and summarises: inverter price by size, battery price per kWh, panel price per watt.
quote.build() uses a part's live price only when it is within the same sanity band as the system price; otherwise
the quotation keeps GieBee's own price and the difference is shown to staff on the Market page.
"""
import datetime as dt
import json
import re
import statistics
import time

import requests

import db
import market

SOURCES = [
    {"name": "Solar District inverters", "url": "https://solardistrict.co.zw/product-category/inverters/", "kind": "inverter"},
    {"name": "Solar District inverters p2", "url": "https://solardistrict.co.zw/product-category/inverters/page/2/", "kind": "inverter"},
    {"name": "Solar District batteries", "url": "https://solardistrict.co.zw/product-category/batteries/", "kind": "battery"},
    {"name": "Solar District panels", "url": "https://solardistrict.co.zw/product-category/solar-panels/", "kind": "panel"},
]
PRICE = re.compile(r"^\s*(?:US\s?\$|USD|\$)\s?(\d{1,3}(?:,\d{3})+|\d{2,6})(?:\.\d{1,2})?\b")
KW = re.compile(r"(?i)(\d{1,2}(?:\.\d)?)\s?-?\s?k(?:w|va)\b")
AH = re.compile(r"(?i)(\d{2,4})\s?ah\b")
VOLT = re.compile(r"(?i)\b(12|24|48|51\.2)\s?v\b")
KWH = re.compile(r"(?i)(\d{1,2}(?:\.\d{1,2})?)\s?kwh\b")
WATT = re.compile(r"(?i)\b(\d{3})\s?w(?:p|att|atts)?\b")
SANE = {"inverter": (150, 60000), "battery": (100, 30000), "panel": (40, 600)}


def spec(kind, name):
    """The size of a part from its name: kW, kWh or watts; None if the name does not say."""
    if kind == "inverter":
        m = KW.search(name)
        return float(m.group(1)) if m else None
    if kind == "battery":
        m = KWH.search(name)
        if m:
            return float(m.group(1))
        a, v = AH.search(name), VOLT.search(name)
        if a:
            return round(int(a.group(1)) * float(v.group(1) if v else 48) / 1000, 2)
        return None
    m = WATT.search(name)
    return float(m.group(1)) if m else None


def parse(text, kind):
    """Products found in a page's text: a name line followed (within a few lines) by a '$price' line."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    out, seen = [], set()
    for i, l in enumerate(lines):
        m = PRICE.match(l)
        if not m:
            continue
        price = float(m.group(1).replace(",", ""))
        for back in range(1, 4):                      # the product name is the nearest line above the price
            if i - back < 0:
                break
            name = lines[i - back]
            if PRICE.match(name) or len(name) < 6 or len(name) > 160:
                continue
            s = spec(kind, name)
            if s and SANE[kind][0] <= price <= SANE[kind][1] and (name, price) not in seen:
                seen.add((name, price))
                out.append({"name": name, "price": price, "spec": s, "kind": kind,
                            "excl_vat": "vat" in " ".join(lines[i:i + 2]).lower()})
            break
    return out


def refresh():
    """Read every part source now; keeps the old prices for a source that fails."""
    items, status = [], {}
    old = _load().get("items", [])
    for n, src in enumerate(SOURCES):
        if n:
            time.sleep(2)
        got, note = [], ""
        try:
            if not market.allowed(src["url"]):
                note = "skipped: robots.txt"
            else:
                r = requests.get(src["url"], headers={"User-Agent": market.USER_AGENT, "Accept": "text/html"}, timeout=25)
                if r.status_code != 200:
                    note = f"failed: {r.status_code}"
                else:
                    got = parse(market.text_of(r.text[:market.MAX_BYTES]), src["kind"])
                    note = f"ok: {len(got)} products"
        except Exception as exc:
            note = f"failed: {type(exc).__name__}"
        for g in got:
            g["source"] = src["name"]
        items += got or [o for o in old if o.get("source") == src["name"]]
        status[src["name"]] = note
    db.set_setting("component_prices", json.dumps({"items": items, "status": status,
                                                   "refreshed": dt.datetime.now(dt.timezone.utc).isoformat()}))
    return status


def _load():
    try:
        return json.loads(db.get_setting("component_prices") or "{}")
    except ValueError:
        return {}


def summary():
    """{'inverter': {size_kw: median price}, 'battery_per_kwh': x, 'panel_per_w': y, 'refreshed': iso, 'count': n}."""
    d = _load()
    items = d.get("items", [])
    inv = {}
    for it in items:
        if it["kind"] == "inverter":
            inv.setdefault(it["spec"], []).append(it["price"])
    bat = [it["price"] / it["spec"] for it in items if it["kind"] == "battery" and it["spec"]]
    pan = [it["price"] / it["spec"] for it in items if it["kind"] == "panel" and it["spec"]]
    med = lambda xs: round(statistics.median(xs), 2) if len(xs) >= 2 else None
    return {"inverter": {k: round(statistics.median(v)) for k, v in sorted(inv.items())},
            "battery_per_kwh": med(bat), "panel_per_w": med(pan), "refreshed": d.get("refreshed"),
            "count": len(items), "status": d.get("status", {})}


def live_parts(size_kva):
    """Live unit prices for a system of this size: {"inverter": USD, "battery_kwh": USD/kWh, "panel_w": USD/W, ...}."""
    s = summary()
    out = {}
    sizes = s["inverter"]
    if sizes:
        near = min(sizes, key=lambda k: abs(k - size_kva))
        if abs(near - size_kva) <= 0.6:
            out["inverter"] = sizes[near]
    if s["battery_per_kwh"]:
        out["battery_kwh"] = s["battery_per_kwh"]
    if s["panel_per_w"]:
        out["panel_w"] = s["panel_per_w"]
    if out and s["refreshed"]:
        try:
            out["when"] = dt.datetime.fromisoformat(s["refreshed"]).strftime("%d %b").lstrip("0")
        except ValueError:
            pass
    return out


def tier_inverter_prices(size_kva, brands_by_tier):
    """Median live supplier price of inverters of this size (within 1.5 kW) for each tier's brands: {tier: USD}."""
    out = {}
    items = [i for i in _load().get("items", []) if i["kind"] == "inverter" and abs(i["spec"] - size_kva) <= 1.5]
    for tier, brands in brands_by_tier.items():
        prices = [i["price"] / i["spec"] * size_kva for i in items if any(b in i["name"].lower() for b in brands)]
        if len(prices) >= 2:
            out[tier] = round(statistics.median(prices))
    return out
