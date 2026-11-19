"""System sizing worked step by step with the arithmetic done in code (never by the language model).

Peak sun hours are conservative design values for Zimbabwe (the country averages roughly 4.6-6.6 kWh/m2/day); a
technician confirms them for the exact site."""
import math

PSH = {"harare": 5.4, "chitungwiza": 5.4, "marondera": 5.3, "bulawayo": 5.5, "gweru": 5.5, "kwekwe": 5.5,
       "kadoma": 5.5, "mutare": 5.0, "masvingo": 5.6, "beitbridge": 5.8, "victoria falls": 5.5, "hwange": 5.5,
       "kariba": 5.7, "chinhoyi": 5.4, "bindura": 5.4, "rusape": 5.1}
DEFAULT_PSH = 5.2
DERATE = 0.80          # wiring, heat, dust and inverter losses
DOD = 0.80             # usable battery depth for lithium
NIGHT_SHARE = 0.60     # share of the day's energy used when the sun is down
PANEL_W = 550
BATTERY_V = 48


def psh(town=""):
    return PSH.get((town or "").strip().lower(), DEFAULT_PSH)


def size(daily_kwh, town=""):
    """-> dict with the numbers and a worked text. daily_kwh is the energy to cover each day."""
    daily_kwh = float(daily_kwh)
    h = psh(town)
    pv_w = daily_kwh * 1000 / (h * DERATE)
    panels = max(1, math.ceil(pv_w / PANEL_W))
    night_kwh = daily_kwh * NIGHT_SHARE
    batt_kwh = night_kwh / DOD
    batt_ah = batt_kwh * 1000 / BATTERY_V
    text = (f"📐 *How I worked it out* ({town or 'Zimbabwe'}, about {h:g} sun-hours a day)\n"
            f"• Solar: {daily_kwh:g} kWh ÷ ({h:g} h × {DERATE:g}) = {pv_w / 1000:.1f} kW, about {panels} × {PANEL_W} W panels\n"
            f"• Battery: {night_kwh:.1f} kWh used at night ({NIGHT_SHARE:.0%} of the day) ÷ {DOD:g} usable = "
            f"{batt_kwh:.1f} kWh, about {batt_ah:.0f} Ah at {BATTERY_V} V")
    return {"psh": h, "pv_w": round(pv_w), "panels": panels, "battery_kwh": round(batt_kwh, 1),
            "battery_ah": round(batt_ah), "text": text}
