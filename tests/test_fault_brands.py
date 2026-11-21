"""Fault tables for Victron (full list), Luxpower (E/W codes), GoodWe and Solis (named alarms). Every entry was copied
from the manufacturer's manual; anything not in a table must be handed to a technician, never guessed."""
import json
import re
from pathlib import Path

import knowledge

UNSAFE = re.compile(r"\b(unscrew|remove the (?:cover|lid)|open the (?:inverter|battery|cabinet|box)|touch the wiring|"
                    r"bypass|jump)\b", re.I)


def test_victron_full_list():
    r = knowledge.lookup("victron error 14")
    assert r["state"] == "found" and r["severity"] == "stop" and "cables" in r["meaning"] + r["check"]
    assert knowledge.lookup("victron multiplus error 17")["meaning"].startswith("Phase master")
    assert knowledge.lookup("victron error 9")["state"] == "unknown_code"      # not on Victron's page: not guessed


def test_luxpower_error_and_warning_letters_are_not_mixed_up():
    e = knowledge.lookup("luxpower E012")
    w = knowledge.lookup("luxpower W012")
    assert e["meaning"] == "UPS output short circuit" and w["meaning"] == "Battery on-MOS warning"
    assert knowledge.lookup("my sna inverter shows w015")["severity"] == "stop"      # battery in reverse
    assert knowledge.lookup("luxpower e999")["state"] == "unknown_code"
    assert knowledge.lookup("luxpower error 26")["state"] == "unknown_code"           # no letter: could be E026 or W026


def test_named_alarms_need_the_brand():
    assert knowledge.lookup("my goodwe says Utility Loss")["code"] == "Utility Loss"
    assert knowledge.lookup("Utility Loss") is None                    # no brand, no code: nothing to look up
    r = knowledge.lookup("solis inverter showing NO-Battery")
    assert r["state"] == "found" and r["code"] == "NO-Battery"
    assert knowledge.lookup("solis alarm 1055")["code"] == "NO-Battery"
    assert knowledge.lookup("solis OV-Vbatt-H")["code"] == "OV-Vbatt-H"   # longest name wins over OV-Vbatt


def test_arc_and_leakage_faults_go_to_a_technician():
    for text in ("solis ARC-FAULT", "solis ILeak-PRO 01", "goodwe Ground I Failure", "goodwe Isolation Failure"):
        assert knowledge.lookup(text)["severity"] == "stop", text


def test_explain_names_the_brand():
    assert knowledge.explain(knowledge.lookup("goodwe Utility Loss")).startswith("*GoodWe fault Utility Loss")
    assert "Felicity, Luxpower, GoodWe, Solis" in knowledge.explain({"state": "need_brand", "code": "5"})


def test_tables_are_complete_and_safe():
    tables = json.loads((Path(knowledge.__file__).parent / "data" / "faults.json").read_text(encoding="utf-8"))["tables"]
    for key, tab in tables.items():
        assert tab["source"].startswith("http"), key
        rows = list(tab["codes"].values()) + list((tab.get("names") or {}).values())
        assert rows, key
        for meaning, severity, check in rows:
            assert severity in ("stop", "check", "restart", "info"), (key, meaning)
            assert meaning and check and not UNSAFE.search(check), (key, meaning)
        for alt, name in (tab.get("name_codes") or {}).items():
            assert name in tab["names"], (key, alt)


def test_felicity_deye_growatt_from_the_manuals():
    f = knowledge.lookup("felicity inverter fault 71")
    assert f["state"] == "found" and f["severity"] == "stop" and "leakage" in f["meaning"].lower()
    assert knowledge.lookup("my ivgm shows error 77")["severity"] == "stop"            # PV insulation
    assert knowledge.lookup("felicity error 60")["state"] == "unknown_code"                  # not in the manual: not guessed
    assert knowledge.lookup("deye F63")["severity"] == "stop"                          # arc fault
    assert knowledge.lookup("deye F22")["code"] == "F22"
    assert knowledge.lookup("deye F13")["meaning"].startswith("Working mode")          # existing entries untouched
    assert knowledge.lookup("growatt spf 3000t fault 20")["meaning"] == "BMS communication error"
    assert knowledge.lookup("growatt error 13")["severity"] == "stop"
