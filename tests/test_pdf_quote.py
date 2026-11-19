"""PDF quotations (WhatsApp document + web link) and live market pricing."""
import quote
import app as bot


def _flow_to_quote(chat):
    chat.say("I want a 5kVA solar system")
    for _ in range(6):
        low = chat.last["text"].lower()
        if "town" in low or "suburb" in low:
            chat.say("Borrowdale, Harare")
        elif "name" in low and "?" in low:
            chat.say("Tendai Moyo")
        elif "send it to our team" in low or "shall i send" in low:
            chat.tap("send")
        elif chat.last.get("document"):
            break
        else:
            break
    return chat.last


def test_finished_quote_carries_a_pdf_that_opens(chat, client):
    r = _flow_to_quote(chat)
    doc = r.get("document")
    assert doc and doc["token"], r["text"]
    res = client.get(f"/quote/{doc['token']}.pdf")
    assert res.status_code == 200 and res.mimetype == "application/pdf" and res.data.startswith(b"%PDF")
    assert client.get("/quote/not-a-real-token.pdf").status_code == 404


def test_web_chat_returns_the_pdf_link(client):
    sid, r = "pdftestsession1", {"reply": ""}
    text = "I want a 5kVA solar system"
    for _ in range(8):
        r = client.post("/api/chat", json={"session": sid, "text": text}).get_json()
        if r.get("document"):
            break
        low = r["reply"].lower()
        keys = [o["id"] for o in r["options"]]
        if "btn:send" in keys:
            text = "btn:send"
        elif "phone number" in low:
            text = "0771234567"
        elif "name" in low and "?" in low:
            text = "Tendai Moyo"
        else:
            text = "Harare"
    assert r.get("document") and r["document"]["url"].endswith(".pdf")


def test_whatsapp_gets_the_pdf_then_a_short_message(chat, monkeypatch):
    sent = []
    monkeypatch.setattr(bot, "WHATSAPP_TOKEN", "x")
    monkeypatch.setattr(bot, "PHONE_NUMBER_ID", "1")
    monkeypatch.setattr(bot, "send_document", lambda to, pdf, name, caption="": sent.append(("pdf", name, len(pdf))) or True)
    monkeypatch.setattr(bot, "send_text", lambda to, body: sent.append(("text", body)) or True)
    monkeypatch.setattr(bot, "send_interactive", lambda to, i: sent.append(("buttons", i["body"]["text"])) or True)
    r = _flow_to_quote(chat)
    bot.send_reply(chat.who, r)
    assert sent[0][0] == "pdf" and sent[0][1].endswith(".pdf") and sent[0][2] > 1000
    assert "in the PDF above" in sent[1][1]


def test_whatsapp_falls_back_to_the_text_quote_if_the_pdf_cannot_be_sent(chat, monkeypatch):
    sent = []
    monkeypatch.setattr(bot, "WHATSAPP_TOKEN", "x")
    monkeypatch.setattr(bot, "PHONE_NUMBER_ID", "1")
    monkeypatch.setattr(bot, "send_document", lambda *a, **k: False)
    monkeypatch.setattr(bot, "send_text", lambda to, body: sent.append(body) or True)
    monkeypatch.setattr(bot, "send_interactive", lambda to, i: sent.append(i["body"]["text"]) or True)
    bot.send_reply(chat.who, _flow_to_quote(chat))
    assert "QUOTATION" in " ".join(sent) and "Total" in " ".join(sent)


def test_live_market_price_moves_the_quote_and_bad_data_is_ignored():
    book = quote.build(5)["total"]
    live = quote.build(5, {"median": 2200, "size": 5, "count": 9, "sources": 4, "when": "19 Nov"})
    assert abs(live["total"] - 2200) < 40 and "market" in live["basis"]
    junk = quote.build(5, {"median": 400, "size": 5, "count": 9, "sources": 4, "when": "19 Nov"})
    assert junk["total"] == book and "GieBee" in junk["basis"]
    assert quote.build(5, {"median": 2200, "size": 10, "count": 9, "sources": 4})["total"] == book   # other size


def test_pdf_has_the_items_and_total():
    q = quote.build(5)
    pdf = quote.pdf_bytes(q, "GieBee Engineering", {"phone": "+263"}, "Tendai", "Harare", "WA-1")
    assert pdf.startswith(b"%PDF") and len(pdf) > 2000


def test_part_prices_are_read_and_used_only_when_close_to_ours():
    import components
    page = ("Growatt SPF 5000 ES 5KW 48V Inverter\n$897 Excl. VAT\nSunsynk Sun 5kW Hybrid Inverter\n$1,778 Excl. VAT\n"
            "Deye 12kW Three Phase Inverter\nAdd to quote\n")
    items = components.parse(page, "inverter")
    assert [i["price"] for i in items] == [897.0, 1778.0] and all(i["spec"] == 5.0 for i in items)
    book = quote.build(5)
    far = quote.build(5, parts={"inverter": 897})                 # 2.8x our price: not used, but compared
    assert far["total"] == book["total"] and far["compare"][0]["market"] == 897
    near = quote.build(5, parts={"inverter": 380, "when": "19 Nov"})
    assert near["total"] != book["total"] and "live supplier prices" in near["basis"]


def test_site_visit_can_be_booked_after_a_quote(chat):
    r = _flow_to_quote(chat)
    assert "btn:visit" in [o["id"] for o in r["options"]]
    assert "what day" in chat.tap("visit").lower()
    done = chat.say("Tuesday morning")
    assert "Tuesday morning" in done and "technician visit" in done


def test_staff_test_quote_has_pdf_and_buttons():
    r = bot.test_quote_reply()
    assert r["document"]["token"] and len(r["options"]) == 3
    pdf, name = bot.quote_pdf(r["document"]["token"])
    assert pdf.startswith(b"%PDF")


def test_tiers_run_from_budget_to_premium_with_real_differences():
    opts = quote.tiers(5)
    totals = [o["total"] for o in opts]
    assert [o["key"] for o in opts] == ["economy", "standard", "premium"] and totals == sorted(totals)
    assert totals[0] == quote.build(5)["total"]                      # budget is GieBee's own quotation price
    prem = quote.build(5, tier="premium")
    assert "Sunsynk" in prem["lines"][0]["item"] and "branded lithium" in prem["lines"][1]["item"]
    assert quote.build(5, tier="standard", tier_inverters={"standard": 1000})["lines"][0]["unit"] == 1000


def test_customer_can_compare_options_and_get_a_premium_pdf(chat):
    r = _flow_to_quote(chat)
    assert "Compare options" in r["text"] or "btn:tiers" in [o["id"] for o in r["options"]]
    text = chat.tap("tiers")
    assert "Budget" in text and "Standard" in text and "Premium" in text and "cheapest to best quality" in text
    chat.tap("tier_premium")
    doc = chat.last["document"]
    assert doc and "Premium" in doc["title"]
    pdf, _ = bot.quote_pdf(doc["token"])
    assert pdf.startswith(b"%PDF")


def test_typed_question_about_cheaper_or_better_inverters_gets_the_options(chat):
    reply = chat.say("what is the difference between cheap and premium inverters, and lead acid batteries?")
    assert "Budget" in reply and "Premium" in reply and "lithium" in reply.lower()
