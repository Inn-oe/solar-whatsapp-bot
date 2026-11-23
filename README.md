# Solar WhatsApp assistant

Chats with customers on WhatsApp (and a web chat) in plain language. The customer writes however they like, a
trained language model works out what they want, and the bot replies like a person would. Closed choices also come
with WhatsApp buttons or a list, but typing always works too. Everything it learns about a customer is stored for
the business platform.

## How it understands people

`nlu.py` is the language engine (open source, scikit-learn):

| Step | What it does |
|---|---|
| Clean-up | lower-case, strips accents, expands SMS shorthand (`pls`, `u`, `abt`), squeezes `pleaseee` |
| **Intent** | a classifier trained on example phrases decides what the message is (buy a system, a fault, a price, a thank-you...) |
| **Entities** | pulls out town, phone number, kWh, system size (kVA), budget, appliances and names |
| **Category + priority** | `purchase` / `enquiry` / `maintenance`, and `urgent` / `high` / `medium` / `low` |
| **Not sure?** | below the confidence threshold the bot asks a clarifying question once, then hands over to a person and puts the message in the *Review* queue so you can teach it |

Two interchangeable models behind the same interface:

* **`tfidf`** (default): TF-IDF word + character n-grams, logistic regression. Tiny, fast, runs on Render's free plan.
* **`transformer`**: a sentence-embedding transformer (`BAAI/bge-small-en-v1.5` through ONNX Runtime, no PyTorch)
  feeding the same classifier. Understands paraphrases and misspellings much better. Needs about 300 MB more RAM.
  Enable: set `NLU_BACKEND=transformer` (optionally `NLU_MODEL`). It loads in the background after start-up; the quick model answers meanwhile.
  If it cannot load, the bot falls back to `tfidf` automatically.

There is no model file. The model is rebuilt from the example phrases in the database at start-up and
whenever you press **Train now**, so the database is the only thing to keep.

## Training it (from the business platform)

WhatsApp ▸ **Train the bot**:

1. **What the bot understands**: every intent with its example phrases, replies, type and priority. Add phrases
   (one per line, English / Shona / slang), edit replies, or create a new intent.
2. **Review messages**: messages the bot was unsure about. Pick what each one meant and it becomes a new example.
3. **Test a message**: see how any text is read (intent, confidence, details found, priority reasons).
4. Press **Train now**. The panel shows the held-out accuracy (each phrase is predicted by a model that never saw it).

Replies may use `{first_comma}` (", Tendai"), `{company}`, `{phone}`, `{hours}`, `{promise}` ("within 1 hour" or
"when we open tomorrow at 08:00"). Put several versions in one reply, separated by a line with `---`, and the
bot picks one at random so it does not sound like a script.

The starting intents live in `nlu_seed.json` (and one intent per FAQ in `catalog.json`). Seeding only adds what
is missing, so your edits are never overwritten.

## Quotations learned from GieBee's own

`quote_book.json` holds, for every inverter size GieBee quotes often (1.5 to 20kVA), the parts of a typical
quotation with the median unit price and quantity from its 2026 quotations (300 quotations, about 2,000 line items,
read-only from the ERP database; only aggregated prices are kept, no customer data). `quote.py` turns a size into an
itemised quotation in GieBee's format: inverter, battery, panels, protection kit, mounting, cabling and labour, with
a total, plus the transport and deposit notes. The bot uses it for price questions ("marii 3kva"), the read-back
before sending, and the quotation it sends with the confirmation; the staff alert and the lead carry the same
figures. The catalog's package prices are the matching itemised totals. Refresh the book after new quotations:

```bash
ERP_DATABASE_URL=postgresql://... python training/build_quote_book.py     # or: ... build_quote_book.py items.csv
```

## Natural replies (language model)

The trained model (TF-IDF or the transformer) recognises what a message is about; the answers staff wrote in the
training sheet say what to tell the customer. With `LLM_API_KEY` set, the bot asks the language model to write each reply naturally in the context of the
conversation (`llm.natural_reply`), and to answer questions the trained
model does not recognise from the company's facts only: FAQ answers, packages and quotation prices
(`llm.answer_from_facts`). Every number in a reworded reply must appear in the approved text, otherwise the approved
text is sent unchanged; safety advice, technician handoffs and formal quotations are not reworded. The existing
conversation state still tracks quote details and troubleshooting outcomes; the language model makes the wording
flexible without taking unsafe actions or inventing company facts. The same approved reply is used when the key is
missing, rewording is disabled or the model is slow. Two separate switches: `LLM_REWORD` (default `on`) lets the model write conversational replies;
`LLM_ANSWERS` (default `on`) lets it answer unrecognised questions from the company facts. Questions answered this way still go to the Review queue so staff can teach the trained model.

## Buttons and typed answers

Buttons are shortcuts for closed choices; open details are always typed. Every button stands for words the bot
already understands, and typing a button's title counts as tapping it.

| Step | Question | Buttons / list | Typed |
|---|---|---|---|
| Greeting / unsure | How can I help? | Get a quotation · Fix a problem · Ask a question (or Talk to a person) | anything |
| Quote: what to power | What should it run? | list of packages with prices | appliances, kVA, budget |
| Quote: where | Town or suburb? | none | the address |
| Quote: confirm | Read-back with an instant estimate | ✅ Send request · ✏️ Change details | a correction |
| Quote again (3 days) | Another system, or change the last? | Another system · Change that one | |
| Fault: what's wrong | What is the system doing? | list of 8 common problems | a description |
| Fault: display | Error code, beeping or warning light? | No code or beeps · It's beeping · Screen is blank | the code |
| Fault: each fix | Did that fix it? | ✅ Fixed · ❌ Still the same · 👤 Talk to a person | what they see |
| Ask a question | Pick a topic | list of common topics | own question |
| Quote offers | Want a free quotation? | Yes, get a quote · No, thanks | yes / no |

WhatsApp limits are enforced in `interactive_payload` (3 buttons of 20 characters, or a list of 10 rows of 24);
a long message is sent as text first, with the question and its buttons after it. If WhatsApp refuses the
buttons, the reply goes out as text with the choices written in.

## Conversations

* **Buying**: asks what to power, then where, then confirms; skips anything already said ("5kVA with fridge and TV in Bulawayo" is complete in one message).
* **Maintenance**: faults and services, with safety advice first for smoke, fire, sparks or shocks (urgent).
* **Troubleshooting** (`troubleshoot.py`): for a plain fault the bot acts like a technician. It asks what the
  inverter shows (error code, beeps, warning light), then walks the customer through up to **two** safe fixes,
  asking after each whether it worked. Fixed: logged as handled, no staff alert. Still broken after both: it
  collects the town and hands over to the team as a priority, with the fixes already tried in the lead so nobody
  repeats them. Guides cover not charging, overload/tripping, fast-draining batteries, low production, geyser/pump
  not running, no power, and error codes or beeping, plus a generic one. Steps only use switches, breakers, the
  display and what can be seen from the ground, never opening equipment. Smoke, fire, sparks or shock skip the fixes
  and go straight to safety advice and an urgent alert; services and warranty claims go straight to a person.
  Add or change guides in `troubleshoot.py`.
* **Side questions** are answered without losing your place, then the bot returns to the question it was asking.
* **Not sure twice in a row**: hands over to the team and alerts staff.

## Environment variables

See the docstring of `app.py`. New ones: `NLU_BACKEND` (`tfidf`, `transformer` or `remote`), `NLU_MODEL`,
`NLU_REMOTE_URL`, `NLU_REMOTE_KEY`, `NLU_REMOTE_TIMEOUT` (seconds, default 4).

**Remote embeddings (optional).** `NLU_BACKEND=remote` sends the text to an embedding service you host yourself (`NLU_REMOTE_URL`, `NLU_REMOTE_KEY`); the quick model answers whenever it is unreachable. The service itself is not part of this repository.
**AI fallback (large language model).** When the quick model is unsure, `llm.py` asks an LLM to pick one of the
bot's intents (it never writes the reply itself). It works with any OpenAI-compatible API: set `LLM_API_KEY`, and
optionally `LLM_API_URL` (default Mistral, `https://api.mistral.ai/v1`; Groq `https://api.groq.com/openai/v1`;
OpenRouter `https://openrouter.ai/api/v1`), `LLM_MODEL` (default `mistral-small-latest`) and `LLM_TIMEOUT` (default 8 s).
Messages it classifies are queued on the Train page for staff to confirm, so the quick model learns them too.

**Using Claude.** Set `LLM_API_KEY` to an Anthropic key (it starts `sk-ant-`) and the bot uses Anthropic's own
Messages API with `claude-haiku-5-5` (change with `LLM_MODEL`, force with `LLM_PROVIDER=anthropic`). The long list
of intents goes in a cached system prompt, so repeat calls are cheaper and quicker. A wrong key, an unknown model or
no credit shows on the Train page ("AI fallback ... failed calls (last: HTTP 401 ...)") and the bot carries on with
the trained model. `LLM_REWORD=on` also lets the model reword the approved answers more naturally.
`INTERNAL_KEY` also protects `/internal/nlu/train|test|status`, which the platform calls.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```


## PDF quotations and live prices

When a customer finishes a quotation request, the bot builds the itemised quotation (`quote.py`) and saves it
(`app.save_quote`). On WhatsApp it uploads the PDF and sends it as a document, followed by a short message with the
total; if WhatsApp refuses the document, the full itemised text is sent instead. On the website chat the same quotation
appears as a "Download quotation (PDF)" link (`/quote/<secret token>.pdf`). The PDF has the company header, reference,
date, 14-day validity, the itemised table, total and terms.

Prices follow the live Zimbabwe market: the Market page's adverts (`market.py`, refreshed daily from the listed seller
pages) give the typical installed price for the size, and every line of the quotation is scaled by the same factor so
the total follows the market. A market price outside 0.6-1.6x of GieBee's own price, or for a different size, is
treated as bad data and GieBee's quotation prices are used. The quotation says which basis it used.

## Technician knowledge

* `data/faults.json`: fault codes for Deye/Sunsynk, Axpert-family (Axpert, Mecer, Kodak, Conversol, Must...), Growatt
  SPF and Victron, each with the manufacturer's meaning and a customer-safe first check. `knowledge.py` matches the
  code exactly, asks for the brand when a bare number is ambiguous, sends serious faults (insulation, leakage,
  internal failures) straight to a technician, and never guesses a code that is not in the table.
* `sizing.py`: solar, battery and inverter sizing with the arithmetic done in code and Zimbabwe peak sun hours by town.
* ZERA / net-metering facts (SI 86 of 2018, SI 38 of 2022, SI 125/126 of 2026) are in `knowledge.REGULATION_FACTS`;
  verify them against ZERA before relying on them for a customer's compliance.

## Part prices, test quote and site visits

* `components.py` reads inverter, battery and panel prices from supplier shop pages once a day (Solar District to
  start; edit `SOURCES`). A part's live price is used in a quotation only when it is within 0.6-1.6x of GieBee's own
  price; the comparison is kept on the quotation (`q["compare"]`) so staff can see where the market differs.
* Staff numbers (`ALERT_NUMBERS`) can send **test quote** on WhatsApp to receive a sample quotation PDF with buttons,
  to check delivery on a real phone.
* After a quotation the customer can tap **Book site visit**, say when suits them, and the team gets an alert to
  confirm the day and time.

## Quote tiers (budget to premium)

`data/tiers.json` defines three options per system size. **Budget** is GieBee's own quotation price (from the 300
quotes: budget hybrid inverters of the Must type and generic LiFePO4 lithium batteries). **Standard** (Growatt /
LuxPower class) and **Premium** (Sunsynk / Deye / Victron) use supplier list prices per kW, replaced by the live
supplier price when `components.py` has one for that size, and a branded-battery factor taken from GieBee's own Deye
battery price. The bot shows the options on every quotation, explains what each buys (quality, support, typical
warranty, lithium vs lead-acid), and sends the full PDF for whichever the customer picks. The PDF lists all options.
Update the numbers in `tiers.json` when supplier prices move.

## Training on your own PC (when the server is too small to train)

Render's free 512 MB server runs out of memory while *training* (the bot is killed and the ERP shows "bot not
reachable"), but *loading* a trained model needs very little. So train on your PC and ship the model as a file:

1. Close heavy programs (training needs ~250 MB). Optional: for the phrases staff added on the Train page, copy the
   **External Database URL** (Render `giebee-db` ▸ Connections) and set it before starting Jupyter:
   `$env:DATABASE_URL = "postgresql://..."` (it is only read, never changed). Without it the notebook trains on the
   starter phrases (seed + Q&A sheet).
2. `pip install jupyterlab` in `.venv`, start `jupyter lab` from this folder, open **`train_model_enhanced.ipynb`**
   and choose *Run All*. It loads the phrases, adds your terminology (glossary, reviewed CSV phrases), measures held-out
   accuracy, tries several settings, trains, and writes `models/nlu_model.pkl` (the old file is copied aside first).
3. `git add models/nlu_model.pkl models/nlu_train_report.json; git commit -m "Retrained model"; git push` — Render
   deploys and the bot loads the file at start-up instead of training. The ERP Train page tells you if staff changed
   phrases after the model was trained.

While the file is present, **Train now** only reloads it (it cannot train on the server). After upgrading to a
server with enough memory, set `NLU_PRETRAINED=off` on the bot (or delete the file) and Train now works again.
`requirements.txt` pins scikit-learn/numpy/scipy so the file and the server agree; a model from another
scikit-learn version is refused and the bot trains normally instead.
