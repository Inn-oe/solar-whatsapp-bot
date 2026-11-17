# Solar WhatsApp assistant

Chats with customers on WhatsApp (and a web chat) in plain language. No buttons or menus: the customer
writes however they like, a trained language model works out what they want, and the bot replies like a
person would. Everything it learns about a customer is stored for the business platform.

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

## Conversations

* **Buying**: asks what to power, then where, then confirms; skips anything already said ("5kVA with fridge and TV in Bulawayo" is complete in one message).
* **Maintenance**: faults and services, with safety advice first for smoke, fire, sparks or shocks (urgent).
* **Side questions** are answered without losing your place, then the bot returns to the question it was asking.
* **Not sure twice in a row**: hands over to the team and alerts staff.

## Environment variables

See the docstring of `app.py`. New ones: `NLU_BACKEND` (`tfidf` or `transformer`), `NLU_MODEL`.
`INTERNAL_KEY` also protects `/internal/nlu/train|test|status`, which the platform calls.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```
