"""Train the bot's language model on this computer and save it as a file the hosted bot loads.

Why: Render's free 512 MB server runs out of memory while training (the bot is killed and customers get nothing),
but loading a trained model needs very little. So train here, commit the file, deploy.

    # 1. phrases from the LIVE database (includes everything staff added on the Train page):
    #    Render dashboard -> giebee-db -> Connections -> "External Database URL"
    set DATABASE_URL=postgresql://...            (PowerShell:  $env:DATABASE_URL = "postgresql://...")
    .venv\\Scripts\\python train_local.py

    # 2. or, with no DATABASE_URL, the starter phrases only (seed + the Q&A sheet), same as a fresh install
    .venv\\Scripts\\python train_local.py

    # 3. publish it
    git add models/nlu_model.pkl && git commit -m "Retrained model" && git push

The file is models/nlu_model.pkl. The bot checks that it was made with the same scikit-learn version (pinned in
requirements.txt) and tells the ERP Train page when the phrases changed after it was trained.
Options: --no-check skips the accuracy measurement (a few minutes on a slow PC); --out PATH changes the file.
"""
import argparse
import json
import logging
import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(HERE / "models" / "nlu_model.pkl"))
    ap.add_argument("--no-check", action="store_true", help="skip the accuracy measurement")
    args = ap.parse_args()

    live = bool(os.getenv("DATABASE_URL"))
    if not live:      # a throw-away database holding just the starter phrases
        os.environ["DATABASE_URL"] = "sqlite:///" + os.path.join(tempfile.mkdtemp(), "train.db").replace(os.sep, "/")
    os.environ.setdefault("MARKET_AUTO", "0")
    os.environ["NLU_ACCURACY_CHECK"] = "off"      # measured below, on purpose, after training
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    import brain
    import db
    import nlu

    if not live:
        print("No DATABASE_URL set: training on the starter phrases only (seed + Q&A sheet).")
        brain.seed(json.loads((HERE / "catalog.json").read_text(encoding="utf-8")))
        brain._import_sheet()
    intents = db.list_intents()
    ok = {n for n, i in intents.items() if i["enabled"]}
    examples = [(e["intent"], e["text"]) for e in db.list_examples() if e["intent"] in ok]
    if len(ok) < 2 or len(examples) < 50:
        sys.exit(f"Only {len(examples)} phrases for {len(ok)} intents found. Is DATABASE_URL the right database?")
    print(f"Training on {len(examples)} phrases, {len({i for i, _ in examples})} intents "
          f"({'live database' if live else 'starter set'}) ...")

    info = nlu.engine.train(examples, backend="tfidf")
    if not info.get("trained"):
        sys.exit(f"Training failed: {info.get('reason')}")
    if not args.no_check:
        print("Measuring accuracy (each phrase predicted by a model that never saw it) ...")
        nlu.engine.info["accuracy"] = nlu.Engine._cross_validate(nlu.engine._quick["x"], nlu.engine._quick["y"])
        print(f"Held-out accuracy: {nlu.engine.info['accuracy']}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = nlu.engine.save(out, digest=nlu.examples_digest(examples))
    print(f"Saved {out} ({out.stat().st_size / 1024:.0f} KB): {meta['examples']} phrases, {meta['intents']} intents, "
          f"scikit-learn {meta['sklearn']}.")
    print("Next: git add models/nlu_model.pkl && git commit -m \"Retrained model\" && git push")


if __name__ == "__main__":
    main()
