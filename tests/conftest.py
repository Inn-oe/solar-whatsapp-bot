import os
import sys
import tempfile
from pathlib import Path

import pytest

# Throwaway database and no outside calls. Must be set before the app modules are imported.
_tmp = tempfile.mkdtemp()
os.environ["DATABASE_URL"] = "sqlite:///" + os.path.join(_tmp, "test.db").replace(os.sep, "/")
os.environ["MARKET_AUTO"] = "0"
os.environ["INTERNAL_KEY"] = "test-key"
os.environ.pop("WHATSAPP_TOKEN", None)
os.environ.pop("PHONE_NUMBER_ID", None)
os.environ.pop("ALERT_NUMBERS", None)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import app as bot  # noqa: E402
import brain  # noqa: E402
import db  # noqa: E402


@pytest.fixture(scope="session")
def client():
    return bot.app.test_client()


class Chat:
    """Talks to the bot the way a customer would, keeping the saved conversation state between messages."""

    def __init__(self, who="263771234567", name="Tendai Moyo"):
        self.who, self.name = who, name

    def say(self, text):
        db.touch_contact(self.who, self.name, "in")
        return bot.handle(self.who, self.name, text)["text"]


@pytest.fixture()
def chat():
    n = len(db.list_contacts()) + 1
    return Chat(who=f"2637700{n:05d}")
