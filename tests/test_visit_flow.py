"""The site-visit step must flow: it only books a visit when the customer gives a day or time, and a different
request in the middle (for example "I need a pdf quote") is answered, not booked as the visit time."""
import pytest

from test_pdf_quote import _flow_to_quote


def at_visit_question(chat):
    _flow_to_quote(chat)
    assert "what day" in chat.tap("visit").lower()


def test_a_pdf_request_in_the_middle_is_answered_not_booked(chat):
    """The customer's screenshot: asked for a day and time, answered 'I need a pdf quote'."""
    at_visit_question(chat)
    reply = chat.say("I need a pdf quote")
    assert "booked" not in reply.lower() and "technician visit for" not in reply.lower()
    assert chat.last.get("document"), "the quotation PDF should be sent again"
    assert "btn:visit" in chat.keys()                                    # the visit can still be booked from here
    assert "what day" in chat.tap("visit").lower()
    assert "Tuesday morning" in chat.say("Tuesday morning")


@pytest.mark.parametrize("when", ["Tuesday morning", "Friday after 2pm", "tomorrow", "this Saturday", "10am",
                                  "anytime", "next week", "the 25th", "after lunch", "Monday"])
def test_real_days_and_times_are_booked(chat, when):
    at_visit_question(chat)
    done = chat.say(when)
    assert "technician visit" in done and when in done


def test_something_that_is_not_a_time_asks_again(chat):
    at_visit_question(chat)
    reply = chat.say("I am not sure")
    assert "technician visit for" not in reply.lower()
    assert "day" in reply.lower() or "when" in reply.lower()
    assert "Friday" in chat.say("Friday")                                 # and the next answer still books it


def test_cancelling_still_works(chat):
    at_visit_question(chat)
    assert "no problem" in chat.say("not now").lower()


def test_pdf_request_after_a_quote_resends_it(chat):
    _flow_to_quote(chat)
    chat.say("can you send the quotation as a pdf")
    assert chat.last.get("document")
