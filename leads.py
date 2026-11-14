"""Turn what a person did in the chat into a lead score, reasons and a recommended next step.

Every rule here is based only on things the person actually did (signals recorded by the bot)
and on when they last wrote, so staff can see *why* a contact is marked hot, warm or cold.
"""
import datetime as dt

STAGES = ('new', 'contacted', 'quoted', 'survey booked', 'won', 'lost')
QUOTE_STEPS = {'name': 'their name', 'location': 'the installation town/suburb',
               'contact': 'a phone number', 'need': 'what they want to power'}


def _utc(value):
    if value is None:
        return None
    return value if value.tzinfo else value.replace(tzinfo=dt.timezone.utc)


def analyse(contact, now=None):
    """contact: dict from db.contact_dict(). Returns score, temperature, reasons, next_action, flags."""
    now = now or dt.datetime.now(dt.timezone.utc)
    s = contact.get('signals') or {}
    reasons, score = [], 0

    def add(points, why):
        nonlocal score
        score += points
        reasons.append(why)

    if s.get('callback'):
        add(30, 'Asked to speak to a person')
    if s.get('quote_completed'):
        add(35, 'Completed a quotation request')
    elif s.get('quote_requested'):
        add(20, 'Started a quotation request')
    if s.get('recommended'):
        add(15, f"Sized a system (about {s.get('sized_kwh', '?')} kWh/day) - suggested {s['recommended']}")
    if s.get('package'):
        add(5, f"Looked at {s['package']}")
    if s.get('viewed_packages'):
        add(10, 'Checked package prices')
    if s.get('location'):
        add(10, f"Gave location: {s['location']}")
    faqs = s.get('faq') or []
    if 'payment' in faqs:
        add(10, 'Asked about payment or deposit')
    other = [f for f in faqs if f != 'payment']
    if other:
        add(min(10, 5 * len(other)), 'Asked about ' + ', '.join(other))
    if (contact.get('inbound_count') or 0) >= 5:
        add(10, f"Engaged: {contact['inbound_count']} messages")

    last_in = _utc(contact.get('last_inbound'))
    hours = (now - last_in).total_seconds() / 3600 if last_in else None
    if hours is not None:
        if hours <= 24:
            add(10, 'Active in the last 24 hours')
        elif hours <= 72:
            add(5, 'Active in the last 3 days')
        elif hours > 24 * 14:
            score -= 15
            reasons.append('No message for over 2 weeks')
    score = max(0, min(100, score))
    temperature = 'hot' if score >= 60 else 'warm' if score >= 30 else 'cold'

    is_whatsapp = not str(contact.get('who', '')).startswith('web:')
    window_open = is_whatsapp and hours is not None and hours < 24
    stage = contact.get('stage') or 'new'
    where = f" in {s['location']}" if s.get('location') else ''
    want = s.get('need') or s.get('recommended') or s.get('package')

    if stage == 'won':
        action = 'Customer won: schedule the installation and ask for a referral.'
    elif stage == 'lost':
        action = 'Closed as lost. Optionally check back in about 3 months.'
    elif s.get('callback'):
        action = 'Call them now: they asked to speak to a person.'
    elif s.get('quote_completed') and stage in ('new', 'contacted'):
        action = (f"Prepare a quotation{f' for {want}' if want else ''} and offer a site survey{where}.")
    elif stage == 'quoted':
        action = 'Follow up on the quotation: answer questions and book a site survey.'
    elif stage == 'survey booked':
        action = 'Confirm the survey date and technician, then send the final quotation.'
    elif s.get('quote_requested'):
        missing = QUOTE_STEPS.get(s.get('quote_step'), 'their details')
        action = f'Started a quote but stopped before giving {missing}. Message them to finish it.'
    elif s.get('recommended'):
        action = f"Interested in {s['recommended']}. Offer a free site survey and ask about their budget."
    elif s.get('viewed_packages') or s.get('package'):
        action = 'Browsing prices. Ask what they want to power and suggest the right package.'
    else:
        action = 'Only said hello. Send a short, friendly message asking what they need.'

    if is_whatsapp and not window_open and stage not in ('won', 'lost'):
        action += (' Note: their last message is over 24 hours old, so WhatsApp only allows an approved'
                   ' template message to re-open the chat.')
    if not is_whatsapp:
        phone = s.get('contact')
        action += f' Website visitor: contact them on {phone}.' if phone else ' Website visitor without a phone number.'

    follow_up = contact.get('follow_up')
    due = bool(follow_up) and follow_up <= now.date().isoformat() and stage not in ('won', 'lost')
    return {'score': score, 'temperature': temperature, 'reasons': reasons, 'next_action': action,
            'window_open': window_open, 'is_whatsapp': is_whatsapp, 'follow_up_due': due}
