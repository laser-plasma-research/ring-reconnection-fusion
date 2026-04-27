"""conference_abstract.py — Abstract for conference submission."""
from orchestrator.deliverable_handlers.base import DeliverableHandler

class ConferenceAbstractHandler(DeliverableHandler):
    """Draft a conference abstract (typically 250-500 words)."""
    DEFAULT_MAX_TOKENS = 800
    DEFAULT_TEMPERATURE = 0.5
