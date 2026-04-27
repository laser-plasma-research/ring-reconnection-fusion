"""press_release.py — Public-facing press release."""
from orchestrator.deliverable_handlers.base import DeliverableHandler

class PressReleaseHandler(DeliverableHandler):
    """Draft a press release for public outreach."""
    DEFAULT_MAX_TOKENS = 1500
    DEFAULT_TEMPERATURE = 0.9   # more engaging tone
