"""lay_summary.py — Plain-language summary for general audience."""
from orchestrator.deliverable_handlers.base import DeliverableHandler

class LaySummaryHandler(DeliverableHandler):
    """Draft a lay summary accessible to non-specialists."""
    DEFAULT_MAX_TOKENS = 1200
    DEFAULT_TEMPERATURE = 0.85
