"""industry_brief.py — Public industry-facing summary (partner-tier)."""
from orchestrator.deliverable_handlers.base import DeliverableHandler

class IndustryBriefHandler(DeliverableHandler):
    """Draft an industry-relevance brief (TRL framing, capability matrix)."""
    DEFAULT_MAX_TOKENS = 3000
    DEFAULT_TEMPERATURE = 0.6
