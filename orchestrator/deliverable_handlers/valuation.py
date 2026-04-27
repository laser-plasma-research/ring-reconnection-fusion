"""valuation.py — IP analysis: valuation estimate (private)."""
from orchestrator.deliverable_handlers.base import DeliverableHandler

class ValuationHandler(DeliverableHandler):
    """Draft a valuation analysis. Internal-only, attorney-review required."""
    DEFAULT_MAX_TOKENS = 3500
    DEFAULT_TEMPERATURE = 0.4

    def __init__(self, paper, deliverable_name, program_config, runs_root,
                 paper_repo_root=None, preflight=False, claude_client=None,
                 citation_engine=None, attorney_review_required=False):
        super().__init__(paper, deliverable_name, program_config, runs_root,
                        paper_repo_root, preflight, claude_client,
                        citation_engine, attorney_review_required=True)
