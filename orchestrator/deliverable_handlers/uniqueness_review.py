"""uniqueness_review.py — IP analysis: prior art and uniqueness (private)."""
from orchestrator.deliverable_handlers.base import DeliverableHandler

class UniquenessReviewHandler(DeliverableHandler):
    """Draft a uniqueness/novelty review. Internal-only."""
    DEFAULT_MAX_TOKENS = 4000
    DEFAULT_TEMPERATURE = 0.4

    def __init__(self, paper, deliverable_name, program_config, runs_root,
                 paper_repo_root=None, preflight=False, claude_client=None,
                 citation_engine=None, attorney_review_required=False):
        super().__init__(paper, deliverable_name, program_config, runs_root,
                        paper_repo_root, preflight, claude_client,
                        citation_engine, attorney_review_required=True)
