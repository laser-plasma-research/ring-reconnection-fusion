"""potential_claims.py — IP analysis: potential patent claims (private)."""
from orchestrator.deliverable_handlers.base import DeliverableHandler

class PotentialClaimsHandler(DeliverableHandler):
    """Draft a potential claims analysis. Always attorney-review required."""
    DEFAULT_MAX_TOKENS = 4000
    DEFAULT_TEMPERATURE = 0.4   # more deterministic for legal-adjacent content

    def __init__(self, paper, deliverable_name, program_config, runs_root,
                 paper_repo_root=None, preflight=False, claude_client=None,
                 citation_engine=None, attorney_review_required=False):
        super().__init__(paper, deliverable_name, program_config, runs_root,
                        paper_repo_root, preflight, claude_client,
                        citation_engine, attorney_review_required=True)
