"""vc_briefing.py — Internal VC-facing briefing (private-tier)."""
from orchestrator.deliverable_handlers.base import DeliverableHandler

class VCBriefingHandler(DeliverableHandler):
    """Draft a VC briefing. Internal-only."""
    DEFAULT_MAX_TOKENS = 4000
    DEFAULT_TEMPERATURE = 0.5

    def __init__(self, paper, deliverable_name, program_config, runs_root,
                 paper_repo_root=None, preflight=False, claude_client=None,
                 citation_engine=None, attorney_review_required=False):
        # Force attorney review flag for VC briefings (they discuss IP)
        super().__init__(paper, deliverable_name, program_config, runs_root,
                        paper_repo_root, preflight, claude_client,
                        citation_engine, attorney_review_required=True)
