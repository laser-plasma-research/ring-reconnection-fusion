"""
Deliverable handlers — one class per deliverable type.

Each handler takes a paper config + program config + AI clients, drafts
the corresponding deliverable, validates the content, and writes to disk.

Usage by orchestrator:
    from orchestrator.deliverable_handlers import get_handler
    HandlerClass = get_handler('manuscript')
    handler = HandlerClass(paper, 'manuscript', cfg, runs_root, ...)
    output_path = handler.run()
"""

from orchestrator.deliverable_handlers.base import (
    DeliverableHandler, StubHandler,
    DeliverableError, ContentValidationError,
)


def get_handler(deliverable_name: str):
    """
    Return the handler class for a deliverable type.

    Falls back to StubHandler if real handler not available (Step 4 work-in-progress).
    """
    # Lazy imports — let this module load even if some handlers fail to import
    handler_map = {}
    try:
        from orchestrator.deliverable_handlers.manuscript import ManuscriptHandler
        handler_map['manuscript'] = ManuscriptHandler
    except ImportError:
        pass
    try:
        from orchestrator.deliverable_handlers.references import ReferencesHandler
        handler_map['references'] = ReferencesHandler
    except ImportError:
        pass
    try:
        from orchestrator.deliverable_handlers.press_release import PressReleaseHandler
        handler_map['press_release'] = PressReleaseHandler
    except ImportError:
        pass
    try:
        from orchestrator.deliverable_handlers.lay_summary import LaySummaryHandler
        handler_map['lay_summary'] = LaySummaryHandler
    except ImportError:
        pass
    try:
        from orchestrator.deliverable_handlers.conference_abstract import ConferenceAbstractHandler
        handler_map['conference_abstract'] = ConferenceAbstractHandler
    except ImportError:
        pass
    try:
        from orchestrator.deliverable_handlers.industry_brief import IndustryBriefHandler
        handler_map['industry_brief'] = IndustryBriefHandler
    except ImportError:
        pass
    try:
        from orchestrator.deliverable_handlers.vc_briefing import VCBriefingHandler
        handler_map['vc_briefing'] = VCBriefingHandler
    except ImportError:
        pass
    try:
        from orchestrator.deliverable_handlers.potential_claims import PotentialClaimsHandler
        handler_map['potential_claims'] = PotentialClaimsHandler
    except ImportError:
        pass
    try:
        from orchestrator.deliverable_handlers.licensing_targets import LicensingTargetsHandler
        handler_map['licensing_targets'] = LicensingTargetsHandler
    except ImportError:
        pass
    try:
        from orchestrator.deliverable_handlers.valuation import ValuationHandler
        handler_map['valuation'] = ValuationHandler
    except ImportError:
        pass
    try:
        from orchestrator.deliverable_handlers.uniqueness_review import UniquenessReviewHandler
        handler_map['uniqueness_review'] = UniquenessReviewHandler
    except ImportError:
        pass
    try:
        from orchestrator.deliverable_handlers.metadata import MetadataHandler
        handler_map['metadata'] = MetadataHandler
    except ImportError:
        pass

    return handler_map.get(deliverable_name, StubHandler)


__all__ = ['DeliverableHandler', 'StubHandler', 'get_handler',
           'DeliverableError', 'ContentValidationError']
