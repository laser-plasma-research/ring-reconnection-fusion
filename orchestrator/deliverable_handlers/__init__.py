"""
Deliverable handler implementations.

Each deliverable type defined in program_config.yaml has a corresponding
handler class. Handlers translate paper context into AI prompts, call
the API, validate output, and write to the configured location.

Step 3 ships only the base class skeleton. Step 4 fills in the concrete
handlers (manuscript, references, press_release, etc.).
"""

from orchestrator.deliverable_handlers.base import DeliverableHandler, StubHandler

__all__ = ['DeliverableHandler', 'StubHandler']
