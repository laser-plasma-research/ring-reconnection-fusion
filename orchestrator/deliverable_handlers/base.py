"""
base.py — Base class for deliverable handlers (Step 4).

Provides:
  - DeliverableHandler: Abstract base with API integration
  - StubHandler: Step 3 fallback that writes placeholder content
  - Lifecycle: render_prompt -> call_api -> validate -> write
  - Retry on validation failures (up to 3 attempts)
  - Budget enforcement via UsageTracker
  - Content validation via ContentValidator
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Optional


class DeliverableError(Exception):
    """Raised when a deliverable cannot be produced."""
    pass


class ContentValidationError(DeliverableError):
    """Raised when AI output fails forbidden/required content checks."""
    pass


# ---------------------------------------------------------------------------
# Base handler
# ---------------------------------------------------------------------------

class DeliverableHandler:
    """
    Abstract base for deliverable handlers.

    Subclasses override:
      - get_prompt() to build the {system, user} prompt
      - get_max_tokens() to control output length
      - get_temperature() for creativity vs determinism
      - validate_output() if extra checks beyond default validator

    The default run() drives the whole pipeline.
    """

    DEFAULT_MAX_TOKENS = 4096
    DEFAULT_TEMPERATURE = 1.0
    MAX_VALIDATION_RETRIES = 1   # attempts after first; total = 2 calls max

    def __init__(self, paper, deliverable_name: str,
                 program_config, runs_root,
                 paper_repo_root: Optional[Path] = None,
                 preflight: bool = False,
                 claude_client=None,
                 citation_engine=None,
                 attorney_review_required: bool = False):
        self.paper = paper
        self.name = deliverable_name
        self.config = program_config
        self.runs_root = Path(runs_root)
        self.preflight = preflight
        self.claude_client = claude_client
        self.citation_engine = citation_engine
        self.attorney_review_required = attorney_review_required

        # Look up the deliverable type definition
        self.type_def = program_config.get_deliverable_type(deliverable_name)
        if not self.type_def:
            raise DeliverableError(
                f"Unknown deliverable type: {deliverable_name}. "
                f"Define it in program_config.yaml deliverable_types section."
            )

        # Resolve paper repo root
        if paper_repo_root is None:
            program_root = program_config.program_root
            self.paper_repo_root = program_root / 'papers' / paper.repo_name
        else:
            self.paper_repo_root = Path(paper_repo_root)

    # ---- Lifecycle ----

    def run(self) -> Path:
        """
        Execute the deliverable end-to-end. Returns the output path.
        """
        output_path = self.resolve_output_path()
        prompt = self.get_prompt()

        # Try up to MAX_VALIDATION_RETRIES + 1 times
        last_error = None
        for attempt in range(self.MAX_VALIDATION_RETRIES + 1):
            text = self.call_api(prompt, attempt=attempt)

            try:
                self.validate_output(text)
                # Validation passed — write and return
                if self.attorney_review_required:
                    text = self._inject_attorney_review_warning(text)
                self.write_output(text, output_path)
                return output_path
            except ContentValidationError as e:
                last_error = e
                # Augment prompt with violations for retry
                prompt = self._build_retry_prompt(prompt, e)

        # Exhausted retries
        raise ContentValidationError(
            f"Content validation failed after {self.MAX_VALIDATION_RETRIES + 1} "
            f"attempts for {self.paper.id}/{self.name}: {last_error}"
        )

    # ---- Path resolution ----

    def resolve_output_path(self) -> Path:
        """Resolve the output_path template using paper context."""
        template = self.type_def.get('output_path', '')
        if not template:
            raise DeliverableError(
                f"Deliverable {self.name!r} has no output_path defined")

        substitutions = {
            'paper_id': self.paper.id,
            'tag': self.paper.tag,
            'language': self.paper.drafting_context.get('default_language', 'en'),
            'event': (self.paper.drafting_context.get('target_event', 'unspecified')
                      .replace(' ', '_')),
        }

        try:
            relative = template.format(**substitutions)
        except KeyError as e:
            raise DeliverableError(
                f"Output path template {template!r} uses unknown placeholder: {e}"
            )

        return self.paper_repo_root / relative

    # ---- Prompt construction ----

    def get_prompt(self) -> dict:
        """
        Build {system, user} prompt from type_def + paper context.
        Subclasses can override for special handling.
        """
        prompt_def = self.type_def.get('prompt', {})
        system = prompt_def.get('system', '')
        user_template = prompt_def.get('user_template', '')

        # Adjust system for preflight
        if self.preflight:
            preflight_note = (
                "\n\n[PREFLIGHT MODE: produce a short skeleton (~10% normal length) "
                "to validate the pipeline. Mark output as PREFLIGHT.]"
            )
            system = system + preflight_note

        substitutions = self._collect_substitutions()
        user = self._safe_format(user_template, substitutions)

        # Add attorney review warning to user prompt if applicable
        if self.attorney_review_required:
            user += (
                "\n\nIMPORTANT: This paper introduces matter beyond the v4 PPA. "
                "Output is for internal review only and MUST be marked as such. "
                "Begin with: 'CONFIDENTIAL — REQUIRES ATTORNEY REVIEW BEFORE PUBLIC RELEASE'."
            )

        return {'system': system, 'user': user}

    def _collect_substitutions(self) -> dict:
        """Build a dict of variables for prompt template substitution."""
        return {
            'title': self.paper.title,
            'tag': self.paper.tag,
            'paper_id': self.paper.id,
            'target_journal': self.paper.target_journal,
            'language': self.paper.drafting_context.get('default_language', 'en'),
            'drafting_context': self.paper.drafting_context,
            'citation_strategy': self.paper.citation_strategy,
            'analysis_outputs': self._load_analysis_outputs(),
            'figures_with_captions': self._figures_summary(),
            'manuscript_text': self._load_existing('paper/manuscript.tex'),
            'verified_candidates': '<populated by ReferencesHandler>',
            'v4_ppa_excerpt': '<v4 PPA excerpt — see internal patent files>',
            'authors': self.paper.drafting_context.get('authors', 'Worth, J. B.'),
            'orcid': self.paper.drafting_context.get('orcid', '0009-0005-5000-9497'),
            'abstract': self.paper.drafting_context.get('abstract', ''),
            'sections': '\n'.join(self.paper.drafting_context.get('sections_to_draft', [])),
            'latex_class': self.config.get_journal(self.paper.target_journal).get(
                'latex_class', 'article'),
        }

    def _load_analysis_outputs(self) -> str:
        """Load relevant analysis outputs for this paper from runs/."""
        # Look for paper-specific analysis files
        paper_runs_dir = self.runs_root / f'paper{self.paper.paper_num:02d}' \
            if self.paper.paper_num else None

        if not paper_runs_dir or not paper_runs_dir.exists():
            return f"<no analysis outputs found at {paper_runs_dir}>"

        chunks = []
        # Look for key text/csv files
        for pattern in ['*.txt', '*_summary.csv', '*_results.csv']:
            for f in sorted(paper_runs_dir.glob(pattern))[:5]:
                try:
                    content = f.read_text(errors='replace')
                    chunks.append(f"\n--- {f.name} ---\n{content[:2000]}")
                except Exception:
                    pass

        if not chunks:
            return f"<paper{self.paper.paper_num:02d}/ exists but no readable analysis files>"
        return '\n'.join(chunks[:8000])   # cap total length

    def _figures_summary(self) -> str:
        """Summarize figures expected for this paper."""
        figs = self.paper.raw.get('figures', [])
        if not figs:
            return "<no figures specified for this paper>"
        lines = []
        for f in figs:
            if isinstance(f, str):
                fig_def = self.config.get_figure_type(f)
                lines.append(f"- {f}: {fig_def.get('description', '(no description)')}")
            elif isinstance(f, dict):
                lines.append(f"- {f.get('type')}: {f.get('caption', '(no caption)')}")
        return '\n'.join(lines)

    def _load_existing(self, relative_path: str) -> str:
        """Try to load an existing file (e.g. manuscript.tex for follow-on deliverables)."""
        path = self.paper_repo_root / relative_path
        if path.exists():
            try:
                return path.read_text(errors='replace')[:30000]
            except Exception:
                pass
        return f"<file {relative_path} not yet generated>"

    @staticmethod
    def _safe_format(template: str, mapping: dict) -> str:
        """Format with dotted paths e.g. {drafting_context.framing.problem}."""
        import string

        def resolve(name):
            parts = name.split('.')
            value = mapping
            for part in parts:
                if isinstance(value, dict):
                    if part not in value:
                        return f'<missing: {name}>'
                    value = value[part]
                else:
                    return f'<missing: {name}>'
            if isinstance(value, (list, tuple)):
                return '\n'.join(f'  - {v}' for v in value)
            return str(value)

        out = []
        for literal, field_name, _, _ in string.Formatter().parse(template):
            out.append(literal)
            if field_name is not None:
                out.append(resolve(field_name))
        return ''.join(out)

    # ---- API call ----

    def get_max_tokens(self) -> int:
        if self.preflight:
            return int(self.DEFAULT_MAX_TOKENS *
                      self.config.preflight.get('length_factor', 0.1))
        return self.DEFAULT_MAX_TOKENS

    def get_temperature(self) -> float:
        return self.DEFAULT_TEMPERATURE

    def get_model(self) -> str:
        if self.preflight:
            return self.config.preflight.get(
                'model', 'claude-haiku-4-5-20251001')
        return self.config.default_model

    def call_api(self, prompt: dict, attempt: int = 0) -> str:
        """
        Make the API call. Subclasses can override for special pre/post processing.
        """
        if self.claude_client is None:
            raise DeliverableError(
                f"No claude_client available; can't call API for {self.name}. "
                f"Use StubHandler instead, or pass --paper without --preflight "
                f"to disable AI calls."
            )

        result = self.claude_client.complete(
            system=prompt['system'],
            user=prompt['user'],
            model=self.get_model(),
            max_tokens=self.get_max_tokens(),
            temperature=self.get_temperature(),
            deliverable=self.name,
            paper_id=self.paper.id,
        )
        return result['text']

    # ---- Validation ----

    def validate_output(self, text: str) -> None:
        """
        Apply forbidden_content / required_content rules.
        Raises ContentValidationError on violations.
        """
        from orchestrator.content_validator import ContentValidator, ContentValidationError as CVE

        prompt_def = self.type_def.get('prompt', {})
        forbidden = prompt_def.get('forbidden_content', [])
        required = prompt_def.get('required_content', [])

        validator = ContentValidator(self.name, self.paper.id)
        try:
            validator.validate(text, forbidden, required, strict=True)
        except CVE as e:
            # Re-raise as our local ContentValidationError
            raise ContentValidationError(str(e))

    def _build_retry_prompt(self, prompt: dict, validation_error) -> dict:
        """Augment prompt with validation feedback for retry."""
        addendum = (
            f"\n\nIMPORTANT — your previous attempt failed validation: "
            f"{validation_error}. "
            f"Please regenerate avoiding these violations."
        )
        return {
            'system': prompt['system'],
            'user': prompt['user'] + addendum,
        }

    # ---- Output writing ----

    def write_output(self, text: str, path: Path):
        """Write the deliverable text to disk, creating dirs as needed."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)

    def _inject_attorney_review_warning(self, text: str) -> str:
        """Prepend an attorney review warning header."""
        format_type = self.type_def.get('format', 'markdown')
        if format_type == 'latex':
            warning = ('% =====================================================\n'
                       '% CONFIDENTIAL — REQUIRES ATTORNEY REVIEW\n'
                       '% This deliverable contains matter beyond the v4 PPA.\n'
                       '% DO NOT distribute, publish, or share publicly.\n'
                       '% =====================================================\n\n')
        elif format_type == 'json':
            # Don't modify JSON; warning would break it
            return text
        else:
            warning = ('---\n'
                       '**CONFIDENTIAL — REQUIRES ATTORNEY REVIEW**\n\n'
                       'This deliverable contains matter beyond the v4 PPA. '
                       'DO NOT distribute, publish, or share publicly until reviewed.\n'
                       '---\n\n')
        return warning + text


# ---------------------------------------------------------------------------
# StubHandler — fallback when no real handler implemented
# ---------------------------------------------------------------------------

class StubHandler(DeliverableHandler):
    """
    Mock handler that writes placeholder text instead of calling the API.
    Used as fallback when a real handler isn't available, and for tests
    that should not consume API credits.
    """

    def call_api(self, prompt: dict, attempt: int = 0) -> str:
        """Generate placeholder content describing what would be drafted."""
        visibility = self.type_def.get('visibility', 'public')
        format_type = self.type_def.get('format', 'markdown')

        if format_type == 'latex':
            return self._latex_stub(prompt, visibility)
        elif format_type == 'bibtex':
            return self._bibtex_stub(prompt, visibility)
        elif format_type == 'json':
            return self._json_stub(prompt, visibility)
        else:
            return self._markdown_stub(prompt, visibility)

    def validate_output(self, text: str) -> None:
        """Skip validation for stub content."""
        pass

    def _markdown_stub(self, prompt: dict, visibility: str) -> str:
        return f"""# {self.name.upper()} — STUB

**Paper:** {self.paper.id} — {self.paper.title}
**Visibility:** {visibility}
**Generated:** {datetime.now().isoformat(timespec='seconds')}

This is a placeholder generated by StubHandler. Replace with real handler
in production.

---

*End of stub.*
"""

    def _latex_stub(self, prompt: dict, visibility: str) -> str:
        return f"""% STUB — {self.name} for {self.paper.id}
% Generated: {datetime.now().isoformat(timespec='seconds')}

\\documentclass{{article}}
\\title{{{self.paper.title} (STUB)}}
\\author{{Worth, J. B.}}
\\begin{{document}}
\\maketitle

\\begin{{abstract}}
This is a stub generated by StubHandler.
\\end{{abstract}}

\\section{{Introduction}}
Stub content for {self.paper.id}.

\\end{{document}}
"""

    def _bibtex_stub(self, prompt: dict, visibility: str) -> str:
        return f"""% STUB references for {self.paper.id}
@misc{{stub_reference,
  author = {{Stub Author}},
  title = {{Placeholder for {self.paper.id}}},
  year = {{2026}},
  note = {{Generated by StubHandler}}
}}
"""

    def _json_stub(self, prompt: dict, visibility: str) -> str:
        import json
        return json.dumps({
            "@context": "https://schema.org",
            "@type": "ScholarlyArticle",
            "name": self.paper.title,
            "_stub": True,
            "_generated": datetime.now().isoformat(timespec='seconds'),
        }, indent=2)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("base.py module loaded successfully")
    print(f"  Classes: DeliverableHandler, StubHandler")
    print(f"  Errors: DeliverableError, ContentValidationError")
