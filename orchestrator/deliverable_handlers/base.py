"""
base.py — Base class for deliverable handlers.

Each deliverable type (manuscript, press_release, vc_briefing, etc.) is
implemented as a subclass of DeliverableHandler. The base class defines
the contract:

  - resolve_output_path(paper, deliverable_type) -> Path
  - render_prompt(paper, type_def, context) -> dict   # {system, user}
  - call_api(prompt) -> str                            # actual Claude call
  - validate_output(text, type_def) -> bool            # forbidden/required check
  - write_output(text, path)                           # writes file

In Step 3 we ship a StubHandler that fakes the API call (writes
placeholder content) so the orchestrator can be tested end-to-end
without burning API credits. Step 4 replaces the stub with real
handlers per deliverable type.
"""

import os
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
    Abstract base. Subclasses implement specific deliverable types.

    A handler is created per (paper, deliverable_type) pairing and
    invoked once via run().
    """

    def __init__(self, paper, deliverable_name: str,
                 program_config, runs_root,
                 paper_repo_root: Optional[Path] = None,
                 preflight: bool = False):
        """
        paper: PaperConfig
        deliverable_name: e.g. 'manuscript'
        program_config: ProgramConfig
        runs_root: where simulation outputs live
        paper_repo_root: where deliverable outputs are written; defaults to
                         {program_root}/papers/{repo_name}/
        preflight: if True, use cheap model and short outputs
        """
        self.paper = paper
        self.name = deliverable_name
        self.config = program_config
        self.runs_root = Path(runs_root)
        self.preflight = preflight

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

        Subclasses typically don't override this — they override the
        render_prompt(), call_api(), and validate_output() methods.
        """
        # Resolve the output path
        output_path = self.resolve_output_path()

        # Build the prompt
        prompt = self.render_prompt()

        # Call API (or stub)
        text = self.call_api(prompt)

        # Validate forbidden/required content
        self.validate_output(text)

        # Write output
        self.write_output(text, output_path)

        return output_path

    # ---- Path resolution ----

    def resolve_output_path(self) -> Path:
        """Resolve the output_path template using paper context."""
        template = self.type_def.get('output_path', '')
        if not template:
            raise DeliverableError(
                f"Deliverable {self.name!r} has no output_path defined")

        # Substitute common placeholders
        substitutions = {
            'paper_id': self.paper.id,
            'tag': self.paper.tag,
            'language': self.paper.drafting_context.get('default_language', 'en'),
        }

        # Some deliverables use {event} for conference-specific abstracts
        substitutions['event'] = self.paper.drafting_context.get(
            'target_event', 'unspecified').replace(' ', '_')

        try:
            relative = template.format(**substitutions)
        except KeyError as e:
            raise DeliverableError(
                f"Output path template {template!r} contains unknown "
                f"placeholder: {e}. Available: {list(substitutions.keys())}"
            )

        return self.paper_repo_root / relative

    # ---- Prompt rendering (subclasses override or extend) ----

    def render_prompt(self) -> dict:
        """
        Build {system, user} prompt from type_def + paper context.

        Subclasses can override this for special handling. Default impl
        substitutes paper context into the prompt templates from
        program_config.yaml.
        """
        prompt_def = self.type_def.get('prompt', {})
        system = prompt_def.get('system', '')
        user_template = prompt_def.get('user_template', '')

        substitutions = self._collect_substitutions()
        try:
            user = self._safe_format(user_template, substitutions)
        except KeyError as e:
            raise DeliverableError(
                f"Prompt template for {self.name} references unknown "
                f"variable: {e}"
            )

        return {'system': system, 'user': user}

    def _collect_substitutions(self) -> dict:
        """Build a dict of variables for prompt template substitution."""
        # Make a flat dict that handles dotted access via _safe_format
        return {
            'title': self.paper.title,
            'tag': self.paper.tag,
            'paper_id': self.paper.id,
            'target_journal': self.paper.target_journal,
            'language': self.paper.drafting_context.get('default_language', 'en'),
            'drafting_context': self.paper.drafting_context,
            'citation_strategy': self.paper.citation_strategy,
            'internal_patent_strategy': self.paper.internal_patent_strategy,
            'analysis_outputs': '<analysis_outputs not yet wired in Step 3>',
            'figures_with_captions': '<figures not yet wired in Step 3>',
            'manuscript_text': '<manuscript_text not yet wired in Step 3>',
            'verified_candidates': '<citation candidates not yet wired in Step 3>',
            'v4_ppa_excerpt': '<PPA excerpt not yet wired in Step 3>',
            'authors': self.paper.drafting_context.get('authors', 'Worth, J. B.'),
            'orcid': self.paper.drafting_context.get('orcid', '0009-0005-5000-9497'),
            'abstract': self.paper.drafting_context.get('abstract', ''),
            'sections': '\n'.join(self.paper.drafting_context.get('sections_to_draft', [])),
            'latex_class': self.config.get_journal(self.paper.target_journal).get(
                'latex_class', 'article'),
        }

    @staticmethod
    def _safe_format(template: str, mapping: dict) -> str:
        """
        Format with dotted paths e.g. {drafting_context.framing.problem}.

        Handles missing keys by substituting <missing: key> rather than raising.
        """
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

        # Use string.Formatter to walk the template
        out = []
        for literal, field_name, _, _ in string.Formatter().parse(template):
            out.append(literal)
            if field_name is not None:
                out.append(resolve(field_name))
        return ''.join(out)

    # ---- API call (subclass overrides for real impl) ----

    def call_api(self, prompt: dict) -> str:
        """
        Call the AI API with the given prompt. Returns text.

        Base implementation raises NotImplementedError. The StubHandler
        overrides this to write placeholder text. Step 4 introduces a
        real handler that calls Claude via anthropic SDK.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement call_api(). "
            f"Use StubHandler in Step 3 or wait for Step 4 handlers."
        )

    # ---- Output validation ----

    def validate_output(self, text: str) -> None:
        """
        Check forbidden_content and required_content rules.

        Forbidden patterns are case-insensitive substring checks against
        the content. Required patterns are absence-checked similarly.

        Raises ContentValidationError on violation.
        """
        prompt_def = self.type_def.get('prompt', {})
        forbidden = prompt_def.get('forbidden_content', [])
        required = prompt_def.get('required_content', [])

        text_lower = text.lower()
        violations = []

        # Forbidden checks: simple substring matching for now (Step 4 may
        # add more sophisticated regex/semantic checks).
        # NOTE: forbidden_content lists are descriptive guidelines, not
        # literal strings to match. We do a heuristic check on a small
        # set of high-confidence patterns. Step 4 will refine this.
        HEURISTIC_FORBIDDEN_PATTERNS = {
            'patent claim numbers': [
                r'claim\s+\d+',  # "Claim 33"
                r'claim\s+number\s+\d+',
            ],
            'gain lever references': [
                r'lever\s+\d+',  # "Lever 4"
                r'gain stack',
            ],
        }

        # Apply heuristic checks for items where we have specific patterns
        import re
        for desc in forbidden:
            desc_lower = desc.lower()
            for pattern_key, regexes in HEURISTIC_FORBIDDEN_PATTERNS.items():
                if pattern_key in desc_lower:
                    for rgx in regexes:
                        if re.search(rgx, text, re.IGNORECASE):
                            violations.append(
                                f"forbidden pattern matched ({pattern_key}): "
                                f"regex {rgx!r}")

        # Required content: heuristic check only, log warnings rather
        # than fail. Step 4 will refine.
        warnings = []
        for desc in required:
            # Light heuristic for sections that should appear
            desc_lower = desc.lower()
            if 'abstract' in desc_lower and 'abstract' not in text_lower:
                warnings.append(f"required section 'abstract' not detected")
            if 'data availability' in desc_lower and 'data availability' not in text_lower:
                warnings.append(f"required section 'data availability' not detected")

        if violations:
            raise ContentValidationError(
                f"Output for {self.name} ({self.paper.id}) violated "
                f"forbidden_content rules:\n  " + "\n  ".join(violations)
            )

        # Warnings printed for visibility but don't fail
        if warnings:
            import sys
            for w in warnings:
                print(f"  WARN: {self.paper.id}/{self.name}: {w}",
                      file=sys.stderr)

    # ---- File writing ----

    def write_output(self, text: str, path: Path):
        """Write the deliverable text to disk, creating dirs as needed."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)


# ---------------------------------------------------------------------------
# StubHandler — used in Step 3 testing without API calls
# ---------------------------------------------------------------------------

class StubHandler(DeliverableHandler):
    """
    Mock handler that writes placeholder text instead of calling the API.

    Used in Step 3 to verify the orchestrator pipeline end-to-end without
    burning API credits. Each output explicitly marks itself as a stub.
    """

    def call_api(self, prompt: dict) -> str:
        """Generate placeholder content describing what would be drafted."""
        from datetime import datetime

        visibility = self.type_def.get('visibility', 'public')
        format_type = self.type_def.get('format', 'markdown')

        # Generate format-appropriate placeholder
        if format_type == 'latex':
            return self._latex_stub(prompt, visibility)
        elif format_type == 'bibtex':
            return self._bibtex_stub(prompt, visibility)
        elif format_type == 'json':
            return self._json_stub(prompt, visibility)
        else:
            return self._markdown_stub(prompt, visibility)

    def _markdown_stub(self, prompt: dict, visibility: str) -> str:
        from datetime import datetime
        return f"""# {self.name.upper()} — STUB

**Paper:** {self.paper.id} — {self.paper.title}
**Visibility:** {visibility}
**Generated:** {datetime.now().isoformat(timespec='seconds')}
**Model:** (stub — no API call)
**Preflight:** {self.preflight}

This is a placeholder generated by the orchestrator's StubHandler in Step 3.
Step 4 will replace this with a real Claude API call producing genuine content.

---

## Prompt that would have been sent

### System

{prompt.get('system', '(no system prompt)')[:500]}{'...' if len(prompt.get('system', '')) > 500 else ''}

### User

{prompt.get('user', '(no user prompt)')[:1500]}{'...' if len(prompt.get('user', '')) > 1500 else ''}

---

*End of stub.*
"""

    def _latex_stub(self, prompt: dict, visibility: str) -> str:
        from datetime import datetime
        return f"""% STUB — {self.name} for {self.paper.id}
% Generated: {datetime.now().isoformat(timespec='seconds')}
% Visibility: {visibility}
%
% This is a placeholder LaTeX file from the StubHandler.
% Step 4 will replace this with a real Claude-drafted manuscript.

\\documentclass{{article}}
\\title{{{self.paper.title} (STUB)}}
\\author{{Worth, J. B.}}
\\begin{{document}}
\\maketitle

\\begin{{abstract}}
This is a stub. Step 4 will produce real content.
\\end{{abstract}}

\\section{{Introduction}}
Stub content for {self.paper.id}.

\\section{{Method}}
Stub content for {self.paper.id}.

\\section{{Results}}
Stub content for {self.paper.id}.

\\section{{Discussion}}
Stub content for {self.paper.id}.

\\section{{Conclusions}}
Stub content for {self.paper.id}.

\\end{{document}}
"""

    def _bibtex_stub(self, prompt: dict, visibility: str) -> str:
        from datetime import datetime
        return f"""% STUB references for {self.paper.id}
% Generated: {datetime.now().isoformat(timespec='seconds')}
% Step 4 will replace this with a real verified bibliography.

@article{{stub_reference,
  author = {{Stub Author}},
  title = {{Placeholder Reference for {self.paper.id}}},
  journal = {{Stub Journal}},
  year = {{2026}},
  note = {{Generated by StubHandler — replace in Step 4}}
}}
"""

    def _json_stub(self, prompt: dict, visibility: str) -> str:
        import json
        from datetime import datetime
        return json.dumps({
            "@context": "https://schema.org",
            "@type": "ScholarlyArticle",
            "name": self.paper.title,
            "_stub": True,
            "_generated": datetime.now().isoformat(timespec='seconds'),
            "_note": "Stub content. Step 4 will populate with real metadata.",
        }, indent=2)

    def validate_output(self, text: str) -> None:
        """Skip validation for stub content (it's intentionally generic)."""
        pass


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("base.py module loaded successfully")
    print(f"  Classes: DeliverableHandler, StubHandler")
    print(f"  Errors:  DeliverableError, ContentValidationError")
