"""
manuscript.py — Main paper draft (LaTeX manuscript).

Outputs to: paper/manuscript.tex
Format: LaTeX (revtex4-2 typically)
Visibility: public

Post-processes Claude output to strip any markdown wrapping that
sometimes appears around the LaTeX (markdown headers, fenced code
blocks, explanatory preambles).
"""

import re

from orchestrator.deliverable_handlers.base import DeliverableHandler


class ManuscriptHandler(DeliverableHandler):
    """Draft the main paper manuscript in LaTeX."""

    DEFAULT_MAX_TOKENS = 8000   # manuscripts are long
    DEFAULT_TEMPERATURE = 0.7   # slightly less creative for academic prose

    def get_max_tokens(self) -> int:
        if self.preflight:
            return int(8000 * self.config.preflight.get('length_factor', 0.1))
        return 8000

    def get_prompt(self) -> dict:
        """
        Override default prompt to add an explicit instruction that the
        output must be pure LaTeX with no markdown wrapping.
        """
        prompt = super().get_prompt()
        # Append a strict formatting instruction
        latex_instruction = (
            "\n\nCRITICAL FORMATTING REQUIREMENT: Output PURE LATEX ONLY. "
            "Do NOT wrap your output in markdown code fences (no ```latex or ```). "
            "Do NOT include markdown headers (# or ## or ###) before the LaTeX. "
            "Do NOT include preambular commentary or warnings outside the LaTeX. "
            "The very first character of your output should be either a "
            "LaTeX comment (%) or a backslash command (\\documentclass). "
            "Any non-LaTeX content will break downstream PDF compilation."
        )
        prompt['user'] = prompt['user'] + latex_instruction
        return prompt

    def call_api(self, prompt: dict, attempt: int = 0) -> str:
        """Call API and post-process to strip any markdown wrapping."""
        text = super().call_api(prompt, attempt)
        return self._strip_markdown_wrapping(text)

    @staticmethod
    def _strip_markdown_wrapping(text: str) -> str:
        r"""
        Remove markdown headers, fenced code blocks, and explanatory text
        that sometimes wrap the actual LaTeX.

        Strategy: find the first LaTeX-like line (starts with % or \) and
        the last one, take everything between them. If no LaTeX markers
        found, return unchanged.
        """
        lines = text.splitlines()

        # Find first LaTeX line
        first_latex = None
        for i, line in enumerate(lines):
            stripped = line.lstrip()
            if (stripped.startswith('%') or stripped.startswith('\\')):
                first_latex = i
                break

        if first_latex is None:
            return text   # No LaTeX detected; return unchanged

        # Find last LaTeX-relevant line — looking for \end{document} ideally
        last_latex = len(lines) - 1
        for i in range(len(lines) - 1, first_latex - 1, -1):
            stripped = lines[i].rstrip()
            if stripped == '\\end{document}':
                last_latex = i
                break
            # Or any line that looks like LaTeX (not markdown)
            if stripped and not stripped.startswith('```') and not stripped.startswith('#'):
                last_latex = i
                break

        # Extract the LaTeX content
        latex = '\n'.join(lines[first_latex:last_latex + 1])

        # Strip any leftover fenced code block markers within
        latex = re.sub(r'^\s*```[a-z]*\s*$', '', latex, flags=re.MULTILINE)
        latex = re.sub(r'^\s*```\s*$', '', latex, flags=re.MULTILINE)

        # Remove any leading/trailing blank lines
        latex = latex.strip() + '\n'

        return latex
