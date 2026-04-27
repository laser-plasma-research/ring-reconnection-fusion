"""
references.py — Bibliography generation with verified citations.

Most complex handler. Pipeline:
  1. CitationEngine discovers candidates via OpenAlex
  2. Crossref verifies each candidate's DOI
  3. Claude selects which candidates to actually cite (typically 12-25)
  4. Verified BibTeX written to paper/references.bib

Outputs to: paper/references.bib
Format: BibTeX
Visibility: public
"""

import sys
import re
from pathlib import Path

from orchestrator.deliverable_handlers.base import (
    DeliverableHandler, DeliverableError
)


class ReferencesHandler(DeliverableHandler):
    """Generate paper/references.bib with verified citations."""

    DEFAULT_MAX_TOKENS = 4000

    def run(self) -> Path:
        """
        Override the standard run() to drive the citation pipeline.
        """
        if self.citation_engine is None:
            raise DeliverableError(
                "ReferencesHandler requires a citation_engine. "
                "Pass one via paper completion callback."
            )

        output_path = self.resolve_output_path()

        # Step 1: Discover and verify candidates
        candidates = self.citation_engine.discover_for_paper(
            paper_config=self.paper.raw,
            max_candidates=30,
            verify_dois=True,
        )

        if not candidates:
            self._warn("No verified citation candidates found. "
                       "Writing empty BibTeX file.")
            self.write_output(self._empty_bibtex(), output_path)
            return output_path

        # Step 2: Ask Claude to select which to actually cite
        selected_keys = self._select_citations(candidates)

        # Step 3: Render verified BibTeX
        bibtex_chunks = []
        for c in candidates:
            if c.get('doi') in selected_keys:
                metadata = c.get('crossref_metadata') or {}
                if not metadata:
                    continue
                key = self._make_citation_key(c)
                try:
                    bibtex = self.citation_engine._crossref.to_bibtex(
                        metadata, key=key)
                    bibtex_chunks.append(bibtex)
                except Exception as e:
                    self._warn(f"Failed to render BibTeX for {c.get('doi')}: {e}")

        # Step 4: Write
        if not bibtex_chunks:
            output = self._empty_bibtex()
        else:
            output = self._bibtex_header() + '\n\n' + '\n\n'.join(bibtex_chunks) + '\n'

        self.write_output(output, output_path)
        return output_path

    # ---- Selection (uses Claude) ----

    def _select_citations(self, candidates: list) -> set:
        """
        Ask Claude to select which candidates to actually cite.
        Returns set of DOIs to include.
        """
        if self.claude_client is None:
            # No API: include top half by score
            n = max(8, len(candidates) // 2)
            return set(c.get('doi', '') for c in candidates[:n])

        # Build candidate list for prompt
        candidate_summary = []
        for i, c in enumerate(candidates):
            authors = '; '.join(c.get('authors', [])[:3])
            if len(c.get('authors', [])) > 3:
                authors += ' et al.'
            line = (
                f"[{i + 1}] {c.get('title', '(no title)')[:200]}\n"
                f"     Authors: {authors}\n"
                f"     {c.get('venue', '?')} ({c.get('year', '?')}), "
                f"{c.get('citations', 0)} citations\n"
                f"     DOI: {c.get('doi', '')}\n"
            )
            candidate_summary.append(line)

        all_candidates = '\n'.join(candidate_summary)

        # Build paper context
        paper_context = (
            f"Title: {self.paper.title}\n"
            f"Tag: {self.paper.tag}\n"
            f"Topics: {self.paper.citation_strategy.get('topics', [])}\n"
            f"Key authors to consider: "
            f"{self.paper.citation_strategy.get('key_authors_to_cite', [])}\n"
        )

        system = (
            "You are a research librarian curating a citation list for an "
            "academic paper. Your task is to select which candidate papers "
            "from a discovered set should be cited. Aim for 12-20 citations "
            "that strongly support the paper's claims."
        )
        user = (
            f"Here is the paper context:\n\n{paper_context}\n\n"
            f"Here are the discovered citation candidates "
            f"({len(candidates)} total):\n\n"
            f"{all_candidates}\n\n"
            f"Please select 12-20 of these to cite. Output ONLY a "
            f"comma-separated list of citation numbers (e.g. '1, 3, 4, 7, 9, 12, 15, 18, 22'). "
            f"No other text, no explanation."
        )

        try:
            result = self.claude_client.complete(
                system=system,
                user=user,
                model=self.get_model(),
                max_tokens=500,
                temperature=0.3,
                deliverable=self.name,
                paper_id=self.paper.id,
            )
            text = result['text'].strip()
        except Exception as e:
            self._warn(f"Claude selection failed: {e}; "
                       f"falling back to top-scored candidates")
            n = max(12, min(20, len(candidates) // 2))
            return set(c.get('doi', '') for c in candidates[:n])

        # Parse selected indices
        selected_indices = set()
        for match in re.finditer(r'\b(\d+)\b', text):
            idx = int(match.group(1)) - 1
            if 0 <= idx < len(candidates):
                selected_indices.add(idx)

        if not selected_indices:
            self._warn("Claude returned no parseable indices; "
                       "falling back to top half")
            return set(c.get('doi', '') for c in candidates[:len(candidates) // 2])

        return set(candidates[i].get('doi', '') for i in selected_indices)

    # ---- Helpers ----

    @staticmethod
    def _make_citation_key(candidate: dict) -> str:
        """Build a citation key like 'hora2024' from candidate metadata."""
        authors = candidate.get('authors', [])
        if authors:
            first = authors[0]
            if ',' in first:
                last = first.split(',')[0].strip()
            else:
                parts = first.strip().split()
                last = parts[-1] if parts else 'unknown'
            last = ''.join(c for c in last.lower() if c.isalnum())
        else:
            last = 'unknown'
        year = candidate.get('year', 'nd')
        return f"{last}{year}"

    def _bibtex_header(self) -> str:
        from datetime import datetime
        return (
            f"% References for {self.paper.id} — {self.paper.title}\n"
            f"% Generated: {datetime.now().isoformat(timespec='seconds')}\n"
            f"% All DOIs verified against Crossref.\n"
        )

    def _empty_bibtex(self) -> str:
        return (
            self._bibtex_header() + "\n"
            "% No verified citation candidates found.\n"
            "% Re-run with broader citation_strategy keywords.\n"
        )

    def _warn(self, msg: str):
        print(f"  WARN [ReferencesHandler {self.paper.id}]: {msg}",
              file=sys.stderr, flush=True)
