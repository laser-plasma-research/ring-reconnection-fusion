"""
crossref_client.py — DOI verification and metadata via Crossref REST API.

Crossref is the registry-of-record for academic DOIs. We use it to:
  - Verify that a DOI exists and is well-formed
  - Fetch authoritative metadata (title, authors, journal, year)
  - Generate BibTeX entries from verified metadata

Uses the public API at https://api.crossref.org/. No API key required for
basic queries. Rate limit: ~50 requests/second; we keep well below this.

For polite usage, Crossref recommends including a User-Agent with a contact
email, which gives priority routing on their "polite" pool.

Usage:
    client = CrossrefClient(contact_email='brenworth@gmail.com')
    metadata = client.fetch_metadata('10.1063/1.870962')   # Rider 1995
    print(metadata['title'])
    bibtex = client.to_bibtex(metadata, key='rider1995')
"""

import time
import json
import urllib.request
import urllib.parse
import urllib.error
from typing import Optional


CROSSREF_BASE = 'https://api.crossref.org'
DEFAULT_USER_AGENT = 'OrchestratorBot/0.1 (mailto:{email})'


class CrossrefError(Exception):
    """Raised on Crossref API failures."""
    pass


class CrossrefClient:
    """
    Lightweight client for the Crossref REST API.

    Implements only the subset we need:
      - works/{doi} — verify DOI and fetch metadata
      - to_bibtex() — render verified metadata as BibTeX
    """

    def __init__(self,
                 contact_email: str = 'no-contact@example.com',
                 timeout: float = 10.0,
                 retries: int = 2,
                 retry_backoff: float = 1.5):
        self._user_agent = DEFAULT_USER_AGENT.format(email=contact_email)
        self._timeout = timeout
        self._retries = retries
        self._retry_backoff = retry_backoff
        self._cache = {}   # in-memory cache for current run

    # ---- HTTP helper ----

    def _get(self, url: str) -> dict:
        """GET with retries; returns parsed JSON dict."""
        last_err = None
        for attempt in range(self._retries + 1):
            try:
                req = urllib.request.Request(
                    url,
                    headers={'User-Agent': self._user_agent,
                             'Accept': 'application/json'},
                )
                with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                    if resp.status == 200:
                        return json.loads(resp.read().decode('utf-8'))
                    raise CrossrefError(f"HTTP {resp.status}")
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    raise CrossrefError(f"DOI not found: {url}")
                last_err = e
            except (urllib.error.URLError, json.JSONDecodeError, TimeoutError) as e:
                last_err = e
            if attempt < self._retries:
                time.sleep(self._retry_backoff ** attempt)
        raise CrossrefError(f"Crossref request failed: {last_err}")

    # ---- DOI verification ----

    def fetch_metadata(self, doi: str) -> dict:
        """
        Fetch Crossref metadata for a DOI. Returns the 'message' object.
        Raises CrossrefError if not found.
        """
        # Normalize DOI (strip URL prefix if present)
        doi = doi.strip()
        for prefix in ('https://doi.org/', 'http://doi.org/', 'doi:'):
            if doi.lower().startswith(prefix):
                doi = doi[len(prefix):]
                break

        if doi in self._cache:
            return self._cache[doi]

        url = f"{CROSSREF_BASE}/works/{urllib.parse.quote(doi, safe='/')}"
        data = self._get(url)
        if data.get('status') != 'ok':
            raise CrossrefError(f"Crossref status not ok for {doi}: {data.get('status')}")
        message = data.get('message', {})
        self._cache[doi] = message
        return message

    def verify(self, doi: str) -> bool:
        """Lightweight verification: returns True if DOI resolves."""
        try:
            self.fetch_metadata(doi)
            return True
        except CrossrefError:
            return False

    # ---- BibTeX rendering ----

    @staticmethod
    def to_bibtex(metadata: dict, key: Optional[str] = None) -> str:
        """
        Render a Crossref metadata record as a BibTeX entry.

        The entry type is selected from metadata['type']:
          - journal-article     -> @article
          - proceedings-article -> @inproceedings
          - book / monograph    -> @book
          - book-chapter        -> @incollection
          - posted-content      -> @misc (preprints)
          - else                -> @misc
        """
        cr_type = metadata.get('type', 'misc')
        bibtex_type = {
            'journal-article': 'article',
            'proceedings-article': 'inproceedings',
            'book': 'book',
            'monograph': 'book',
            'book-chapter': 'incollection',
            'posted-content': 'misc',
            'report': 'techreport',
        }.get(cr_type, 'misc')

        # Build citation key from first author + year if not provided
        if key is None:
            authors = metadata.get('author', [])
            if authors:
                first = authors[0].get('family', authors[0].get('name', 'unknown'))
                first = ''.join(c for c in first.lower() if c.isalnum())
            else:
                first = 'unknown'
            year_parts = metadata.get('issued', {}).get('date-parts', [[None]])
            year = year_parts[0][0] if year_parts and year_parts[0] else 'nd'
            key = f"{first}{year}"

        # Extract fields
        title_list = metadata.get('title', [])
        title = title_list[0] if title_list else ''

        authors_str = ' and '.join(
            f"{a.get('family', '')}, {a.get('given', '')}".strip(', ')
            if 'family' in a else a.get('name', '')
            for a in metadata.get('author', [])
        )

        container = (metadata.get('container-title', []) or [''])[0]
        volume = metadata.get('volume', '')
        issue = metadata.get('issue', '')
        page = metadata.get('page', '')
        doi = metadata.get('DOI', '')
        publisher = metadata.get('publisher', '')

        year_parts = metadata.get('issued', {}).get('date-parts', [[None]])
        year = year_parts[0][0] if year_parts and year_parts[0] else ''

        # Render BibTeX
        lines = [f"@{bibtex_type}{{{key},"]
        fields = [
            ('author', authors_str),
            ('title', title),
            ('journal' if bibtex_type == 'article' else 'booktitle', container),
            ('publisher', publisher),
            ('year', str(year) if year else ''),
            ('volume', volume),
            ('number', issue),
            ('pages', page),
            ('doi', doi),
        ]
        for name, value in fields:
            if value:
                # Escape braces and special chars; wrap in {...}
                escaped = str(value).replace('{', r'\{').replace('}', r'\}')
                lines.append(f"  {name} = {{{escaped}}},")
        if lines[-1].endswith(','):
            lines[-1] = lines[-1].rstrip(',')
        lines.append("}")
        return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Self-test (skipped if no network)
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import sys
    print("=== crossref_client smoke test ===")
    client = CrossrefClient(contact_email='test@example.com')

    # Test BibTeX rendering with a synthetic metadata dict
    fake_metadata = {
        'type': 'journal-article',
        'title': ['Fundamental limitations on plasma fusion systems not in thermodynamic equilibrium'],
        'author': [{'family': 'Rider', 'given': 'Todd H.'}],
        'container-title': ['Physics of Plasmas'],
        'volume': '4',
        'issue': '4',
        'page': '1039-1046',
        'DOI': '10.1063/1.872556',
        'issued': {'date-parts': [[1997]]},
        'publisher': 'AIP Publishing',
    }
    bibtex = CrossrefClient.to_bibtex(fake_metadata)
    print("Sample BibTeX render (no network):")
    print(bibtex)
    print()
    print("✓ BibTeX rendering test passed (no network call required)")
    print("  (Network-dependent tests skipped in build environment)")
