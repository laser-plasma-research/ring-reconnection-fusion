"""
openalex_client.py — Citation discovery via OpenAlex.

OpenAlex (https://openalex.org) is a free, open-source index of scholarly
work. We use it to discover papers that should be cited:

  - Search by topic terms ("magnetic reconnection p-11B fusion")
  - Search by author name ("Hora", "Eliezer")
  - Filter by year, citation count, journal
  - Get DOIs that we then verify against Crossref

OpenAlex is rate-limited (10/sec polite, 100/sec per email). Using a
contact email gives priority routing.

Usage:
    client = OpenAlexClient(contact_email='brenworth@gmail.com')
    works = client.search_topics(['magnetic reconnection', 'fusion'],
                                  min_year=2015, max_results=20)
    for w in works:
        print(w['title'], w.get('doi'))
"""

import time
import json
import urllib.request
import urllib.parse
import urllib.error
from typing import Optional


OPENALEX_BASE = 'https://api.openalex.org'


class OpenAlexError(Exception):
    """Raised on OpenAlex API failures."""
    pass


class OpenAlexClient:
    """Lightweight client for OpenAlex search and works lookup."""

    def __init__(self,
                 contact_email: str = 'no-contact@example.com',
                 timeout: float = 15.0,
                 retries: int = 2,
                 retry_backoff: float = 1.5):
        self._contact_email = contact_email
        self._timeout = timeout
        self._retries = retries
        self._retry_backoff = retry_backoff
        self._cache = {}

    # ---- HTTP helper ----

    def _get(self, path: str, params: Optional[dict] = None) -> dict:
        params = dict(params or {})
        params['mailto'] = self._contact_email   # polite-pool routing
        url = f"{OPENALEX_BASE}{path}?{urllib.parse.urlencode(params, doseq=True)}"

        last_err = None
        for attempt in range(self._retries + 1):
            try:
                req = urllib.request.Request(
                    url,
                    headers={'User-Agent': f'OrchestratorBot/0.1 ({self._contact_email})',
                             'Accept': 'application/json'},
                )
                with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                    return json.loads(resp.read().decode('utf-8'))
            except (urllib.error.URLError, urllib.error.HTTPError,
                    json.JSONDecodeError, TimeoutError) as e:
                last_err = e
            if attempt < self._retries:
                time.sleep(self._retry_backoff ** attempt)
        raise OpenAlexError(f"OpenAlex request failed: {last_err}")

    # ---- Search functions ----

    def search_topics(self,
                      keywords: list,
                      min_year: Optional[int] = None,
                      max_year: Optional[int] = None,
                      min_citations: int = 0,
                      max_results: int = 25,
                      sort_by: str = 'cited_by_count:desc') -> list:
        """
        Search for papers matching topic keywords.

        Args:
            keywords: list of search terms (joined with spaces for OpenAlex)
            min_year, max_year: filter by publication year
            min_citations: minimum citation count
            max_results: cap on returned papers
            sort_by: OpenAlex sort field (default: most-cited first)

        Returns: list of work dicts (normalized to a small set of fields).
        """
        query = ' '.join(keywords)
        filters = []
        if min_year:
            filters.append(f'from_publication_date:{min_year}-01-01')
        if max_year:
            filters.append(f'to_publication_date:{max_year}-12-31')
        if min_citations > 0:
            filters.append(f'cited_by_count:>{min_citations}')

        params = {
            'search': query,
            'per-page': min(max_results, 200),
            'sort': sort_by,
        }
        if filters:
            params['filter'] = ','.join(filters)

        data = self._get('/works', params=params)
        results = data.get('results', [])
        return [self._normalize(w) for w in results[:max_results]]

    def search_by_author(self,
                         author_name: str,
                         topic_keywords: Optional[list] = None,
                         min_year: Optional[int] = None,
                         max_results: int = 10) -> list:
        """
        Find papers by an author, optionally filtered by topic.
        """
        params = {
            'search': author_name,
            'per-page': max_results,
            'sort': 'cited_by_count:desc',
        }
        # OpenAlex syntax for "papers by authors named X" uses authorships
        # search; the simpler approach is to filter on the search index
        if topic_keywords:
            params['search'] = f"{author_name} {' '.join(topic_keywords)}"
        if min_year:
            params['filter'] = f'from_publication_date:{min_year}-01-01'

        data = self._get('/works', params=params)
        results = data.get('results', [])
        return [self._normalize(w) for w in results[:max_results]]

    def fetch_work(self, openalex_id_or_doi: str) -> dict:
        """Fetch a specific work by ID or DOI."""
        if openalex_id_or_doi.startswith('10.'):
            path = f'/works/doi:{openalex_id_or_doi}'
        else:
            path = f'/works/{openalex_id_or_doi}'
        data = self._get(path)
        return self._normalize(data)

    # ---- Normalization ----

    @staticmethod
    def _normalize(work: dict) -> dict:
        """Reduce OpenAlex's verbose response to a compact dict."""
        if not work:
            return {}

        # Title
        title = work.get('title') or work.get('display_name') or ''

        # DOI (without URL prefix)
        doi = work.get('doi', '')
        if doi:
            for prefix in ('https://doi.org/', 'http://doi.org/'):
                if doi.startswith(prefix):
                    doi = doi[len(prefix):]
                    break

        # Authors
        authors = []
        for ship in work.get('authorships', [])[:10]:
            author = ship.get('author', {})
            name = author.get('display_name') or ''
            if name:
                authors.append(name)

        # Venue
        primary = work.get('primary_location', {}) or {}
        source = primary.get('source', {}) or {}
        venue = source.get('display_name', '')

        # Citation count
        citations = work.get('cited_by_count', 0) or 0

        # Year
        year = work.get('publication_year', None)

        # Abstract — OpenAlex uses inverted index
        abstract = ''
        ai = work.get('abstract_inverted_index')
        if ai:
            # Reverse the inverted index to text (truncated)
            try:
                positions = []
                for word, idxs in ai.items():
                    for i in idxs:
                        positions.append((i, word))
                positions.sort()
                abstract = ' '.join(w for _, w in positions[:300])
            except Exception:
                pass

        return {
            'title': title,
            'doi': doi,
            'authors': authors,
            'venue': venue,
            'year': year,
            'citations': citations,
            'abstract': abstract,
            'openalex_id': work.get('id', ''),
            'is_oa': work.get('open_access', {}).get('is_oa', False),
            'type': work.get('type', ''),
        }


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("=== openalex_client smoke test ===")
    print("(Network-dependent tests skipped in build environment)")
    print("Module structure check:")
    
    client = OpenAlexClient(contact_email='test@example.com')
    print(f"  ✓ Client instantiates")
    
    # Test normalization with synthetic data
    fake_work = {
        'title': 'Test Paper',
        'doi': 'https://doi.org/10.1234/test',
        'authorships': [{'author': {'display_name': 'Smith, John'}}],
        'primary_location': {'source': {'display_name': 'Test Journal'}},
        'cited_by_count': 42,
        'publication_year': 2024,
    }
    normalized = OpenAlexClient._normalize(fake_work)
    assert normalized['title'] == 'Test Paper'
    assert normalized['doi'] == '10.1234/test'   # URL prefix stripped
    assert normalized['citations'] == 42
    print(f"  ✓ Normalization works")
    print(f"  ✓ self-test complete")
