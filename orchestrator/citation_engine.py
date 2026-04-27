"""
citation_engine.py — Strategic citation discovery and verification.

Coordinates OpenAlex (discovery) and Crossref (verification) to build a
ranked list of citation candidates for a paper. The handler then asks
Claude to select which candidates to actually cite.

Pipeline:
  1. Read citation_strategy from paper config
  2. Discover candidates via OpenAlex
       - Topic searches (keywords)
       - Author searches (key_authors_to_cite)
       - Recent papers (within last N years)
  3. Score candidates by:
       - Recency (newer = better, decaying)
       - Citation count (proxy for importance)
       - Target journal proximity (preferred journals score higher)
       - Strategic relevance (named authors score higher)
  4. Deduplicate by DOI
  5. Verify top candidates against Crossref
  6. Return verified list ready for Claude to select from

Usage:
    engine = CitationEngine(openalex, crossref)
    candidates = engine.discover_for_paper(
        paper_config=paper.raw,
        max_candidates=30,
    )
    # Each candidate: dict with title, authors, year, doi, score, ...
"""

import sys
from typing import Optional


class CitationEngine:
    """
    Discovers and verifies citation candidates for a paper.
    """

    DEFAULT_RECENCY_WINDOW_YEARS = 8   # papers within this window get recency boost
    DEFAULT_MAX_CANDIDATES = 30

    def __init__(self,
                 openalex_client,
                 crossref_client,
                 current_year: Optional[int] = None,
                 max_per_topic_search: int = 12,
                 max_per_author_search: int = 6):
        self._openalex = openalex_client
        self._crossref = crossref_client
        if current_year is None:
            from datetime import datetime
            current_year = datetime.now().year
        self._current_year = current_year
        self._max_per_topic = max_per_topic_search
        self._max_per_author = max_per_author_search

    # ---- Discovery ----

    def discover_for_paper(self,
                           paper_config: dict,
                           max_candidates: int = DEFAULT_MAX_CANDIDATES,
                           verify_dois: bool = True) -> list:
        """
        Discover, rank, and verify citation candidates for a paper.

        Args:
            paper_config: full paper YAML dict (with citation_strategy)
            max_candidates: cap on total candidates returned
            verify_dois: if True, verify each candidate's DOI exists on Crossref.
                         False is faster but allows hallucinated DOIs to pass through
                         (use False only for testing).

        Returns: list of candidate dicts, ranked by score descending. Each has:
            title, authors, year, doi, venue, citations, abstract, score, sources
        """
        strategy = paper_config.get('citation_strategy', {}) or {}
        candidates_by_doi = {}

        # 1. Topic searches (try both 'topics' and 'required_topic_coverage' keys)
        topics = (strategy.get('topics') or
                  strategy.get('required_topic_coverage') or [])
        for topic_query in topics:
            try:
                results = self._openalex.search_topics(
                    keywords=[topic_query] if isinstance(topic_query, str) else topic_query,
                    min_year=self._current_year - self.DEFAULT_RECENCY_WINDOW_YEARS - 5,
                    max_results=self._max_per_topic,
                )
            except Exception as e:
                self._log_warn(f"OpenAlex topic search failed ({topic_query!r}): {e}")
                continue
            for w in results:
                self._add_candidate(candidates_by_doi, w, source=f"topic:{topic_query}")

        # 2. Author searches
        authors = strategy.get('key_authors_to_cite', [])
        for author_name in authors:
            try:
                results = self._openalex.search_by_author(
                    author_name=author_name,
                    topic_keywords=topics[:1] if topics else None,
                    min_year=self._current_year - 30,   # broader window for known authors
                    max_results=self._max_per_author,
                )
            except Exception as e:
                self._log_warn(f"OpenAlex author search failed ({author_name!r}): {e}")
                continue
            for w in results:
                self._add_candidate(candidates_by_doi, w, source=f"author:{author_name}")

        # 3. Score and rank
        preferred_venues = self._normalize_venues(strategy.get('preferred_journals', []))
        named_authors = set(self._normalize_author(a) for a in authors)

        for c in candidates_by_doi.values():
            c['score'] = self._score(c, preferred_venues, named_authors)

        ranked = sorted(candidates_by_doi.values(),
                        key=lambda c: c['score'], reverse=True)

        # 4. Cap to max_candidates
        candidates = ranked[:max_candidates]

        # 5. Verify DOIs if requested
        if verify_dois and self._crossref is not None:
            verified = []
            for c in candidates:
                if not c.get('doi'):
                    continue
                try:
                    metadata = self._crossref.fetch_metadata(c['doi'])
                    c['crossref_metadata'] = metadata
                    c['verified'] = True
                    verified.append(c)
                except Exception as e:
                    self._log_warn(f"Crossref verification failed for "
                                   f"{c.get('doi')}: {e}")
                    # Skip unverified DOIs
            candidates = verified

        return candidates

    # ---- Helpers ----

    def _add_candidate(self, by_doi: dict, work: dict, source: str):
        """Deduplicate and merge by DOI; track sources."""
        doi = work.get('doi', '').strip().lower()
        if not doi:
            return
        if doi in by_doi:
            by_doi[doi].setdefault('sources', set()).add(source)
            return
        work = dict(work)   # copy
        work['sources'] = {source}
        by_doi[doi] = work

    def _score(self, candidate: dict, preferred_venues: set, named_authors: set) -> float:
        """
        Compute a composite score for ranking.

        Components:
          - recency: newer papers get up to +1.0
          - citations: log of citation count, normalized
          - venue: +1.0 if matches preferred journals
          - author: +1.5 if any author matches named author
          - sources: +0.3 per discovery source (recovered from multiple queries)
        """
        score = 0.0

        year = candidate.get('year') or 0
        if year:
            age = max(0, self._current_year - year)
            recency = max(0, 1.0 - (age / (self.DEFAULT_RECENCY_WINDOW_YEARS * 2)))
            score += recency

        citations = candidate.get('citations', 0) or 0
        if citations > 0:
            import math
            score += min(2.0, math.log1p(citations) / math.log(100))

        venue = (candidate.get('venue') or '').strip().lower()
        if venue and any(pv in venue or venue in pv for pv in preferred_venues):
            score += 1.0

        # Named author check
        author_match = False
        for a in candidate.get('authors', []):
            normalized = self._normalize_author(a)
            for named in named_authors:
                if named in normalized or normalized in named:
                    author_match = True
                    break
            if author_match:
                break
        if author_match:
            score += 1.5

        sources = len(candidate.get('sources', set()) or set())
        score += 0.3 * (sources - 1) if sources > 1 else 0

        return score

    @staticmethod
    def _normalize_venues(venues: list) -> set:
        return set(v.strip().lower() for v in venues if v)

    @staticmethod
    def _normalize_author(name: str) -> str:
        """Lowercase, drop initials, keep last name."""
        if not name:
            return ''
        # Last token is usually surname (works for "Last, F." or "F. Last")
        name = name.strip().lower()
        if ',' in name:
            return name.split(',')[0].strip()
        parts = name.split()
        return parts[-1] if parts else name

    @staticmethod
    def _log_warn(msg: str):
        print(f"  WARN [CitationEngine]: {msg}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("=== citation_engine self-test ===")

    # Test scoring without network
    class MockOpenAlex:
        def search_topics(self, **kw): return []
        def search_by_author(self, **kw): return []

    class MockCrossref:
        def fetch_metadata(self, doi):
            return {'DOI': doi, 'title': ['Mock'], 'type': 'journal-article'}

    engine = CitationEngine(MockOpenAlex(), MockCrossref(), current_year=2026)
    print(f"  ✓ Engine instantiates with mock clients")

    # Test scoring
    candidate = {
        'title': 'Test',
        'authors': ['Hora, Heinrich', 'Eliezer, Shalom'],
        'year': 2024,
        'citations': 50,
        'venue': 'Physics of Plasmas',
        'doi': '10.1234/test',
        'sources': {'topic:fusion', 'author:Hora'},
    }
    preferred = {'physics of plasmas', 'nuclear fusion'}
    named = {'hora', 'eliezer'}
    score = engine._score(candidate, preferred, named)
    print(f"  ✓ Score for high-quality candidate: {score:.2f}")
    assert score > 3.0   # should be high

    candidate_low = {
        'title': 'Off-topic paper',
        'authors': ['Unknown, A.'],
        'year': 2010,
        'citations': 5,
        'venue': 'Random Journal',
        'doi': '10.5678/lowq',
        'sources': {'topic:vague'},
    }
    score_low = engine._score(candidate_low, preferred, named)
    print(f"  ✓ Score for low-quality candidate: {score_low:.2f}")
    assert score_low < score

    print("  ✓ self-test complete")
