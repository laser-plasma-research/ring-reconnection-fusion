"""
content_validator.py — Validate AI-drafted output against forbidden
and required content rules.

Each deliverable type in program_config.yaml declares:
  - forbidden_content: descriptions of content that must NOT appear
  - required_content: descriptions of content that MUST appear

This module translates those descriptions into concrete patterns and
applies them to AI output. On violation, raises ContentValidationError
which the deliverable handler catches and uses to trigger regeneration.

Validation philosophy:
  - Forbidden patterns: regex-based, hard fail if matched
  - Required patterns: heuristic check, soft warn (not blocking)
  - The Claude prompt is the primary defense; validation catches obvious lapses
  - Patterns are CONTEXT-AWARE: matches in section headers (line-start with
    markdown formatting like '## Claim 1:' or '**Claim 1**:') are EXEMPT.
    The intent is to forbid PROSE references like 'as set forth in Claim 5
    of the patent', NOT to forbid using 'Claim N' as a section heading
    for analytical discussion in IP analysis deliverables.
"""

import re
import sys
from typing import Optional


class ContentValidationError(Exception):
    """Raised when AI output fails forbidden_content or required_content checks."""
    def __init__(self, deliverable: str, paper_id: str, violations: list,
                 warnings: Optional[list] = None):
        self.deliverable = deliverable
        self.paper_id = paper_id
        self.violations = violations
        self.warnings = warnings or []
        msg = (f"Content validation failed for {paper_id}/{deliverable}: "
               f"{len(violations)} violation(s)")
        super().__init__(msg)


# ---------------------------------------------------------------------------
# Forbidden pattern catalog
# ---------------------------------------------------------------------------
#
# Each key is a substring that may appear in a forbidden_content description.
# Each value is a list of regex patterns. If any pattern matches the AI output
# AND is not exempt under the section-header rule, the content is flagged.
#
# Keep this catalog conservative: false positives waste API tokens on retries,
# false negatives let bad content slip through.
#
# Patterns intended to forbid PROSE references — e.g. "as described in Claim 5"
# — should not match SECTION HEADERS used for discussion — e.g. "## Claim 1:".

# Lines that look like section headers should NOT trigger forbidden-pattern
# checks for ambiguous patterns like "Claim N" or "Lever N". A section header
# is identified by:
#   - Line starting with markdown header markers (#, ##, ###, ####)
#   - OR line wrapped entirely in markdown bold (**...**)
#   - OR line ending with a colon (the heading-of-a-block convention)
#
# We strip these section-header lines from the text BEFORE applying ambiguous
# patterns, so they don't trigger false positives. Other patterns (gain stack,
# patent strategy, breakeven claims) are NOT exempted — those terms are
# unambiguous regardless of context.

SECTION_HEADER_PATTERNS = [
    r'^\s*#{1,6}\s+.*$',                     # markdown headers
    r'^\s*\*\*[^*\n]+\*\*\s*:?\s*$',         # **bold heading**
    r'^\s*\*\*[^*\n]*Claim\s+\d+[^*\n]*\*\*', # **Claim 1: ...**
]

# Patterns that ARE context-aware — they get the section-header exemption
CONTEXT_AWARE_PATTERN_KEYS = {
    'patent claim numbers',
    'specific claim',
    'gain lever',
}

FORBIDDEN_PATTERNS = {
    # Patent-related terms (CONTEXT-AWARE — exempted in section headers)
    'patent claim numbers': [
        r'\bclaim\s+\d+\b',
        r'\bclaims?\s+\d+\s*(?:through|to|–|—|-)\s*\d+\b',
    ],
    'specific claim': [
        r'\bclaim\s+\d+\b',
    ],
    'gain lever': [
        r'\blever\s+\d+\b',
        r'\bgain\s+lever\b',
    ],

    # Patterns that are unambiguous (NOT context-aware)
    'gain stack': [
        r'\bgain\s+stack\b',
    ],
    'patent strategy': [
        r'\bpatent\s+strategy\b',
        r'\bstrategic\s+claim\b',
    ],
    'pre-ionisation': [
        r'\bpre[-\s]?ioni[sz]ation\s+lever\b',
        r'\bavalanche\s+pre[-\s]?ioni[sz]ation\b',
    ],

    'internal terminology': [
        # Defined per-paper; this is a stub
    ],
    'investor language': [
        r'\bvaluation\s+of\s+\$',
        r'\binvestment\s+round\b',
        r'\bSeries\s+[A-Z]\s+round\b',
        r'\bcap\s+table\b',
    ],

    # Speculation that goes too far
    'commercial timeline': [
        r'\bby\s+(?:the\s+)?(?:end\s+of\s+)?(?:202[5-9]|2030)\b.*(?:fusion|breakeven|net\s+power|commercial)',
        r'\b(?:fusion|breakeven|net\s+power|commercial)\b.*\bby\s+202[5-9]\b',
    ],

    # Misleading or unsupported claims
    'breakeven claim': [
        r'\bachieves?\s+(?:net\s+)?(?:energy\s+)?breakeven\b',
        r'\bQ\s*[><]\s*1\s*(?:has\s+been|is)\s+achiev',
    ],
    'fusion working': [
        r'\bfusion\s+(?:has been|is)\s+(?:demonstrated|achieved)\b',
    ],
}


REQUIRED_HEURISTICS = {
    'abstract': [r'(?i)\babstract\b'],
    'introduction': [r'(?i)\bintroduction\b', r'(?i)^#+\s*1\.?\s*introduction'],
    'methods': [r'(?i)\bmethods?\b', r'(?i)\bmethodology\b'],
    'results': [r'(?i)\bresults\b', r'(?i)\bfindings\b'],
    'discussion': [r'(?i)\bdiscussion\b'],
    'conclusion': [r'(?i)\bconclusion'],
    'references': [r'(?i)\breferences\b', r'(?i)\bbibliography\b'],
    'data availability': [r'(?i)data\s+availability', r'(?i)data\s+access'],
    'acknowledgments': [r'(?i)acknowledg(?:ments?|ements?)'],
    'doi': [r'\bdoi\s*[:=]', r'\bdoi\s*\.org/'],
    'orcid': [r'\bORCID\b', r'orcid\.org'],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strip_section_headers(text: str) -> str:
    """
    Replace lines matching SECTION_HEADER_PATTERNS with blank lines, so that
    pattern matches inside section headers don't trigger forbidden-content
    violations.

    Returns text with same line count (positions preserved for error reporting).
    """
    lines = text.splitlines()
    cleaned = []
    for line in lines:
        is_header = False
        for hdr_pattern in SECTION_HEADER_PATTERNS:
            if re.match(hdr_pattern, line, re.IGNORECASE):
                is_header = True
                break
        cleaned.append('' if is_header else line)
    return '\n'.join(cleaned)


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

class ContentValidator:
    """Apply forbidden and required content rules to AI output."""

    def __init__(self, deliverable_name: str, paper_id: str):
        self.deliverable = deliverable_name
        self.paper_id = paper_id

    def validate(self, text: str, forbidden_descriptions: list,
                 required_descriptions: list,
                 strict: bool = True) -> dict:
        """
        Apply rules to `text`.

        Args:
            text: AI-generated output
            forbidden_descriptions: list of strings describing things to forbid
            required_descriptions: similar for required
            strict: if True, raise ContentValidationError on violations.

        Returns:
            dict {violations: [...], warnings: [...]}

        Raises:
            ContentValidationError if strict and any violations.
        """
        violations = self._check_forbidden(text, forbidden_descriptions)
        warnings = self._check_required(text, required_descriptions)

        if strict and violations:
            raise ContentValidationError(
                self.deliverable, self.paper_id, violations, warnings
            )

        if warnings:
            for w in warnings:
                print(f"  WARN [ContentValidator] {self.paper_id}/{self.deliverable}: {w}",
                      file=sys.stderr, flush=True)

        return {'violations': violations, 'warnings': warnings}

    def _check_forbidden(self, text: str, descriptions: list) -> list:
        """
        Apply forbidden patterns; return list of violations.

        For context-aware patterns, applies the check to text with section
        headers stripped, so 'Claim 1:' as a section header doesn't trigger
        the 'patent claim numbers' rule.
        """
        violations = []
        text_no_headers = _strip_section_headers(text)

        for desc in descriptions:
            desc_lower = desc.lower()
            for pattern_key, regexes in FORBIDDEN_PATTERNS.items():
                if pattern_key not in desc_lower:
                    continue

                # Choose context: section-header-stripped for context-aware,
                # full text otherwise
                if pattern_key in CONTEXT_AWARE_PATTERN_KEYS:
                    text_to_check = text_no_headers
                else:
                    text_to_check = text

                for rgx in regexes:
                    match = re.search(rgx, text_to_check, re.IGNORECASE)
                    if match:
                        violations.append({
                            'rule': desc,
                            'pattern_key': pattern_key,
                            'regex': rgx,
                            'matched_text': match.group(0),
                            'position': match.start(),
                            'context_aware': pattern_key in CONTEXT_AWARE_PATTERN_KEYS,
                        })
        return violations

    def _check_required(self, text: str, descriptions: list) -> list:
        """Apply required heuristics; return list of warnings (missing items)."""
        warnings = []
        for desc in descriptions:
            desc_lower = desc.lower()
            for pattern_key, regexes in REQUIRED_HEURISTICS.items():
                if pattern_key in desc_lower:
                    if not any(re.search(rgx, text) for rgx in regexes):
                        warnings.append(
                            f"required content '{pattern_key}' not detected "
                            f"(rule: {desc!r})"
                        )
        return warnings

    @staticmethod
    def format_violations(violations: list) -> str:
        """Format violations for inclusion in retry prompt."""
        if not violations:
            return ""
        lines = ["The previous draft violated these forbidden_content rules:"]
        for v in violations:
            lines.append(f"  - Rule: {v['rule']!r}")
            lines.append(f"    Matched text: {v['matched_text']!r} "
                        f"(at position {v['position']})")
        lines.append("Please regenerate without these violations.")
        return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("=== content_validator self-test (v2 with context-aware patterns) ===")

    validator = ContentValidator('uniqueness_review', 'A1')

    # Test cases that previously caused false positives
    test_cases = [
        # (description, text, forbidden, expected_violations)
        ('section header — should NOT trigger',
         '## Claim 1: Centre region 95th percentile proton energy reaches >300 keV',
         ['patent claim numbers'],
         0),
        
        ('bold heading — should NOT trigger',
         '**Claim 2: Field depletion at X-line**',
         ['patent claim numbers'],
         0),
        
        ('multiple section headers — should NOT trigger',
         '''# Novelty Assessment

## Claim 1: Field depletion
Some prose.
## Claim 2: Energy threshold
More prose.''',
         ['patent claim numbers'],
         0),
        
        ('prose reference — SHOULD trigger',
         'As set forth in Claim 5 of the v4 PPA, the geometry is novel.',
         ['patent claim numbers'],
         1),
        
        ('mixed: section header OK, prose reference triggers',
         '''## Claim 1: Energy threshold
The paper claims that as set forth in Claim 33 of the patent, energies exceed 300 keV.''',
         ['patent claim numbers'],
         1),

        ('Lever N as section header — should NOT trigger',
         '## Lever 4: avalanche pre-ionization',
         ['gain lever'],
         0),

        ('Lever N as prose — SHOULD trigger',
         'The paper exploits Lever 4 of the analysis.',
         ['gain lever'],
         1),  # 'lever 4' fires the gain lever rule

        ('gain stack — always forbidden',
         'This is part of the gain stack analysis.',
         ['gain stack'],
         1),
    ]

    print("\nForbidden pattern false-positive tests:")
    all_ok = True
    for desc, text, forbidden, expected_count in test_cases:
        violations = validator._check_forbidden(text, forbidden)
        got = len(violations)
        passed = got == expected_count
        all_ok = all_ok and passed
        marker = '✓' if passed else '✗'
        print(f"  {marker}  {desc:55s} expected={expected_count}, got={got}")
        if not passed:
            for v in violations:
                print(f"     unexpected match: {v['matched_text']!r} "
                      f"(context_aware={v.get('context_aware')})")

    if all_ok:
        print("\n  ✓ All tests passed — context-aware patterns working correctly")
    else:
        print("\n  ✗ Some tests failed; review patterns")
        import sys
        sys.exit(1)

    print("\n  ✓ self-test complete")
