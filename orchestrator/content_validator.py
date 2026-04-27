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
# Each value is a list of regexes. If any regex matches the AI output, the
# content is flagged as violating that forbidden rule.
#
# Keep this catalog conservative: false positives waste API tokens on retries,
# false negatives let bad content slip through.

FORBIDDEN_PATTERNS = {
    # Patent-related terms
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

    # Internal-only terms
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
# Validator
# ---------------------------------------------------------------------------

class ContentValidator:
    """
    Apply forbidden and required content rules to AI output.
    """

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
                                    (from program_config.yaml)
            required_descriptions: similar for required
            strict: if True, raise ContentValidationError on violations.
                    If False, return result dict with violations listed.

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
        """Apply forbidden patterns; return list of violations."""
        violations = []
        for desc in descriptions:
            desc_lower = desc.lower()
            for pattern_key, regexes in FORBIDDEN_PATTERNS.items():
                if pattern_key in desc_lower:
                    for rgx in regexes:
                        match = re.search(rgx, text, re.IGNORECASE)
                        if match:
                            violations.append({
                                'rule': desc,
                                'pattern_key': pattern_key,
                                'regex': rgx,
                                'matched_text': match.group(0),
                                'position': match.start(),
                            })
        return violations

    def _check_required(self, text: str, descriptions: list) -> list:
        """Apply required heuristics; return list of warnings (missing items)."""
        warnings = []
        text_lower = text.lower()
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
    print("=== content_validator self-test ===")

    validator = ContentValidator('manuscript', 'A1')

    # Test forbidden pattern detection
    test_cases = [
        ("Normal academic text about reconnection physics.",
         ['patent claim numbers', 'gain lever'],
         []),

        ("As described in Claim 17, the geometry...",
         ['patent claim numbers'],
         ['claim 17']),

        ("The combined effect of Lever 4 and Lever 7...",
         ['gain lever'],
         ['lever 4', 'lever 7']),

        ("By 2030 net fusion power will be commercial.",
         ['commercial timeline'],
         ['by 2030', '2030']),

        ("This paper discusses the gain stack architecture.",
         ['gain stack'],
         ['gain stack']),
    ]

    print("\nForbidden pattern tests:")
    all_ok = True
    for text, forbidden, expected_matches in test_cases:
        violations = validator._check_forbidden(text, forbidden)
        matched = [v['matched_text'].lower() for v in violations]
        any_expected_found = any(em in str(matched) for em in expected_matches) if expected_matches else len(violations) == 0
        ok = '✓' if (any_expected_found or len(expected_matches) == 0) else '✗'
        if not (any_expected_found or len(expected_matches) == 0):
            all_ok = False
        print(f"  {ok}  text: {text[:50]!r:55s} expected hits: {expected_matches}, got: {matched}")

    # Test required content
    print("\nRequired content heuristic test:")
    text = """
# Introduction

Some content.

# Methods

Method details.

# Results

Results here.
"""
    warnings = validator._check_required(
        text,
        ['abstract section', 'introduction section', 'data availability', 'doi reference'],
    )
    for w in warnings:
        print(f"  - {w}")

    # Test strict mode
    print("\nStrict-mode test:")
    try:
        validator.validate(
            "This refers to Claim 5 and Lever 3.",
            forbidden_descriptions=['patent claim numbers', 'gain lever'],
            required_descriptions=[],
            strict=True,
        )
        print("  ✗ Should have raised ContentValidationError")
    except ContentValidationError as e:
        print(f"  ✓ Caught: {e}")

    print("  ✓ self-test complete")
