"""
visibility.py — Generate .gitignore files and validate output paths.

The orchestrator must prevent private content (IP analysis, internal
strategy memos, credentials) from reaching public GitHub repos. This
module:

  - Generates .gitignore files from visibility config
  - Validates output paths against public/private classification
  - Provides helpers for deliverable handlers to determine where output goes

Visibility classification (from program_config.yaml):
  - public_paths    -> tracked by Git, pushed to GitHub
  - private_paths   -> gitignored, stays local
  - always_private  -> credentials/secrets, hard-coded, cannot be public

Files matching neither default to PRIVATE (default-deny).
"""

from pathlib import Path
from datetime import datetime


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class VisibilityError(Exception):
    """Raised when an output path violates visibility rules."""
    pass


# ---------------------------------------------------------------------------
# Path classification
# ---------------------------------------------------------------------------

def matches_pattern(path: str, patterns: list) -> bool:
    """
    Check whether `path` matches any of the gitignore-style patterns.
    Handles:
      - Exact filename matches (.env, secrets/)
      - Glob wildcards (*.token)
      - Directory patterns (ip_analysis/**)
    """
    from fnmatch import fnmatch

    p = Path(path)
    basename = p.name

    for pat in patterns:
        # Strip leading "./" (literal) — but NOT a dot at start of filename
        if pat.startswith('./'):
            pat = pat[2:]

        # Exact name match (.env matches both '.env' and 'shared/.env')
        if pat == basename:
            return True

        # Exact full-path match
        if pat == path:
            return True

        # Trailing slash means directory pattern
        if pat.endswith('/'):
            dir_pat = pat.rstrip('/')
            if path == dir_pat or path.startswith(dir_pat + '/'):
                return True
            continue

        # Path.match for simple patterns
        try:
            if p.match(pat):
                return True
        except ValueError:
            pass

        # Prefix-style patterns (ip_analysis/**)
        prefix = pat.rstrip('/**').rstrip('/')
        if prefix and (path == prefix or path.startswith(prefix + '/')):
            return True

        # Glob match for trailing wildcards (*.token)
        if '*' in pat:
            if fnmatch(path, pat):
                return True
            if fnmatch(basename, pat):
                return True
            # Try without leading directory separator
            if fnmatch(path, '*/' + pat):
                return True

    return False


def classify_path(path: str, visibility: dict) -> str:
    """
    Classify `path` as 'public', 'private', or 'always_private'.

    Default-deny: paths matching no list return 'private'.
    """
    always_private = visibility.get('always_private', [])
    private_paths = visibility.get('private_paths', [])
    public_paths = visibility.get('public_paths', [])

    if matches_pattern(path, always_private):
        return 'always_private'
    if matches_pattern(path, private_paths):
        return 'private'
    if matches_pattern(path, public_paths):
        return 'public'
    return 'private'   # default-deny


def is_public(path: str, visibility: dict) -> bool:
    return classify_path(path, visibility) == 'public'


def is_private(path: str, visibility: dict) -> bool:
    return classify_path(path, visibility) in ('private', 'always_private')


# ---------------------------------------------------------------------------
# .gitignore generation
# ---------------------------------------------------------------------------

GITIGNORE_HEADER = """\
# AUTO-GENERATED .gitignore — do not edit by hand.
# Generated from program_config.yaml visibility section.
# Generated: {timestamp}
# Regenerate with: python3 -c "from orchestrator.visibility import write_gitignore; ..."

"""


def generate_gitignore(visibility: dict) -> str:
    """
    Build a .gitignore file from the visibility config.

    Strategy: explicit denies for known private paths and credentials,
    plus standard noise (Python caches, OS files). Public paths are
    tracked by default (no entries needed for them).
    """
    lines = [GITIGNORE_HEADER.format(timestamp=datetime.now().isoformat(timespec='seconds'))]

    # Section 1: always-private (credentials, secrets)
    lines.append("# ─── Credentials and secrets (always private) ───\n")
    for pat in visibility.get('always_private', []):
        lines.append(pat)
    lines.append("")

    # Section 2: private paths (IP analysis, internal companion materials, etc.)
    lines.append("# ─── Internal content (not for public GitHub) ───")
    for pat in visibility.get('private_paths', []):
        lines.append(pat)
    lines.append("")

    # Section 3: standard noise that should never appear in any repo
    lines.append("# ─── Standard exclusions ───")
    lines.extend([
        "# macOS",
        ".DS_Store",
        "",
        "# Python",
        "__pycache__/",
        "*.pyc",
        "*.pyo",
        "*.pyd",
        ".pytest_cache/",
        ".mypy_cache/",
        "",
        "# IDE",
        ".vscode/",
        ".idea/",
        "*.swp",
        "*.swo",
        "",
        "# Simulation outputs (large; go to Zenodo, not Git)",
        "runs/",
        "*.log",
        "Backtrace.*",
        "warpx_used_inputs",
        "",
        "# Build artifacts",
        "build/",
        "dist/",
        "*.egg-info/",
        "",
    ])

    return '\n'.join(lines)


def write_gitignore(visibility: dict, path):
    """Write a generated .gitignore to `path` (overwriting any existing)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    content = generate_gitignore(visibility)
    p.write_text(content)
    return p


# ---------------------------------------------------------------------------
# Output path validation
# ---------------------------------------------------------------------------

def validate_output_path(
    output_path: str,
    declared_visibility: str,
    visibility_config: dict,
) -> None:
    """
    Verify that `output_path` is consistent with `declared_visibility`.

    A deliverable handler declares its visibility (public, partner, internal).
    The visibility config tells us where each tier goes. If a handler
    declares 'internal' but the path is in a public location, raise.

    Raises VisibilityError on mismatch.
    """
    actual = classify_path(output_path, visibility_config)

    if declared_visibility in ('public', 'partner'):
        if actual != 'public':
            raise VisibilityError(
                f"Declared visibility={declared_visibility!r} but path "
                f"{output_path!r} classified as {actual!r}. Either fix the "
                f"deliverable type's output_path or update visibility config."
            )
    elif declared_visibility == 'internal':
        if actual == 'public':
            raise VisibilityError(
                f"Declared visibility=internal but path {output_path!r} "
                f"classified as public. Internal content MUST NOT be in "
                f"a public path. Move output to a private directory."
            )
    else:
        raise VisibilityError(
            f"Unknown declared_visibility={declared_visibility!r}. "
            f"Expected one of: public, partner, internal."
        )


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    # Sanity check the matching against typical patterns
    config = {
        'public_paths': [
            'README.md',
            'paper/**',
            'simulation/**',
            'companion/public/**',
            'companion/partner/**',
        ],
        'private_paths': [
            'ip_analysis/**',
            'companion/internal/**',
            '*.private.md',
        ],
        'always_private': [
            '.env',
            '*.token',
            'secrets/',
        ],
    }

    test_cases = [
        ('paper/manuscript.tex', 'public'),
        ('simulation/main.py', 'public'),
        ('companion/public/press_release_en.md', 'public'),
        ('companion/partner/industry_brief.md', 'public'),
        ('companion/internal/vc_briefing.md', 'private'),
        ('ip_analysis/potential_claims.md', 'private'),
        ('strategy.private.md', 'private'),
        ('.env', 'always_private'),
        ('shared/.env', 'always_private'),
        ('something_secret.token', 'always_private'),
        ('unknown_file.md', 'private'),    # default-deny
    ]

    print("=== visibility.py self-test ===")
    for path, expected in test_cases:
        actual = classify_path(path, config)
        ok = '✓' if actual == expected else '✗'
        print(f"  {ok}  {path:45s}  expected={expected:15s}  actual={actual}")

    print()
    print("=== generated .gitignore preview ===")
    print(generate_gitignore(config)[:600] + '...')
