"""
env_loader.py — Load shared/.env into os.environ.

Minimal alternative to python-dotenv. Reads KEY=VALUE pairs from a .env
file, stripping quotes if present. Does not export non-existent variables
or override existing ones (unless override=True).

Usage:
    from orchestrator.api.env_loader import load_dotenv
    load_dotenv('shared/.env')
    api_key = os.environ['ANTHROPIC_API_KEY']
"""

import os
from pathlib import Path
from typing import Optional


def load_dotenv(path, override: bool = False, required_keys: Optional[list] = None) -> dict:
    """
    Read KEY=VALUE pairs from `path` into os.environ.

    Args:
        path: path to .env file
        override: if True, replace existing env vars; if False (default),
                  preserve existing values (CI env wins over .env file)
        required_keys: optional list of keys that MUST be present after
                       loading; raises KeyError if any missing

    Returns:
        dict of keys/values that were loaded (regardless of whether they
        were applied to environ).

    Raises:
        FileNotFoundError if path doesn't exist
        KeyError if required_keys not satisfied
    """
    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f".env file not found: {p}")

    loaded = {}
    with open(p) as f:
        for line_num, raw_line in enumerate(f, 1):
            line = raw_line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' not in line:
                # Skip malformed lines silently (or could warn)
                continue
            key, _, value = line.partition('=')
            key = key.strip()
            value = value.strip()

            # Strip surrounding quotes if present
            if len(value) >= 2:
                if (value[0] == value[-1]) and value[0] in ('"', "'"):
                    value = value[1:-1]

            loaded[key] = value
            if override or key not in os.environ:
                os.environ[key] = value

    if required_keys:
        missing = [k for k in required_keys if k not in os.environ or not os.environ[k]]
        if missing:
            raise KeyError(
                f"Required env vars missing after loading {p}: {missing}"
            )

    return loaded


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import tempfile
    import sys

    # Test with a temp .env file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
        f.write('# A comment\n')
        f.write('SIMPLE=value1\n')
        f.write('WITH_QUOTES="quoted value"\n')
        f.write("WITH_SINGLE_QUOTES='another value'\n")
        f.write('EMPTY=\n')
        f.write('  WITH_SPACES = trimmed  \n')
        path = f.name

    loaded = load_dotenv(path)
    print('=== env_loader self-test ===')
    for k, v in loaded.items():
        print(f'  {k:25s} = {v!r}')
    
    os.unlink(path)
    print('  ✓ self-test complete')
