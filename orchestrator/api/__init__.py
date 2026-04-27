"""
API integration package.

Provides:
  - env_loader: Load shared/.env into os.environ
  - claude_client: Anthropic SDK wrapper with retries and budget tracking
  - crossref_client: DOI verification and metadata lookup
  - openalex_client: Citation discovery via OpenAlex
  - usage_tracker: Persistent spend tracking against budget caps
"""

# Lazy imports — submodules are imported as needed by callers.
# This allows the package to load even if optional dependencies
# (anthropic, requests) are missing.

