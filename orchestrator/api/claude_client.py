"""
claude_client.py — Anthropic Claude API wrapper.

Encapsulates:
  - SDK initialization with API key from env
  - Retry on transient failures (rate limit, overload, network)
  - Pre-call budget check via UsageTracker
  - Token accounting and post-call recording
  - Model selection (production vs preflight)

Usage:
    from orchestrator.api.claude_client import ClaudeClient
    from orchestrator.api.usage_tracker import UsageTracker
    
    tracker = UsageTracker(budget={...})
    client = ClaudeClient(usage_tracker=tracker)
    
    response = client.complete(
        model='claude-opus-4-6',
        system='You are a careful scientific writer.',
        user='Draft an abstract for...',
        max_tokens=4000,
        deliverable='manuscript', paper_id='A1',
    )
    print(response['text'])
"""

import os
import time
import sys
from typing import Optional


class ApiError(Exception):
    """Raised on non-recoverable Anthropic API failures."""
    pass


# Re-export for convenience
from orchestrator.api.usage_tracker import (
    UsageTracker, BudgetExceededError, estimate_cost
)


# ---------------------------------------------------------------------------
# Token estimation (input)
# ---------------------------------------------------------------------------

def estimate_input_tokens(system: str, user: str) -> int:
    """
    Crude token estimate: ~3.5 characters per token for English+code.
    Used for pre-call budget check before the API call returns actual usage.
    """
    total_chars = len(system or '') + len(user or '')
    return max(100, total_chars // 3)   # underestimate is fine for budgeting


# ---------------------------------------------------------------------------
# Claude client
# ---------------------------------------------------------------------------

class ClaudeClient:
    """
    Wrapper around anthropic.Anthropic with retries and usage tracking.

    Initialize once per orchestrator run; reuse for all API calls.
    """

    DEFAULT_MAX_TOKENS = 4096
    DEFAULT_RETRIES = 3
    RETRY_BACKOFF_BASE = 2.0   # exponential: 2s, 4s, 8s

    def __init__(self,
                 api_key: Optional[str] = None,
                 usage_tracker: Optional[UsageTracker] = None,
                 default_model: str = 'claude-opus-4-6',
                 retries: int = None):
        """
        Args:
            api_key: if None, read from ANTHROPIC_API_KEY env var
            usage_tracker: for budget enforcement and recording
            default_model: model to use if not specified per call
            retries: max retries on transient errors
        """
        # Lazy import — avoids hard dependency if user only uses stubs
        try:
            import anthropic
        except ImportError:
            raise ApiError(
                "anthropic SDK not installed. Run: pip install anthropic"
            )

        self._anthropic = anthropic
        api_key = api_key or os.environ.get('ANTHROPIC_API_KEY')
        if not api_key:
            raise ApiError(
                "ANTHROPIC_API_KEY not set. Configure shared/.env or export it."
            )

        self._client = anthropic.Anthropic(api_key=api_key)
        self._tracker = usage_tracker
        self._default_model = default_model
        self._retries = retries if retries is not None else self.DEFAULT_RETRIES

    # ---- Main entry point ----

    def complete(self,
                 system: str,
                 user: str,
                 model: Optional[str] = None,
                 max_tokens: int = DEFAULT_MAX_TOKENS,
                 temperature: float = 1.0,
                 deliverable: Optional[str] = None,
                 paper_id: Optional[str] = None,
                 stop_sequences: Optional[list] = None) -> dict:
        """
        Send a completion request to Claude.

        Returns: dict with keys:
          - text: the model's response text
          - input_tokens: actual input token count
          - output_tokens: actual output token count
          - cost_usd: actual cost (computed from MODEL_PRICING)
          - model: model used
          - stop_reason: e.g. 'end_turn', 'max_tokens'

        Raises:
          - BudgetExceededError if pre-call budget check fails
          - ApiError on permanent API failure
        """
        model = model or self._default_model

        # Pre-call budget check (estimate cost from input alone)
        if self._tracker:
            est_input = estimate_input_tokens(system, user)
            est_cost = estimate_cost(model, est_input, max_tokens)
            self._tracker.check_can_spend(est_cost)

        # Build messages
        messages = [{'role': 'user', 'content': user}]
        kwargs = {
            'model': model,
            'max_tokens': max_tokens,
            'system': system or '',
            'messages': messages,
            'temperature': temperature,
        }
        if stop_sequences:
            kwargs['stop_sequences'] = stop_sequences

        # Retry loop
        last_exception = None
        for attempt in range(self._retries + 1):
            try:
                response = self._client.messages.create(**kwargs)
                break
            except self._anthropic.RateLimitError as e:
                last_exception = e
                if attempt < self._retries:
                    delay = self.RETRY_BACKOFF_BASE ** (attempt + 1)
                    self._log_warn(f"rate limit, retrying in {delay:.1f}s "
                                   f"(attempt {attempt + 1}/{self._retries})")
                    time.sleep(delay)
                    continue
                raise ApiError(f"Rate limited after {self._retries} retries: {e}")
            except (self._anthropic.APIConnectionError,
                    self._anthropic.APITimeoutError,
                    self._anthropic.InternalServerError) as e:
                last_exception = e
                if attempt < self._retries:
                    delay = self.RETRY_BACKOFF_BASE ** (attempt + 1)
                    self._log_warn(f"transient error, retrying in {delay:.1f}s "
                                   f"({type(e).__name__}: {e})")
                    time.sleep(delay)
                    continue
                raise ApiError(f"Transient API error after retries: {e}")
            except self._anthropic.APIError as e:
                # Permanent error — don't retry
                raise ApiError(f"API error: {e}")
            except Exception as e:
                # Unexpected — wrap and re-raise
                raise ApiError(f"Unexpected error: {type(e).__name__}: {e}")
        else:
            raise ApiError(f"Exhausted retries: {last_exception}")

        # Extract response
        text = ''
        for block in response.content:
            if hasattr(block, 'text'):
                text += block.text

        usage = response.usage
        input_tokens = usage.input_tokens
        output_tokens = usage.output_tokens
        actual_cost = estimate_cost(model, input_tokens, output_tokens)

        # Record in usage tracker
        if self._tracker:
            self._tracker.record(
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                actual_cost_usd=actual_cost,
                deliverable=deliverable,
                paper_id=paper_id,
            )

        return {
            'text': text,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'cost_usd': actual_cost,
            'model': model,
            'stop_reason': response.stop_reason,
        }

    @staticmethod
    def _log_warn(msg: str):
        print(f"  WARN [ClaudeClient]: {msg}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import sys
    print("=== claude_client structure check ===")

    # Just verify the module imports and class is well-formed.
    # We don't make API calls in build (no key, no network).

    try:
        # Stub UsageTracker
        from orchestrator.api.usage_tracker import UsageTracker
        tracker = UsageTracker(budget={'per_run_usd': 1.0})
        print(f"  ✓ UsageTracker instantiates")

        # Don't actually create ClaudeClient (would require API key + SDK call)
        # Just verify the class is importable
        print(f"  ✓ ClaudeClient class importable")
        print(f"  ✓ ApiError, BudgetExceededError available")
        print(f"  ✓ estimate_input_tokens('a' * 100, 'b' * 100) = {estimate_input_tokens('a' * 100, 'b' * 100)}")
        print("  ✓ self-test complete")
    except Exception as e:
        print(f"  ✗ self-test failed: {e}", file=sys.stderr)
        sys.exit(1)
