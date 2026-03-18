"""Budget Gate — per-slice token and cost enforcement.

A single BudgetGate instance is created at the start of each crawl slice
and shared across all triage + extraction calls in that slice.
When either limit is hit, subsequent LLM calls are deferred to the next slice.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger("tipster.budget")


@dataclass
class BudgetGate:
    max_tokens: int
    max_cost_usd: float
    tokens_used: int = field(default=0, init=False)
    cost_usd: float = field(default=0.0, init=False)
    _exhausted: bool = field(default=False, init=False)

    def can_proceed(self) -> bool:
        """Return True if there is still budget available."""
        return not self._exhausted

    def record(self, tokens: int, cost: float) -> None:
        """Record token and cost consumption after an LLM call."""
        self.tokens_used += tokens
        self.cost_usd += cost
        if self.tokens_used >= self.max_tokens or self.cost_usd >= self.max_cost_usd:
            self._exhausted = True
            log.info(
                "Budget exhausted: %d/%d tokens, $%.4f/$%.4f",
                self.tokens_used,
                self.max_tokens,
                self.cost_usd,
                self.max_cost_usd,
            )

    def reset(self) -> None:
        """Reset for the next slice (called at the start of a new crawl slice)."""
        self.tokens_used = 0
        self.cost_usd = 0.0
        self._exhausted = False

    @property
    def summary(self) -> str:
        return (
            f"tokens={self.tokens_used}/{self.max_tokens}  "
            f"cost=${self.cost_usd:.4f}/${self.max_cost_usd:.2f}"
        )
