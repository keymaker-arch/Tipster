"""Configuration loading for Tipster.

Reads tipster.yaml for structure/behaviour config and .env for credentials.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Pydantic config models (mirror the tipster.yaml structure)
# ---------------------------------------------------------------------------

class TopicConfig(BaseModel):
    name: str
    description: str = ""
    relevance_hints: list[str] = Field(default_factory=list)
    link_score_hints: dict[str, list[str]] = Field(default_factory=dict)


class DiscoveryConfig(BaseModel):
    link_score_threshold: float = 0.6


class SourcesConfig(BaseModel):
    domain_weights: dict[str, float] = Field(default_factory=dict)
    blacklist: list[str] = Field(default_factory=list)


class ScheduleConfig(BaseModel):
    slice_duration_minutes: int = 60
    report_interval: str = "daily"
    report_time: str = "08:00"


class BudgetConfig(BaseModel):
    max_tokens_per_slice: int = 500_000
    max_cost_per_slice_usd: float = 0.50


class LLMConfig(BaseModel):
    onboard_model: str = "openai/gpt-5"
    triage_model: str = "openai/gpt-5"
    extraction_model: str = "openai/gpt-5"
    link_score_model: str = "openai/gpt-5"
    report_model: str = "openai/gpt-5"
    comment_model: str = "openai/gpt-5"

    # Optional custom base URLs (read from env)
    api_base: Optional[str] = None


class CrawlConfig(BaseModel):
    default_delay_seconds: int = 1
    # Advanced: cap concurrent workers to avoid overwhelming the server or API.
    # Reduce these if you see rate-limit errors or high memory usage.
    max_crawl_workers: int = 10  # max URLs fetched/triaged simultaneously
    max_llm_workers: int = 5     # max concurrent LLM calls across all crawl tasks


class TipsterConfig(BaseModel):
    topic: TopicConfig
    seed_urls: list[str] = Field(default_factory=list)
    discovery: DiscoveryConfig = Field(default_factory=DiscoveryConfig)
    sources: SourcesConfig = Field(default_factory=SourcesConfig)
    schedule: ScheduleConfig = Field(default_factory=ScheduleConfig)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    crawl: CrawlConfig = Field(default_factory=CrawlConfig)

    # Path to the DB file (not in yaml, set by init_db)
    db_path: str = "tipster.db"

    @classmethod
    def from_yaml(cls, yaml_path: str | Path = "tipster.yaml") -> "TipsterConfig":
        """Load config from a YAML file, expanding env var references."""
        path = Path(yaml_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path) as f:
            raw = yaml.safe_load(f)
        if raw is None:
            raw = {}
        return cls.model_validate(raw)


def load_config(
    yaml_path: str | Path = "tipster.yaml",
    env_path: str | Path = ".env",
) -> TipsterConfig:
    """Load tipster.yaml + .env into a TipsterConfig instance."""
    from dotenv import load_dotenv

    env = Path(env_path)
    if env.exists():
        load_dotenv(env)

    cfg = TipsterConfig.from_yaml(yaml_path)

    # Inject API base from env if present
    api_base = os.environ.get("TIPSTER_API_BASE") or os.environ.get("OPENAI_API_BASE")
    if api_base and not cfg.llm.api_base:
        cfg.llm.api_base = api_base

    return cfg
