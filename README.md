# Tipster

**LLM-Powered Autonomous Web Intelligence Crawler**

Tipster is a long-running service that autonomously monitors the web on a topic you define. It discovers and ranks sources, extracts relevant information, and delivers periodic intelligence digests — all driven by LLMs. It learns from your feedback to continuously improve signal-to-noise over time.

## Features

- **Autonomous crawling** — runs 24/7, respects robots.txt and per-domain rate limits
- **LLM-powered triage** — scores content relevance (0–1), filters noise from signal
- **Structured extraction** — pulls facts and summaries from articles
- **Link scoring** — prioritizes new URLs based on topic relevance and past feedback
- **Report synthesis** — generates periodic Markdown digests with source attribution
- **Feedback loop** — user ratings improve future crawling and triage
- **Provider-agnostic** — any LLM via [LiteLLM](https://github.com/BerriAI/litellm) (OpenAI, Anthropic, Ollama, Mistral, Groq, etc.)
- **Terminal UI** — live Textual dashboard showing crawl activity, costs, and reports

## Requirements

- Python 3.11+
- Redis (for task queue)
- An LLM provider API key (or a local model via Ollama)

## Installation

```bash
pip install -e .
```

## Quick Start

**1. Initialize a new topic:**

```bash
tipster init
```

The wizard will ask for your topic, LLM provider credentials, and seed URLs. It generates a `tipster.yaml` config and a `.env` file for secrets.

**2. Start the crawler:**

```bash
tipster start
```

This launches the async crawl scheduler and opens the TUI dashboard.

**3. Other commands:**

```bash
tipster add-url <url>        # Manually add a URL to the crawl queue
tipster status               # Show a summary of crawled content and costs
tipster report               # Generate and print the latest intelligence digest
tipster export               # Merge user-feedback directives back into tipster.yaml
```

## Configuration

After `tipster init`, two files are created:

- **`tipster.yaml`** — topic definition, LLM model assignments, crawl settings, seed URLs
- **`.env`** — API keys (never committed to git)

Example `tipster.yaml` structure:

```yaml
topic:
  name: "AI safety research"
  description: "..."
  relevance_hints:
    - "alignment"
    - "interpretability"

models:
  triage_model: openai/gpt-4o-mini
  extraction_model: openai/gpt-4o
  report_model: anthropic/claude-opus-4-5

seed_urls:
  - https://example.com/ai-safety

budget:
  max_tokens_per_slice: 50000
  max_cost_per_day_usd: 1.00
```

## Architecture

```
tipster/
├── cli.py              # CLI entry point (Click)
├── scheduler.py        # Async crawl orchestration
├── crawler.py          # HTTP fetcher + text extraction (trafilatura)
├── triage.py           # LLM relevance scoring
├── extractor.py        # LLM fact + summary extraction
├── link_scorer.py      # LLM link prioritization
├── reporter.py         # Digest synthesis
├── llm.py              # LiteLLM abstraction
├── budget.py           # Token/cost budgeting
├── feedback.py         # User feedback handling
├── events.py           # Internal event bus
├── tui.py              # Textual dashboard
├── config.py           # Pydantic config model
├── onboarding.py       # Init wizard
└── db/
    ├── models.py        # SQLAlchemy ORM models
    ├── session.py       # DB session management
    └── repositories/    # Data access layer
```

**Database (SQLite by default):** topics, url_registry, content_items, feedback, directives, reports, prompt_examples.

## Development

```bash
pip install -e ".[dev]"
pytest tests/
```

## Roadmap

- [x] Phase 0 — Scaffold, onboarding, DB schema
- [x] Phase 1 — Crawler + text extraction
- [x] Phase 2 — LLM triage + relevance scoring
- [x] Phase 3 — Link scoring + discovery
- [x] Phase 4 — Report synthesis
- [ ] Phase 5 — Celery + production hardening
- [ ] Phase 6 — Multi-topic support, MCP server, REST API

## License

MIT
