# Tipster: LLM-Powered Autonomous Web Intelligence Crawler

## Refined Concept

**Tipster** is a long-running autonomous intelligence service that monitors the web on a given topic, discovers and ranks sources, extracts relevant information, and delivers periodic digests to the user. It learns from feedback to sharpen signal over noise over time.

---

## Architecture Decision: Framework vs. Custom Service

### Why NOT a simple Skill
A Skill is a triggered, single-shot interaction. Tipster needs to run continuously in the background, manage state across sessions, schedule jobs, and react to external events. A Skill is the wrong primitive.

### Why NOT a raw LangChain/LangGraph Agent
- LangChain is designed for single-session, request-scoped agent loops — not persistent 24/7 services.
- LangGraph can model multi-step pipelines but adds graph-wiring overhead for what is fundamentally a pipeline of async workers.
- These frameworks often assume a single LLM provider per session, complicating per-task model routing.
- Upgrading or swapping the LLM layer becomes framework-dependent.

### Why NOT the Anthropic Agent SDK (Claude Code SDK)
- Ties the infrastructure to Anthropic's platform, violating the LLM-agnostic requirement.
- Not designed for long-running background services or job queues.

### Recommended: Custom Autonomous Service with LLM Abstraction Layer
A **Python-based autonomous service** with:
- Clean separation of crawling, analysis, scheduling, and delivery layers.
- A **provider-agnostic LLM interface** (via [LiteLLM](https://github.com/BerriAI/litellm)) supporting OpenAI, Anthropic, Mistral, Ollama, Groq, Cohere, and 100+ others.
- Pluggable LLM assignments per task type (e.g., cheap fast model for relevance triage, stronger model for report generation).
- A standard async Python stack — no opinionated agent framework locking.

---

## System Design

```
┌──────────────────────────────────────────────────────────────────┐
│                      Onboarding (tipster init)                   │
│   User provides free-text description + optional URLs            │
│   LLM → generates tipster.yaml (human-editable)                  │
│   User reviews/edits → tipster start                             │
└───────────────────────────┬──────────────────────────────────────┘
                            │ tipster.yaml (config)
┌───────────────────────────▼──────────────────────────────────────┐
│                        User Interface Layer                      │
│   CLI (MVP) / Web UI / MCP server (future)                       │
│   - View reports & digests                                       │
│   - Submit feedback on items and sources                         │
└───────────────────────────┬──────────────────────────────────────┘
                            │
┌───────────────────────────▼──────────────────────────────────────┐
│                       Orchestration Layer                        │
│   Scheduler: Celery Beat (bundled with Celery task queue)        │
│   - Crawl queue management                                       │
│   - Periodic re-crawl of known URLs                              │
│   - Report generation triggers                                   │
└──────┬──────────────┬──────────────┬────────────────┬────────────┘
       │              │              │                │
┌──────▼──────┐ ┌─────▼──────┐ ┌────▼──────┐ ┌──────▼──────┐
│  Discovery  │ │  Crawler   │ │ Analysis  │ │  Reporter   │
│  Module     │ │  Engine    │ │ Pipeline  │ │  Module     │
│             │ │            │ │           │ │             │
│ - Search    │ │ - Fetch    │ │ - Triage  │ │ - Digest    │
│   engine    │ │   pages    │ │   (LLM)   │ │   synthesis │
│   bootstrap │ │ - Extract  │ │ - Extract │ │   (LLM)     │
│ - Link      │ │   links &  │ │   key     │ │ - Deliver   │
│   scoring   │ │   content  │ │   claims  │ │   via       │
│   (LLM)     │ │ - Robots   │ │ - Score   │ │   channels  │
│             │ │   respect  │ │   source  │ │             │
└──────┬──────┘ └─────┬──────┘ └────┬──────┘ └──────┬──────┘
       │              │              │                │
┌──────▼──────────────▼──────────────▼────────────────▼──────────┐
│                     LLM Abstraction Layer                        │
│   LiteLLM router — provider-agnostic unified interface          │
│   6 task slots: onboard_model, triage_model, link_score_model,  │
│   extraction_model, report_model, comment_model                 │
│   Supports: OpenAI, Anthropic, Ollama, Mistral, Groq, etc.     │
└─────────────────────────────────────────────────────────────────┘
                            │
┌───────────────────────────▼──────────────────────────────────────┐
│                        Persistence Layer                          │
│                                                                   │
│  Storage Abstraction Layer (Repository pattern — swap backend via config)  │
│  ┌──────────────────┐   ┌──────────────────┐   ┌──────────────┐ │
│  │   URL Registry   │   │  Content Store   │   │  Feedback DB │ │
│  │ MVP: SQLite      │   │  MVP: SQLite     │   │  (same DB,   │ │
│  │ Prod: PostgreSQL │   │  Prod: PG +      │   │  tables)     │ │
│  │                  │   │  pgvector        │   │              │ │
│  │ - topic_id (FK)  │   │ - topic_id (FK)  │   │ - topic_id   │ │
│  │ - url            │   │ - url_id (FK)    │   │ - url_id(FK) │ │
│  │ - domain         │   │ - content_hash   │   │ - item_id    │ │
│  │ - relevance_score│   │ - raw_text       │   │   (FK→content│ │
│  │ - source_weight  │   │ - extracted_json │   │   _items)    │ │
│  │ - is_new_source  │   │ - article_sum_md │   │ - judgement  │ │
│  │ - last_checked   │   │   (per-article   │   │  (int/not)   │ │
│  │ - check_interval │   │   LLM summary)   │   │ - weight_    │ │
│  │ - status         │   │ - crawled_at     │   │   delta      │ │
│  │ - next_check_at  │   │ - topic_score    │   │   (float Δ)  │ │
│  │                  │   │ - status         │   │              │ │
│  │                  │   │   (pending_extr- │   │              │ │
│  │                  │   │    action |      │   │              │ │
│  │                  │   │    extracted |   │   │              │ │
│  │                  │   │    failed)       │   │              │ │
│  │                  │   │ - reported(bool) │   │              │ │
│  └──────────────────┘   └──────────────────┘   └──────────────┘ │
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                     Reports Table                        │   │
│  │  - report_id   - topic_id (FK)   - generated_at         │   │
│  │  - narrative_md (full digest text)                       │   │
│  │  - report_json (structured, links to content item IDs)   │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                   │
│  ┌──────────────────┐   ┌──────────────────────────────────────┐ │
│  │ Directive Store  │   │    Task Queue (Redis + Celery)        │ │
│  │ - topic_id (FK)  │   │  crawl_url | score_content |         │ │
│  │ - directive_type │   │  extract_pending | generate_report | │ │
│  │ - target         │   │  bootstrap                           │ │
│  │                  │   └──────────────────────────────────────┘ │
│  │ - params_json    │                                             │
│  │ - created_at     │   ┌──────────────────────────────────────┐ │
│  │ - expires_at     │   │    Prompt Examples Store             │ │
│  │ - applied        │   │  - topic_id (FK)                     │ │
│  └──────────────────┘   │  - domain (nullable)                 │ │
│                         │  - content_snippet                   │ │
│                         │  - judgement / label                 │ │
│                         │  - created_at (for recency sort)     │ │
│                         └──────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
```

---

## Core Modules

### 0. Onboarding Module (`tipster init`)
The entry point for a new monitoring task. Converts a user's natural language intent into a fully structured, human-editable YAML configuration file.

**Flow:**

**Step 0 — LLM Provider Setup (runs first, before anything else)**
1. CLI prompts the user to configure their LLM provider:
   - *"Which LLM provider will you use? (e.g. openai, anthropic, ollama, groq)"*
   - *"API base URL (leave blank for default, or enter e.g. http://localhost:11434 for Ollama):"*
   - *"API key (input is hidden):"*
   - *"Model to use for onboarding (e.g. gpt-4o, claude-opus-4-6, mistral):"*
2. Credentials are written to **`.env`** (not `tipster.yaml`) in the working directory. `.env` is auto-added to `.gitignore`. `tipster.yaml` references credentials only by environment variable name.
3. A **verification call** is made immediately: a minimal single-turn prompt (`"Reply with OK"`) is sent via LiteLLM using the supplied config. If it succeeds, onboarding continues. If it fails, the error is displayed with a suggested fix (wrong key, unreachable endpoint, model not found), and the user is re-prompted. Onboarding does not proceed until verification passes.

**Step 1 — Topic Description**
4. CLI prompts: *"Describe what you want to monitor. You can include starting links."*
5. User pastes free-form text — no structure required.
6. The verified LLM (`onboard_model`) parses the input and extracts:
   - Topic name and a prompt-ready description.
   - Seed URLs (from text or inlined links).
   - Relevance hints: key terms and concepts that define on-topic content.
   - Link score hints: positive anchor patterns (follow these) and negative patterns (skip these).
   - Initial source domain weights (inferred from any named sources or provided URLs).
   - Suggested schedule and budget settings.
7. Output is written to `tipster.yaml` with inline comments explaining every field so the user understands what to tune.
8. CLI renders a summary of what was extracted and asks: *"Does this look right? Edit tipster.yaml to adjust, then run `tipster start`."*

**Authority hierarchy — `tipster.yaml` vs `directive_store`:**
- `tipster.yaml` is the **static baseline**: initial seed URLs, hints, weights, and model config. It is read once at startup.
- `directive_store` holds **runtime overrides** that accumulate from user feedback. At runtime, directives take precedence over the corresponding `tipster.yaml` values (e.g., a `BLACKLIST_SOURCE` directive suppresses a domain even if it appears in `tipster.yaml`'s `domain_weights`).
- Re-running `tipster init` regenerates `tipster.yaml` from scratch. Before overwriting, the CLI warns: *"You have N active directives in the database. Run `tipster export` to merge them into the new config, or they will remain active independently."*
- `tipster export` writes the current effective state (baseline + directives) back to `tipster.yaml`, giving the user a clean merged snapshot they can inspect and edit.

**Example generated `tipster.yaml`:**
**`.env` (written by `tipster init`, never committed):**
```dotenv
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
# OLLAMA_API_BASE=http://localhost:11434
```

**`tipster.yaml` (generated, human-editable, safe to commit):**
```yaml
# tipster.yaml — generated by `tipster init`
# Edit any field to fine-tune crawler behaviour. Re-run `tipster init` to regenerate.
# Credentials are in .env — never put API keys in this file.

topic:
  name: "AI Safety Research"
  description: |
    Monitor developments in AI alignment and interpretability research,
    particularly from Anthropic, DeepMind, and OpenAI. Focus on technical
    papers, blog posts, and community discussions.

  # Terms injected into relevance triage and link scorer prompts.
  # Add or remove to sharpen what counts as on-topic.
  relevance_hints:
    - "AI alignment"
    - "interpretability"
    - "mechanistic interpretability"
    - "RLHF"
    - "AI safety"

  # Anchor text patterns used by the link scorer.
  # 'positive' boosts a link's score; 'negative' suppresses it.
  link_score_hints:
    positive:
      - "paper"
      - "research"
      - "preprint"
      - "alignment forum"
    negative:
      - "job listing"
      - "careers"
      - "press release"
      - "cookie policy"

seed_urls:
  - https://arxiv.org/list/cs.AI/recent
  - https://www.lesswrong.com
  - https://www.alignmentforum.org

discovery:
  # Links scoring below this threshold are not fetched (0.0–1.0).
  link_score_threshold: 0.6

sources:
  # Initial domain weights (0.0–1.0). Adjusted automatically by feedback.
  domain_weights:
    arxiv.org: 0.9
    lesswrong.com: 0.8
    alignmentforum.org: 0.8
    openai.com: 0.7
  blacklist: []

schedule:
  slice_duration_minutes: 60   # length of one crawl/budget time slice
  report_interval: "daily"     # daily | weekly | "0 8 * * *" (cron)
  report_time: "08:00"

budget:
  max_tokens_per_slice: 500000
  max_cost_per_slice_usd: 0.50

llm:
  onboard_model: "openai/gpt-4o"
  triage_model: "ollama/mistral"
  extraction_model: "openai/gpt-4o"
  link_score_model: "groq/llama-3-70b"
  report_model: "anthropic/claude-opus-4-6"
  comment_model: "openai/gpt-4o"
  # API keys are read from .env automatically via python-dotenv.
  # To override the base URL for self-hosted models:
  # ollama_api_base: "${OLLAMA_API_BASE}"

crawl:
  default_delay_seconds: 1  # minimum inter-request delay per domain when robots.txt is absent
```

### 1. Discovery Module
- **Bootstrap (Phase 4+)**: When no seed URLs are present, query search engine APIs (SerpAPI, Brave Search API, or DuckDuckGo scraper) to generate initial URL candidates. In Phase 1–3, seed URLs must be provided either via `tipster.yaml` or the `tipster add-url` CLI command.
- **Link Extractor**: From each crawled page, extract all outbound links.
- **Link Scorer (LLM)**: Given the anchor text, surrounding context, and the topic, the LLM scores each link 0–1 for likely relevance before fetching. Only links above `discovery.link_score_threshold` (default `0.6`, configurable in `tipster.yaml`) enter the crawl queue.

### 2. Crawler Engine
- **Baseline fetcher**: Python `httpx` for async HTTP, falling back to subprocess `curl` for compatibility or edge cases.
- HTML parsing via `trafilatura` (clean text extraction) + `BeautifulSoup` for link extraction.
- Pluggable fetcher interface: baseline HTTP is the default; Playwright (JS rendering) and other methods are future-pluggable options via the same interface.
- Robots.txt and crawl-delay respect enforced at the fetcher level. When `robots.txt` is absent or specifies no `Crawl-delay`, a default minimum inter-request delay of **1 second per domain** applies to avoid overloading servers.
- Pages that return 401/403 or require login are logged as `status=inaccessible` in the URL Registry and skipped; the adaptive interval backs off normally.
- Deduplication via content hash and URL normalization.

### 3. Analysis Pipeline (LLM-powered)
Each piece of fetched content passes through:
1. **Relevance Triage**: Fast/cheap model checks if content is on-topic. Irrelevant content is discarded.
2. **Fact Extraction**: Stronger model extracts key claims, entities, dates, and links from relevant content.
3. **Novelty Check**: Two-phase approach keyed by roadmap phase:
   - **Phase 1–3 (SQLite)**: SHA-256 hash of normalised article text. Exact-duplicate suppression only.
   - **Phase 4+ (PostgreSQL + pgvector)**: semantic embedding similarity added as a second layer. Near-duplicates (cosine similarity > 0.92) are also suppressed, preventing paraphrased re-posts from different sources.
4. **Source Credibility**: Over time, weight sources by user feedback and historical accuracy signals. Each article in the Content Store carries a `url_id` FK linking it back to its originating URL in the URL Registry, enabling per-source aggregation.

### 4. URL Registry & Scheduling
- Each URL has a dynamic `check_interval` that adapts based on observed update frequency:
  - **Signal**: a changed SHA-256 content hash at crawl time indicates new content.
  - **On new content detected**: `new_interval = max(1h, current_interval × 0.75)` — shrink by 25%, floor at 1 hour.
  - **On no new content**: `new_interval = min(7d, current_interval × 2.0)` — double (exponential backoff), cap at 7 days.
  - Default starting interval: 24h.
  - A `SCHEDULE_DEEP_DIVE` directive pins a URL to a specific interval, bypassing the adaptive algorithm for its duration.
- High-value, frequently-updating sources converge on shorter intervals; stale sources back off automatically.
- Scheduler fires re-crawl jobs based on `next_check_at` timestamps.
- **No hard hop limit.** Crawl depth is governed by the time-slice LLM budget (see Budget & Time-Slice Control below). When a time slice's token budget is exhausted, pending crawl tasks for that slice are re-queued for the next slice rather than dropped.

### 5. Budget & Time-Slice Control
- The crawl cycle is divided into configurable **time slices** (e.g., hourly or every N minutes).
- Each time slice has a **token budget** — a ceiling on total LLM tokens that may be consumed across all tasks (triage, link scoring, extraction, etc.).
- The budget is tracked by the orchestrator in real time. When the budget ceiling is hit, no new LLM-dependent tasks are dispatched until the next slice begins.
- Pure crawl/fetch tasks (no LLM) may continue within a slice regardless of budget state.
- This replaces hop-limit crawl scope control: the crawler can follow links as widely as it deems relevant, but is naturally throttled by the per-slice cost envelope.
- Budget config example:
  ```yaml
  budget:
    max_tokens_per_slice: 500000      # across all LLM calls in one slice
    max_cost_per_slice_usd: 0.50     # optional hard USD cap via LiteLLM cost tracking
  # slice_duration_minutes lives under `schedule:` — one canonical field
  ```

### 6. Reporter Module
- Triggered on a configurable schedule (e.g., daily digest, weekly deep-dive).
- Reads Content Store items where `reported=False` AND `status=extracted`. After report generation, all included items are marked `reported=True` — guaranteeing no item is missed or repeated across report cycles.
- LLM synthesizes collected facts into a **dual-format output**:
  - **Natural language narrative** delivered to the user (readable digest with headline findings, source attributions, confidence notes).
  - **Structured JSON** persisted to the content store for programmatic use, downstream tooling, or future trend analysis.
- Each reported intel item carries a unique ID, its originating source, and a flag indicating whether the source is **new** (first time seen) — prompting the user to rate it.
- **Delivery is channel-agnostic** via a pluggable `ReportChannel` interface. MVP ships one channel; future channels are drop-in additions:
  - **MVP**: Textual TUI — when a report fires, the running `tipster start` dashboard surfaces a "New Report" notification panel; the user can open it inline to read the digest and submit per-item feedback without leaving the TUI.
  - **Future**: Web server (FastAPI serving a local dashboard), MCP server (expose reports and feedback as MCP tools for LLM client integration), and others (email, Telegram).

### 7. Feedback Loop
- After receiving a report via the CLI, the user may optionally act on any intel item. All feedback is optional — the system functions fully without it.
- **New sources** are highlighted in the report with a ★ marker, actively inviting first-time feedback.
- Three feedback inputs per item, all optional:
  1. **Judgement**: `interesting` or `not interesting` — adjusts `source_weight` in the URL registry.
  2. **Free-text comment**: The user can annotate any item in natural language (e.g., "focus more on this company", "ignore press releases from this domain", "add forum threads about this topic to the crawl queue").
  3. **No action**: silently skip — item is not penalised.
- **Comment Interpreter (LLM-powered)**: When a comment is present, an LLM parses it and emits one or more structured **directives**. These directives are stored in a `directive_store` table and consumed by the appropriate subsystem on the next cycle:

  | Directive | Triggered by (example) | Effect |
  |-----------|------------------------|--------|
  | `BOOST_CRAWL_PRIORITY(target, magnitude, duration)` | "spend more effort on this company next week" | Shrinks `check_interval` for matching domain URLs by `magnitude` factor for the given `duration`; also boosts those URLs to the front of the crawl queue |
  | `UPDATE_LINK_SCORE_HINT(domain, hint_text)` | "forum threads on this site are more useful than blog posts" | Hint injected into link scorer prompt for that domain |
  | `EXPAND_TOPIC(subtopic)` | "also track their subsidiary XYZ" | Phase 1–3: adds the sub-topic to `relevance_hints` and emits a CLI notice prompting the user to add seed URLs manually via `tipster add-url`. Phase 4+: also triggers search-engine bootstrap to discover seed URLs automatically. |
  | `BLACKLIST_SOURCE(url_or_domain)` | "ignore press releases from this domain" | Source weight floored to zero; URL suppressed from future crawls |
  | `SCHEDULE_DEEP_DIVE(url, frequency)` | "check this page daily for updates" | Overrides the adaptive `check_interval` for that URL |

- **Directive conflict resolution** — applied in this priority order:
  1. `BLACKLIST_SOURCE` always wins over any other directive targeting the same domain (most-restrictive-wins). A blacklisted domain cannot be simultaneously boosted.
  2. For same-type conflicts on the same target (e.g., two `SCHEDULE_DEEP_DIVE` directives for the same URL), **last-write-wins**.
  3. For directives with an `expires_at`, expired entries are ignored at consumption time and purged on the next maintenance cycle.

- Accumulated judgements are persisted to the **Prompt Examples Store** and injected as few-shot examples into relevance triage and link scoring prompts. Selection strategy:
  - **Cap**: maximum 20 examples injected per prompt.
  - **Recency-first**: examples are sorted by `created_at` descending; most recent selected first.
  - **Domain filter**: when scoring links for a specific domain, domain-matching examples are prioritised over generic ones within the cap.
  - This keeps prompt size bounded and examples contextually relevant.

---

## LLM Agnosticism Strategy

Use **LiteLLM** as the single LLM gateway — it normalises the API surface across 100+ providers (OpenAI, Anthropic, Ollama, Groq, Mistral, Cohere, etc.). All internal code calls `litellm.completion(model=cfg.llm.triage_model, ...)`. Switching a provider or model requires only editing the `llm:` block in `tipster.yaml` — no code changes.

The canonical model config lives in `tipster.yaml` (see the Onboarding section for the full example). The six task-specific model slots are:

| Config key | Task | Suggested default |
|---|---|---|
| `onboard_model` | Narrative → config extraction | `openai/gpt-4o` |
| `triage_model` | Bulk relevance triage (high volume) | `ollama/mistral` |
| `link_score_model` | Link scoring before fetch | `groq/llama-3-70b` |
| `extraction_model` | Fact extraction from articles | `openai/gpt-4o` |
| `report_model` | Digest narrative synthesis | `anthropic/claude-opus-4-6` |
| `comment_model` | User comment → directives | `openai/gpt-4o` |

---

## Tech Stack

| Concern              | Technology                                      |
|----------------------|-------------------------------------------------|
| Language             | Python 3.11+                                    |
| Async runtime        | asyncio + httpx                                 |
| HTML parsing         | trafilatura (clean text) + BeautifulSoup        |
| HTTP baseline        | httpx + subprocess curl (fallback)              |
| JS rendering         | Playwright (future plug-in option)              |
| Task queue           | Celery + Redis                                  |
| Scheduler            | Celery Beat (bundled with Celery)               |
| Database (MVP)       | SQLite (via storage abstraction layer)          |
| Database (prod)      | PostgreSQL + pgvector (drop-in swap)            |
| ORM / abstraction    | SQLAlchemy 2.x + Repository pattern             |
| LLM gateway          | LiteLLM                                         |
| Onboarding CLI       | Click + Rich (interactive `tipster init`)       |
| Config format        | YAML with inline comments (`tipster.yaml`)      |
| Runtime TUI          | Textual (long-running dashboard; `tipster start`) |
| Reporter (MVP)       | Textual TUI panel — inline report overlay + feedback prompts |
| Reporter (future)    | FastAPI web dashboard, MCP server               |
| Search bootstrap     | SerpAPI / Brave Search API / DuckDuckGo         |
| Containerization     | Docker + docker-compose                         |
| Config management    | Pydantic Settings (env vars + YAML)             |
| Credential storage   | `.env` file + python-dotenv (never in YAML)     |

---

## Data Flow: Onboarding (`tipster init`)

```
[User] runs `tipster init`
        │
        ▼
[CLI] — Step 0: LLM Provider Setup
        │  prompts: provider, API base URL, API key, model name
        │  writes credentials to .env (masked input, never echoed)
        ▼
[LiteLLM] verification call: minimal prompt → "Reply with OK"
        │
        ├── FAIL → display error + suggested fix → re-prompt (loop until pass)
        │
        └── PASS
                │
                ▼
        [CLI] — Step 1: Topic Description
                │  "Describe what you want to monitor. You can include starting links."
                │  "Monitor AI safety research from LessWrong and arXiv..."
                ▼
        [Onboard LLM] parses intent → extracts:
                │  - topic name + prompt-ready description
                │  - seed URLs (from text + inline links)
                │  - relevance_hints (key terms)
                │  - link_score_hints (positive / negative anchor patterns)
                │  - domain_weights (inferred from named sources)
                │  - suggested schedule + budget
                ▼
        [Config Writer] renders tipster.yaml with inline comments
                │         (.env already written; tipster.yaml refs env var names only)
                ▼
        [CLI] shows summary of extracted fields
                │  "Extracted 3 seed URLs, 5 relevance hints, 4 link score hints.
                │   Edit tipster.yaml to adjust, then run `tipster start`."
                │
                └──▶ User edits tipster.yaml (optional fine-grain tuning)
                              │
                              ▼
                         `tipster start` — see startup flow below
```

## Data Flow: Startup (`tipster start`)

```
[User] runs `tipster start`
        │
        ▼
[Config Loader] reads tipster.yaml → validates with Pydantic
        │
        ▼
[Directive Loader] reads directive_store → merges runtime overrides
        │           (directives take precedence over tipster.yaml values)
        │
        ▼
[URL Registry] — already seeded? (check by topic_id row count)
        │
        ├── NO (fresh start / new topic):
        │       seed_urls from tipster.yaml → inserted into URL Registry
        │       → each seed URL queued as immediate crawl task
        │
        └── YES (restart / resume):
                skip re-seeding — existing registry state preserved
                pending tasks from previous slice re-queued if status=deferred
        │
        ▼
[Celery Beat] starts periodic schedules:
        - crawl re-check jobs (based on next_check_at per URL)
        - report generation (based on schedule.report_interval)
        │
        ▼
[Discovery Module] — no seed URLs in registry at all?
│       └── trigger search-engine bootstrap (SerpAPI / Brave) to populate
        │
        ▼
[Textual TUI] launches and takes over the terminal for the lifetime of the process
        - renders the dashboard (see TUI Layout below)
        - subscribes to the internal event bus for live log lines
        - Celery workers run in background threads / subprocesses
        - user presses `q` or Ctrl-C to initiate graceful shutdown
```

## TUI Layout (`tipster start`)

`tipster start` opens a long-running Textual dashboard that stays alive for the full session, modelled on the style of Claude Code and Codex: a dense header bar, a structured metadata panel, and a scrolling activity log that proves the system is making progress.

```
┌─────────────────────────────────────────────────────────────────────────┐
│ tipster  ●  AI Safety Research          uptime 02:14:33   [q] quit      │
├──────────────────────────────────┬──────────────────────────────────────┤
│  OVERVIEW                        │  SCHEDULE                            │
│  Topic      AI Safety Research   │  Last crawl    2 min ago             │
│  Status     running              │  Next crawl    in 8 min              │
│  URLs known        142           │  Last report   yesterday 08:00       │
│  Extracted          87           │  Next report   today 08:00           │
│  Pending             3           │  Report interval  daily              │
│  Budget used   $0.031 / $0.10    │                                      │
│  Directives active   4           │  DIRECTIVES (active)                 │
│                                  │  BOOST_CRAWL_PRIORITY  lesswrong.com │
│                                  │  BLACKLIST_SOURCE      substack.com  │
│                                  │  UPDATE_LINK_SCORE_HINT  arxiv.org   │
│                                  │  EXPAND_TOPIC  "mechanistic interp." │
├──────────────────────────────────┴──────────────────────────────────────┤
│  ACTIVITY LOG                                                            │
│  [14:22:01] crawl   https://arxiv.org/abs/2401.12345  → relevant        │
│  [14:22:03] extract  item #88  tokens=1240  cost=$0.0008                │
│  [14:22:05] crawl   https://lesswrong.com/posts/abc   → irrelevant      │
│  [14:22:09] link    scored 14 outbound links, 3 added to registry       │
│  [14:22:11] crawl   https://alignmentforum.org/posts/xyz → relevant     │
│  [14:22:14] extract  item #89  tokens=980   cost=$0.0006                │
│  [14:22:18] beat    next crawl batch queued (12 URLs)                   │
└─────────────────────────────────────────────────────────────────────────┘
```

**Panel descriptions:**

| Panel | Content |
|-------|---------|
| Header bar | Topic name, running status indicator (● / ⏸), wall-clock uptime, key hints |
| Overview (top-left) | URL registry size, extracted/pending item counts, slice budget consumed vs cap, active directive count |
| Schedule (top-right) | Last/next crawl timestamps, last/next report timestamps, configured report interval |
| Directives (top-right lower) | Scrollable list of active directives from `directive_store` (type + target) |
| Activity log (bottom) | Streaming log of crawl, triage, extract, link-score, and beat events — auto-scrolls, newest at bottom; proves the system is alive and making progress |

**Report notification:** when the Reporter fires on schedule, the TUI suspends the activity log and opens a full-width **Report Panel** over the bottom half — displaying the Markdown digest, flagging new sources with ★, and presenting per-item feedback prompts inline. Dismissing the panel resumes the activity log.

## Data Flow: Single Crawl Cycle

```
[Celery Beat] fires crawl job for URL X
        │
        ▼
[Crawler Engine] fetches page (httpx / curl) → extracts text + raw links
        │           (fetch always proceeds — not budget-gated)
        │
        ├──▶ [Budget Gate] → [Link Scorer (LLM)] scores each outbound link
        │         │
        │         └── high-score links → added to crawl queue (next slice if budget hit)
        │
        ▼
[Budget Gate] → [Relevance Triage (LLM)] — on-topic? → NO → discard
        │ YES
        ▼
[Budget Gate] — budget remaining for extraction?
        │                          │
        │ YES                      NO
        ▼                          ▼
[Fact Extractor (LLM)]    [Content Store] saves raw_text,
→ { extracted_json +       status=pending_extraction
    article_sum_md }       (picked up by extract_pending
        │                   task in next slice)
        ▼
[Novelty Check] — seen before? → SKIP / STORE
        │ NEW
        ▼
[Content Store] persists extracted_json + article_sum_md,
                status=extracted, reported=False (url_id FK → source)
        │
        ▼
[URL Registry] updates last_checked, adjusts next_check_at, sets is_new_source
               (is_new_source = True if this domain's url_id was never seen before)
```

## Data Flow: Report & Feedback Cycle

```
[Celery Beat] fires report generation trigger
        │
        ▼
[Reporter (LLM)] reads Content Store where reported=False AND status=extracted
        │
        ├──▶ Persists report to Reports Table (narrative_md + report_json with item IDs)
        ├──▶ Marks all included content items: reported=True
        │
        └──▶ Renders Markdown narrative via ReportChannel
                  │  MVP: Textual TUI Report Panel (overlays the running dashboard)
                  │  Future: Web dashboard / MCP server
                  │  - new-source items flagged with ★
        │
        ▼
[User — per item, all optional]
        │
        ├── judgement: "interesting" | "not interesting"
        │         │      (item_id = content_item_id from Content Store;
        │         │       url_id resolved from Content Store → URL Registry)
        │         └──▶ [Feedback Processor]
        │                   ├── writes to Feedback DB (item_id, url_id, judgement, weight_delta)
        │                   ├── updates source_weight in URL Registry via url_id
        │                   └── appends to Prompt Examples Store (capped at 20/prompt, recency-first)
        │
        └── free-text comment
                  │
                  ▼
          [Comment Interpreter (LLM)]
          parses intent → emits structured directives
                  │
                  ▼
          [Directive Store] (persisted, tagged with source item + timestamp)
                  │
                  ├──▶ BOOST_CRAWL_PRIORITY → Scheduler adjusts crawl frequency
                  ├──▶ UPDATE_LINK_SCORE_HINT → Link scorer prompt enriched
                  ├──▶ EXPAND_TOPIC → Search bootstrap re-runs for sub-topic
                  ├──▶ BLACKLIST_SOURCE → URL Registry floors source_weight
                  └──▶ SCHEDULE_DEEP_DIVE → URL check_interval overridden
```

---

## Roadmap Phases

Each phase builds on the previous, has a single focused goal, and ends with concrete acceptance criteria that can be verified manually or with a short test script before moving on.

---

### Phase 0 — Scaffold & Onboarding
**Goal**: Establish project skeleton and get the user through a working onboarding flow that produces a valid, verified configuration.

**Deliverables**:
- Project structure: `pyproject.toml`, dependency lock, linting/formatting config
- Full SQLite schema with all tables (`url_registry`, `content_items`, `feedback`, `directive_store`, `prompt_examples`, `reports`) — all with `topic_id` FK from day one
- Repository abstraction layer (SQLAlchemy 2.x) — all DB access through repository classes, no raw SQL in business logic
- `tipster init`: Step 0 (LLM provider → `.env` + LiteLLM verification loop) + Step 1 (free-text → `onboard_model` → `tipster.yaml` with inline comments)
- `tipster add-url <url>`: inserts a URL into the registry manually
- Config loading: Pydantic Settings reads `tipster.yaml` + `.env`; validates all required fields on startup

**Milestone — verified when**:
- [ ] `tipster init` completes end-to-end: `.env` written, LiteLLM verification call returns `200`, `tipster.yaml` generated with all required sections
- [ ] Re-running `tipster init` with an existing `.env` skips provider setup and goes straight to topic description
- [ ] `tipster add-url https://example.com` inserts the URL into the SQLite `url_registry` table (verify with `sqlite3 tipster.db "SELECT url FROM url_registry"`)
- [ ] Invalid or unreachable LLM config causes a clear error and re-prompt, not a crash

---

### Phase 1 — Basic Crawl & Triage
**Goal**: Fetch seed URLs on a schedule, triage each page for topic relevance, and persist on-topic raw content — end to end, no link following yet.

**Deliverables**:
- `tipster start`: loads config + directives, detects fresh-start vs. resume, seeds URL Registry from `tipster.yaml`, starts Celery Beat, launches the Textual TUI dashboard
- Textual TUI (Phase 1 scope): header bar (topic, status, uptime), Overview panel (URL count, extracted/pending items, budget used), Schedule panel (last/next crawl), Activity Log subview (streaming crawl/triage events); report panel and directive panel deferred to Phase 4
- Crawler Engine: async `httpx` fetcher with subprocess `curl` fallback; `trafilatura` for text extraction; `BeautifulSoup` for raw link extraction (links not scored yet); robots.txt + default 1s/domain delay
- Relevance Triage: LLM call (`triage_model`) against `relevance_hints`; irrelevant content discarded
- Content Store write: on-topic pages saved as `status=pending_extraction`, `reported=False`
- URL Registry update: `last_checked`, `next_check_at` (adaptive interval), `is_new_source`, `status`
- Celery Beat periodic schedule: re-crawl jobs fired based on `next_check_at`
- Internal event bus (asyncio queue or Redis pub/sub): workers emit structured log events; TUI consumes them for the Activity Log

**Milestone — verified when**:
- [ ] `tipster start` with 3 seed URLs results in all 3 appearing in `url_registry` and being fetched within the first crawl cycle
- [ ] Relevant pages appear in `content_items` with `status=pending_extraction`; irrelevant pages produce no row
- [ ] After a page returns unchanged content on re-crawl, `check_interval` doubles (verify `next_check_at` in DB)
- [ ] After a page returns changed content, `check_interval` shrinks by 25%
- [ ] A page returning 403 is marked `status=inaccessible` in `url_registry` and not retried in the same cycle

---

### Phase 2 — Extraction, Deduplication & Budget
**Goal**: Fully extract facts from triaged content, enforce the per-slice token budget, and prevent duplicate storage.

**Deliverables**:
- Fact Extractor: `extraction_model` LLM call producing `extracted_json` + `article_sum_md`; updates `status=extracted`
- `extract_pending` Celery task: runs at the start of each slice, processes all `status=pending_extraction` items before new crawl jobs
- Budget Gate: tracks tokens consumed per slice via LiteLLM cost metadata; blocks new LLM-dependent tasks when `max_tokens_per_slice` or `max_cost_per_slice_usd` is exceeded; defers to next slice
- Novelty Check: SHA-256 of normalised text; near-identical content gets `status=failed` with a `duplicate_of` reference rather than a new row
- `is_new_source` flag set on first-ever article from a domain

**Milestone — verified when**:
- [ ] After a crawl cycle, all rows in `content_items` where `status=pending_extraction` become `status=extracted` with populated `extracted_json` and `article_sum_md` in the following slice
- [ ] Re-crawling a page with identical content produces no new `content_items` row
- [ ] Publishing a test article that exceeds the token budget mid-cycle leaves the item as `status=pending_extraction`; it becomes `status=extracted` in the next slice without re-fetching
- [ ] LiteLLM cost log confirms no slice exceeds `max_cost_per_slice_usd`

---

### Phase 3 — Link Discovery
**Goal**: Grow the crawl frontier autonomously by scoring outbound links and adding high-relevance ones to the URL Registry.

**Deliverables**:
- Link Scorer: `link_score_model` LLM call per outbound link, scored 0–1 using `link_score_hints` + `relevance_hints` as context; only links ≥ `link_score_threshold` are inserted into `url_registry`
- Budget Gate applied to link scoring (same per-slice token budget shared with triage and extraction)
- Deduplication: URL normalisation before registry insert; already-known URLs update metadata but do not create duplicate rows
- `tipster add-url` remains the manual fallback for Phase 1-3 search bootstrap

**Milestone — verified when**:
- [ ] Starting from 2 seed URLs, `url_registry` contains ≥ 10 unique URLs after 2 crawl cycles (demonstrating link discovery is working)
- [ ] Links with anchor text matching `link_score_hints.negative` patterns are not added to the registry
- [ ] Re-discovering an already-known URL does not create a duplicate row
- [ ] When the slice budget is exhausted, discovered links are queued for scoring in the next slice rather than silently dropped

---

### Phase 4 — Reporting & Feedback
**Goal**: Deliver periodic intelligence digests inside the running TUI and close the feedback loop — source weights adapt and user comments become active crawl directives.

**Deliverables**:
- Reporter: `report_model` LLM call reads all `reported=False AND status=extracted` items; generates Markdown narrative + `report_json`; writes to `reports` table; marks items `reported=True`
- TUI ReportChannel (`Textual`): when a report fires, the Activity Log panel is replaced by a Report Panel overlaying the lower half of the dashboard; renders the Markdown digest with Rich markup, flags new sources with ★, presents per-item feedback prompts inline; user dismisses with a keypress to resume the Activity Log
- TUI Directives panel: the Schedule/Directives pane (top-right) now lists active directives from `directive_store`, updating live as new directives are written
- Feedback Processor: writes to `feedback` table; applies `weight_delta` to `source_weight` in `url_registry`; appends to `prompt_examples` (capped at 20/prompt, recency-first, domain-filtered)
- Comment Interpreter: `comment_model` LLM parses free-text comment → emits typed directives to `directive_store`
- Directive consumers wired: `BOOST_CRAWL_PRIORITY` (scheduler), `UPDATE_LINK_SCORE_HINT` (link scorer prompt), `EXPAND_TOPIC` (relevance_hints + TUI notice in Activity Log), `BLACKLIST_SOURCE` (url_registry), `SCHEDULE_DEEP_DIVE` (check_interval override)
- `tipster export`: merges `directive_store` into `tipster.yaml`; warns on `tipster init` re-run if active directives exist

**Milestone — verified when**:
- [ ] `tipster start` → after one report cycle, the TUI Report Panel opens automatically with ≥ 1 intel item; the dashboard remains interactive while the panel is open
- [ ] Rating an item "not interesting" decreases the originating URL's `source_weight` in `url_registry` (verify before/after in DB)
- [ ] Typing "focus more on papers from this domain" as a comment produces a `BOOST_CRAWL_PRIORITY` directive row in `directive_store` and the directive appears in the TUI Directives panel within one refresh cycle
- [ ] `BLACKLIST_SOURCE` directive prevents the domain's URLs from being crawled in the next cycle
- [ ] `tipster export` produces a `tipster.yaml` that includes the blacklist and link score hints from active directives
- [ ] New source items are visually flagged ★ in the TUI Report Panel

---

### Phase 5 — Production Hardening
**Goal**: Make the system reliable, containerised, and observable enough for continuous unattended 24/7 operation.

**Deliverables**:
- Docker + docker-compose: services for `worker` (Celery), `beat` (Celery Beat), `redis`, `db` (PostgreSQL)
- Storage swap: PostgreSQL + pgvector replaces SQLite; Repository layer requires no business logic changes (config-only switch)
- Semantic novelty check: pgvector cosine similarity (threshold 0.92) as second deduplication layer on top of SHA-256 hash
- Search engine bootstrap (SerpAPI / Brave Search API): fires when `url_registry` is empty or on `EXPAND_TOPIC` directive; `tipster start` with no seed URLs works out of the box
- Playwright fetcher plug-in: registered behind the fetcher interface; activated per-domain via config
- Retry logic: exponential back-off on transient HTTP errors (5xx, timeout); max 3 retries before `status=failed`
- Proxy support: configurable per-domain proxy via `tipster.yaml`
- Monitoring: Celery Flower (task queue visibility), structured JSON logs (crawl stats, LLM cost per slice, budget utilisation)

**Milestone — verified when**:
- [ ] `docker compose up` starts all services; `tipster start` inside the container runs without errors
- [ ] `tipster start` with empty `seed_urls` auto-discovers ≥ 5 starting URLs via search engine API within the first slice
- [ ] Switching `database.backend` from `sqlite` to `postgres` in `tipster.yaml` and restarting migrates schema with no data loss (tested with a fresh DB)
- [ ] A URL that returns 503 three times is marked `status=failed` and not retried until the next adaptive interval
- [ ] Celery Flower dashboard shows task queue depth and per-task success/failure rates

---

### Phase 6 — Advanced Features (optional)
**Goal**: Scale to multiple independent monitoring topics and expose intelligence via additional delivery channels.

**Deliverables**:
- Multi-topic: `topics` table added; `tipster topics list/add/remove` CLI sub-commands; `tipster.projects.yaml` lists per-topic config paths; Celery Beat runs isolated crawl loops with independent budgets per topic
- Trend detection: LLM-powered delta analysis between consecutive reports for the same topic — surfaces emerging themes and fading signals
- Entity graph: extracts actors, organisations, events from `extracted_json` and stores relationships for cross-source querying
- Web dashboard: FastAPI serving a local read-only report viewer
- MCP server: exposes reports and feedback as MCP tools, enabling LLM clients to query and interact with Tipster directly

**Milestone — verified when**:
- [ ] Two topics run simultaneously with separate `tipster.yaml` files and independent token budgets — neither topic's budget affects the other
- [ ] A weekly trend report identifies ≥ 1 topic that appeared in the last 3 reports but not before
- [ ] MCP server responds to a `list_reports` tool call with the last 5 report summaries

---

## Design Decisions

| # | Topic | Decision |
|---|-------|----------|
| 1 | **Crawl scope** | No hop limit. Scope is governed by the per-slice LLM token/cost budget. When the budget is exhausted, pending tasks are deferred to the next slice — the crawler can range freely within the envelope. |
| 2 | **Cost management** | Each time slice (configurable, e.g. 1 hour) carries a token budget and optional USD cap. The orchestrator tracks spend in real time via LiteLLM's cost metadata and halts LLM dispatch when the ceiling is hit. Cheap models handle bulk triage; expensive models are reserved for extraction and report synthesis. |
| 3 | **Crawler baseline** | Python `httpx` as the primary async fetcher, with subprocess `curl` as a fallback. Playwright (JS rendering) and other fetch strategies are future plug-in options behind the same fetcher interface — not in MVP scope. |
| 4 | **Persistence** | A **Repository abstraction layer** decouples business logic from the storage backend. MVP ships with SQLite. Swapping to PostgreSQL + pgvector for production requires only a config change. |
| 5 | **Report format** | Dual output per report cycle: a **Markdown/plaintext narrative** delivered to the user, and a **structured JSON record** persisted to the content store for programmatic use and future trend analysis. |
| 6 | **Report delivery** | Pluggable `ReportChannel` interface. **MVP: Textual TUI Report Panel** — when the Reporter fires, the running `tipster start` dashboard surfaces a Report Panel overlay (lower half of screen) that renders the Markdown digest with Rich markup, flags new sources with ★, and collects inline per-item feedback; dismissing the panel resumes the Activity Log. Future channels (web dashboard, MCP server) are drop-in additions with no changes to the report generation logic. |
| 7 | **Feedback UX** | Per intel item: optional **judgement** ("interesting" / "not interesting") + optional **free-text comment**. New sources are flagged ★ to prompt first ratings. Judgements adjust source weight and enrich few-shot prompts. Comments are parsed by a **Comment Interpreter LLM** that emits typed directives (`BOOST_CRAWL_PRIORITY`, `UPDATE_LINK_SCORE_HINT`, `EXPAND_TOPIC`, `BLACKLIST_SOURCE`, `SCHEDULE_DEEP_DIVE`) persisted to a directive store and consumed by the relevant subsystems on the next cycle. |
| 8 | **Onboarding** | `tipster init` accepts a free-text description (with optional inline URLs) and uses an LLM to extract topic intent, seed URLs, relevance hints, link score hints (positive/negative anchor patterns), and initial domain weights. Output is written to **`tipster.yaml`** — a human-readable, commented YAML file that is the single source of configuration truth. Users edit it directly for fine-grain control (e.g. adjusting anchor text, reweighting domains, tuning budget). `tipster init` can be re-run at any time to regenerate from a new description. |
| 9 | **Intel item granularity** | One "intel item" = one crawled article in the Content Store. Each item has a unique `content_item_id`. Reports reference items by this ID. Feedback `item_id` maps directly to `content_item_id`, and `url_id` is resolved from the Content Store to perform source weight updates. |
| 10 | **Rate limiting default** | When `robots.txt` is absent or specifies no `Crawl-delay`, a minimum inter-request delay of 1 second per domain is enforced at the fetcher level. Configurable via `crawl.default_delay_seconds` in `tipster.yaml`. |
| 11 | **Inaccessible pages** | Pages returning 401/403 or requiring login are marked `status=inaccessible` in the URL Registry. The adaptive backoff algorithm applies normally. No retry with credentials is attempted. |
| 12 | **config/directive authority** | `tipster.yaml` is the static baseline read at startup. `directive_store` holds runtime overrides with higher precedence. `tipster export` merges both into a clean snapshot. Re-running `tipster init` warns about active directives before overwriting. |
| 13 | **Budget exhaustion during extraction** | **Defer extraction (Option B)**: when the token budget is exhausted after triage but before extraction, raw text is saved to the Content Store with `status=pending_extraction`. A dedicated `extract_pending` Celery task runs at the start of each new slice, processing deferred items before new crawl jobs are dispatched. No content fetched and triaged as relevant is ever discarded. |
| 14 | **Report time window** | **Unread flag (Option C)**: Content Store carries a `reported` boolean (default `False`). The Reporter reads all items where `reported=False AND status=extracted`. After persisting the report, all included items are marked `reported=True`. This guarantees exactly-once inclusion per item across report cycles, regardless of system pauses or schedule changes. |
| 15 | **LLM provider setup** | `tipster init` runs a **Step 0** before any topic configuration: collects provider, API base URL, API key, and model name interactively. Credentials are written to **`.env`** (masked input; file auto-added to `.gitignore`). `tipster.yaml` references only env var names. A **verification call** (`"Reply with OK"`) is made via LiteLLM before proceeding — onboarding loops on Step 0 until the call succeeds. |