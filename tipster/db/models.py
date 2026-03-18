"""SQLAlchemy ORM models for Tipster."""

from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, relationship


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Topic(Base):
    """Represents a monitored topic (supports multi-topic in Phase 6+)."""

    __tablename__ = "topics"

    topic_id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=_now, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)

    urls = relationship("UrlRegistry", back_populates="topic", cascade="all, delete-orphan")
    content_items = relationship("ContentItem", back_populates="topic", cascade="all, delete-orphan")
    feedback = relationship("Feedback", back_populates="topic", cascade="all, delete-orphan")
    directives = relationship("Directive", back_populates="topic", cascade="all, delete-orphan")
    reports = relationship("Report", back_populates="topic", cascade="all, delete-orphan")
    prompt_examples = relationship("PromptExample", back_populates="topic", cascade="all, delete-orphan")


class UrlRegistry(Base):
    """Registry of all known URLs and their crawl metadata."""

    __tablename__ = "url_registry"

    url_id = Column(Integer, primary_key=True, autoincrement=True)
    topic_id = Column(Integer, ForeignKey("topics.topic_id"), nullable=False, index=True)
    url = Column(Text, nullable=False, unique=True)
    domain = Column(String(255), nullable=False, index=True)
    relevance_score = Column(Float, default=0.5, nullable=False)
    source_weight = Column(Float, default=0.5, nullable=False)
    is_new_source = Column(Boolean, default=False, nullable=False)
    last_checked = Column(DateTime, nullable=True)
    check_interval = Column(Integer, default=3600, nullable=False)  # seconds
    next_check_at = Column(DateTime, nullable=True)
    status = Column(String(50), default="pending", nullable=False)  # pending | active | inaccessible | blacklisted
    added_at = Column(DateTime, default=_now, nullable=False)
    added_by = Column(String(50), default="manual", nullable=False)  # manual | discovery | seed

    topic = relationship("Topic", back_populates="urls")
    content_items = relationship("ContentItem", back_populates="url_entry", cascade="all, delete-orphan")
    feedback = relationship("Feedback", back_populates="url_entry")


class ContentItem(Base):
    """Crawled and (optionally) extracted content from a URL."""

    __tablename__ = "content_items"

    item_id = Column(Integer, primary_key=True, autoincrement=True)
    topic_id = Column(Integer, ForeignKey("topics.topic_id"), nullable=False, index=True)
    url_id = Column(Integer, ForeignKey("url_registry.url_id"), nullable=False, index=True)
    content_hash = Column(String(64), nullable=False, index=True)  # SHA-256 of normalised text
    raw_text = Column(Text, nullable=True)
    extracted_json = Column(Text, nullable=True)   # JSON blob of extracted facts
    article_sum_md = Column(Text, nullable=True)   # per-article LLM summary (Markdown)
    topic_score = Column(Float, nullable=True)     # 0-1 relevance score from triage
    crawled_at = Column(DateTime, default=_now, nullable=False)
    status = Column(
        String(50), default="pending_extraction", nullable=False
    )  # pending_extraction | extracted | failed
    reported = Column(Boolean, default=False, nullable=False)
    is_new_source = Column(Boolean, default=False, nullable=False)
    duplicate_of = Column(Integer, ForeignKey("content_items.item_id"), nullable=True)

    topic = relationship("Topic", back_populates="content_items")
    url_entry = relationship("UrlRegistry", back_populates="content_items")
    feedback = relationship("Feedback", back_populates="content_item")


class Feedback(Base):
    """User feedback on content items or sources."""

    __tablename__ = "feedback"

    feedback_id = Column(Integer, primary_key=True, autoincrement=True)
    topic_id = Column(Integer, ForeignKey("topics.topic_id"), nullable=False, index=True)
    url_id = Column(Integer, ForeignKey("url_registry.url_id"), nullable=True)
    item_id = Column(Integer, ForeignKey("content_items.item_id"), nullable=True)
    judgement = Column(Integer, nullable=True)       # +1 (relevant) / -1 (not relevant)
    weight_delta = Column(Float, nullable=True)
    comment = Column(Text, nullable=True)
    created_at = Column(DateTime, default=_now, nullable=False)
    processed = Column(Boolean, default=False, nullable=False)

    topic = relationship("Topic", back_populates="feedback")
    url_entry = relationship("UrlRegistry", back_populates="feedback")
    content_item = relationship("ContentItem", back_populates="feedback")


class Directive(Base):
    """Runtime directives applied to the crawler (from feedback processing)."""

    __tablename__ = "directive_store"

    directive_id = Column(Integer, primary_key=True, autoincrement=True)
    topic_id = Column(Integer, ForeignKey("topics.topic_id"), nullable=False, index=True)
    directive_type = Column(String(100), nullable=False)
    # e.g. BOOST_CRAWL_PRIORITY, BLACKLIST_SOURCE, EXPAND_TOPIC, UPDATE_LINK_SCORE_HINT, SCHEDULE_DEEP_DIVE
    target = Column(Text, nullable=True)
    params_json = Column(Text, nullable=True)
    created_at = Column(DateTime, default=_now, nullable=False)
    expires_at = Column(DateTime, nullable=True)
    applied = Column(Boolean, default=False, nullable=False)

    topic = relationship("Topic", back_populates="directives")


class Report(Base):
    """Generated reports (digests) for a topic."""

    __tablename__ = "reports"

    report_id = Column(Integer, primary_key=True, autoincrement=True)
    topic_id = Column(Integer, ForeignKey("topics.topic_id"), nullable=False, index=True)
    generated_at = Column(DateTime, default=_now, nullable=False)
    narrative_md = Column(Text, nullable=True)
    report_json = Column(Text, nullable=True)

    topic = relationship("Topic", back_populates="reports")


class PromptExample(Base):
    """Few-shot examples for LLM prompts, accumulated from feedback."""

    __tablename__ = "prompt_examples"

    example_id = Column(Integer, primary_key=True, autoincrement=True)
    topic_id = Column(Integer, ForeignKey("topics.topic_id"), nullable=False, index=True)
    domain = Column(String(255), nullable=True)
    content_snippet = Column(Text, nullable=False)
    judgement = Column(Integer, nullable=True)   # +1 / -1
    label = Column(String(100), nullable=True)
    created_at = Column(DateTime, default=_now, nullable=False)

    topic = relationship("Topic", back_populates="prompt_examples")
