"""SQLAlchemy and Pydantic models for Digital Archeology"""
from datetime import datetime
from typing import Optional, List
from enum import Enum

from sqlalchemy import (
    Column, Integer, String, Text, DateTime, Float, Boolean, ForeignKey, JSON, Index,
    Enum as SQLEnum,
)
from sqlalchemy.orm import relationship
from pydantic import BaseModel, Field, field_validator

from .database import Base


# ===== ENUMS =====
class ProjectStatus(str, Enum):
    CREATED = "created"
    FETCHING_URLS = "fetching_urls"
    URLS_READY = "urls_ready"
    DOWNLOADING = "downloading"
    DOWNLOAD_COMPLETE = "download_complete"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"


class FileStatus(str, Enum):
    PENDING = "pending"
    DOWNLOADED = "downloaded"
    PROCESSING = "processing"
    PROCESSED = "processed"
    ERROR = "error"
    SKIPPED = "skipped"


# ===== SQLAlchemy Models =====
class Project(Base):
    __tablename__ = "projects"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False)
    domain = Column(String(255), nullable=False)
    status = Column(SQLEnum(ProjectStatus), default=ProjectStatus.CREATED)

    # Wayback machine info
    total_snapshots = Column(Integer, default=0)
    date_range_start = Column(String(14))  # YYYYMMDDHHMMSS format
    date_range_end = Column(String(14))

    # Filter settings
    filter_date_start = Column(String(14))
    filter_date_end = Column(String(14))
    filter_frequency = Column(String(50))  # all, yearly, monthly, weekly, daily
    filter_mimetypes = Column(JSON)  # ["text/html", "text/css", etc.]

    # Progress tracking
    files_total = Column(Integer, default=0)
    files_downloaded = Column(Integer, default=0)
    files_processed = Column(Integer, default=0)

    # Ollama settings
    ollama_model = Column(String(100), default="llama3.2:3b")

    # Analysis scope.
    #
    # A domain's capture history is not necessarily one site. An expired domain
    # that gets re-registered keeps its URL but has nothing to do with what came
    # before: b1mas4.com was a Tijuana news site through 2015, then sat parked,
    # then hosted a pharmacy-spam farm from 2019 (9,381 captures -- 75% of that
    # project's corpus) and a gambling affiliate from 2022.
    #
    # Analyzing all of it as one site is wrong twice over: it spends the LLM
    # budget on generated spam, and it mixes that spam into the people, tags and
    # narrative summary that describe the real site. These bounds mark which era
    # is the subject of study; captures outside the window are kept (they are
    # evidence of the takeover) but excluded from analysis and aggregates.
    #
    # Half-open [scope_start, scope_end) over the Wayback YYYYMMDDHHMMSS
    # timestamp. NULL on either side means unbounded; both NULL means the whole
    # capture history is in scope, which is the default for every project.
    scope_start = Column(String(14))
    scope_end = Column(String(14))
    scope_reason = Column(Text)
    # URL patterns kept out of the corpus, one per line (services/exclusions.py).
    exclude_patterns = Column(Text)
    # The project whose outbound links this one was started from (linked_sites).
    source_project_id = Column(Integer, ForeignKey("projects.id"), nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    files = relationship("ArchivedFile", back_populates="project", cascade="all, delete-orphan")
    card = relationship(
        "ArchivalCard", back_populates="project", uselist=False, cascade="all, delete-orphan"
    )


class ArchivedFile(Base):
    __tablename__ = "archived_files"
    # Speeds up the per-project COUNT(...)-by-status queries used all over the app
    # (list page, download/processing progress). Important when the DB is on slow
    # storage (e.g. an NFS-mounted volume) and the table has tens of thousands of rows.
    __table_args__ = (
        Index("ix_archived_files_project_status", "project_id", "status"),
        # Every analysis and results query filters on in_scope as well.
        Index("ix_archived_files_project_scope", "project_id", "in_scope", "status"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False)

    # File info
    original_url = Column(Text, nullable=False)
    wayback_url = Column(Text)
    timestamp = Column(String(14))  # Wayback timestamp YYYYMMDDHHMMSS
    mimetype = Column(String(100))
    status_code = Column(String(10))

    # Local storage
    local_path = Column(Text)
    file_size = Column(Integer)

    # Processing status
    status = Column(SQLEnum(FileStatus), default=FileStatus.PENDING)
    error_message = Column(Text)
    # The model's reply when analysis failed (first 4,000 chars), so the cause
    # is visible; cleared once the file is analyzed.
    last_raw_response = Column(Text)

    # Whether this capture falls inside the project's scope window (see
    # Project.scope_start). Materialized rather than derived from `timestamp` on
    # every query: it is filtered in roughly a dozen places, several of them raw
    # SQL, and it keeps the index above usable. Recomputed by
    # projects.apply_scope() whenever the window changes.
    in_scope = Column(Boolean, nullable=False, default=True, server_default="1")

    # Timestamps
    downloaded_at = Column(DateTime)
    processed_at = Column(DateTime)

    # Relationships
    project = relationship("Project", back_populates="files")
    # foreign_keys is required because FileAnalysis has two FKs to this table
    # (file_id and source_file_id); without it the mapper cannot pick one.
    analysis = relationship(
        "FileAnalysis",
        back_populates="file",
        uselist=False,
        cascade="all, delete-orphan",
        foreign_keys="FileAnalysis.file_id",
    )


class FileAnalysis(Base):
    __tablename__ = "file_analyses"

    id = Column(Integer, primary_key=True, autoincrement=True)
    file_id = Column(Integer, ForeignKey("archived_files.id"), nullable=False, unique=True)

    # When several captures of the same URL exist, only one is sent to the LLM and
    # its result is copied to the siblings. NULL means this row was analyzed directly;
    # non-NULL points at the capture that actually was.
    #
    # IMPORTANT: aggregate statistics must filter on `source_file_id IS NULL`, or every
    # tag/person/sentiment count is inflated by the duplicate captures (~1.5x here).
    source_file_id = Column(Integer, ForeignKey("archived_files.id"), nullable=True)

    # LLM extracted fields
    file_path = Column(Text)
    estimated_date = Column(String(50))  # LLM's guess at original date
    date_confidence = Column(String(20))  # high, medium, low, unknown

    # People and interactions
    people_mentioned = Column(JSON)  # ["name1", "name2", ...]
    interaction_type = Column(String(100))  # discussion, announcement, article, etc.

    # Content analysis
    subjects = Column(JSON)  # ["topic1", "topic2", ...]
    sentiment = Column(String(50))  # positive, negative, neutral, mixed
    sentiment_score = Column(Float)  # -1.0 to 1.0

    # Classification
    content_type = Column(String(100))  # news, economy, discussion, blog, forum, etc.
    tags = Column(JSON)  # ["tag1", "tag2", ...]

    # Full description
    description = Column(Text)

    # Processing metadata
    processing_time_seconds = Column(Float)
    tokens_used = Column(Integer)      # output tokens (eval_count)
    prompt_tokens = Column(Integer)    # input tokens (prompt_eval_count); feeds the cost estimate
    model_used = Column(String(100))
    raw_response = Column(Text)  # Store raw LLM response for debugging

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    file = relationship(
        "ArchivedFile", back_populates="analysis", foreign_keys=[file_id]
    )


class ProcessingRun(Base):
    """One LLM analysis run over a project.

    Progress used to live in a dict on the service singleton, so a restart lost
    it and the UI silently showed zeros for a job that was still queued up. This
    table is the single source of truth instead: the worker heartbeats onto it,
    and the HTMX poll reads it, which means progress survives both a page reload
    and an app restart.
    """
    __tablename__ = "processing_runs"
    __table_args__ = (
        Index("ix_processing_runs_project", "project_id", "id"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False)

    # running | completed | cancelled | interrupted | aborted
    status = Column(String(20), default="running")
    model_used = Column(String(100))
    concurrency = Column(Integer, default=1)

    # Counters
    claimed = Column(Integer, default=0)
    analyzed = Column(Integer, default=0)
    copied = Column(Integer, default=0)   # duplicate captures given a copied analysis
    skipped = Column(Integer, default=0)
    errors = Column(Integer, default=0)

    current_url = Column(Text)
    last_error = Column(Text)  # already redacted before it gets here

    started_at = Column(DateTime, default=datetime.utcnow)
    finished_at = Column(DateTime)
    heartbeat_at = Column(DateTime, default=datetime.utcnow)


class ArchivalCard(Base):
    """Aggregated 'archival card' for a domain: ownership, tech stack, timeline, summary."""
    __tablename__ = "archival_cards"

    id = Column(Integer, primary_key=True, autoincrement=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False, unique=True)

    # WHOIS / RDAP ownership (free sources: current RDAP)
    registrar = Column(String(255))
    whois_created = Column(String(50))
    whois_expires = Column(String(50))
    whois_updated = Column(String(50))
    nameservers = Column(JSON)  # ["ns1.example.com", ...]
    whois_status = Column(JSON)  # ["clientTransferProhibited", ...]
    registrant = Column(JSON)  # {name, org, email, ...} when not redacted
    whois_raw = Column(JSON)  # full RDAP response for reference
    whois_source = Column(String(50))  # rdap, unregistered, unavailable, error
    whois_domain = Column(String(255))  # domain looked up (registered parent of a subdomain)

    # Technology stack detected from archived HTML
    tech_stack = Column(JSON)  # [{name, category, version, evidence}, ...]

    # Wayback timeline aggregated from archived file timestamps
    timeline = Column(JSON)  # {first_seen, last_seen, per_year: {...}, went_dark}

    # People surfaced from content analysis (likely owners/contributors)
    likely_people = Column(JSON)  # [{name, mentions}, ...]

    # LLM narrative synthesis
    llm_summary = Column(Text)
    llm_model_used = Column(String(100))

    # Evidence-backed findings: what the site was, its purpose, how it changed,
    # who ran it, whether it achieved what it set out to do. Unlike llm_summary,
    # every claim carries the ids of the pages it rests on (see
    # services/findings.py), so a historian can check it rather than trust it.
    findings = Column(JSON)
    findings_model = Column(String(100))
    findings_generated_at = Column(DateTime)

    # Timestamps
    generated_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    project = relationship("Project", back_populates="card")


class LinkTarget(Base):
    """A URL the studied site linked to, once per project.

    The site's own pages only tell half the story: what a news site cited, who
    it promoted, which collaborators it pointed at. Links are extracted from the
    in-scope captures without any LLM call (services/links.py), classified by
    heuristics, and the ones worth reading are recovered from the Wayback
    Machine at the capture closest to when they were first cited -- a 2013
    article should be read as it looked in 2013, not as the domain looks now.
    """
    __tablename__ = "link_targets"
    __table_args__ = (
        Index("ux_link_targets_project_key", "project_id", "url_key", unique=True),
        Index("ix_link_targets_project_recovery", "project_id", "recovery_status"),
        Index("ix_link_targets_project_analysis", "project_id", "analysis_status"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False)

    url = Column(Text, nullable=False)
    # Normalized for dedupe: no scheme, no www., no default port, no fragment.
    url_key = Column(Text, nullable=False)
    domain = Column(String(255))
    is_internal = Column(Boolean, nullable=False, default=False)
    # article | taxonomy | navigation | sitewide | social | newsletter |
    # advertising | reference | media | document | download | other
    kind = Column(String(20))

    # Internal links already captured in the corpus need no recovery.
    corpus_file_id = Column(Integer, ForeignKey("archived_files.id"), nullable=True)

    cited_by_pages = Column(Integer, default=0)   # distinct source captures
    first_cited_ts = Column(String(14))
    last_cited_ts = Column(String(14))
    sample_anchor = Column(Text)

    # pending | recovered | not_archived | error | in_corpus | not_selected
    recovery_status = Column(String(20), default="not_selected")
    # Which archive the copy came from (NULL = the Wayback Machine) and when
    # other archives were last asked for a not_archived target (services/memento.py).
    source_archive = Column(String(60))
    memento_checked_at = Column(DateTime)
    wayback_timestamp = Column(String(14))
    wayback_url = Column(Text)
    # Days between the first citation and the capture used. Small is faithful;
    # a capture years away may show a different page than the one cited.
    capture_distance_days = Column(Integer)
    mimetype = Column(String(100))
    local_path = Column(Text)
    file_size = Column(Integer)
    error_message = Column(Text)
    attempts = Column(Integer, default=0)
    recovered_at = Column(DateTime)

    # LLM classification of the recovered page.
    # pending | processed | error | skipped | not_selected
    analysis_status = Column(String(20), default="not_selected")
    title = Column(Text)
    description = Column(Text)
    content_type = Column(String(100))
    sentiment = Column(String(50))
    sentiment_score = Column(Float)
    subjects = Column(JSON)
    tags = Column(JSON)
    people_mentioned = Column(JSON)
    # How the linked material relates to the studied site (see links.RELATIONS).
    relation = Column(String(40))
    relation_note = Column(Text)
    model_used = Column(String(100))
    raw_response = Column(Text)
    analyzed_at = Column(DateTime)


class PageLink(Base):
    """One citation: a captured page of the site linking to a LinkTarget.

    Kept per capture rather than per URL because a homepage links to different
    articles every day; the capture's timestamp is what dates the citation.
    """
    __tablename__ = "page_links"
    __table_args__ = (
        Index("ux_page_links_source_target", "source_file_id", "target_id", unique=True),
        Index("ix_page_links_target", "target_id"),
        Index("ix_page_links_project", "project_id"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False)
    source_file_id = Column(Integer, ForeignKey("archived_files.id"), nullable=False)
    target_id = Column(Integer, ForeignKey("link_targets.id"), nullable=False)
    anchor_text = Column(Text)


class PageImage(Base):
    """An image shown on a page that has no text of its own.

    Early sites (Revista Espiral, 2002) carried whole sections as photo pages:
    an <img> and nothing else. Analysis skips those as "too little content",
    which is right for text but loses the page. This records each image, whether
    the archive kept it, and what a vision model sees in it.
    """
    __tablename__ = "page_images"
    __table_args__ = (
        Index("ux_page_images_file_src", "file_id", "src_url", unique=True),
        Index("ix_page_images_project", "project_id"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False)
    file_id = Column(Integer, ForeignKey("archived_files.id"), nullable=False)
    src_url = Column(Text, nullable=False)       # resolved against the page URL
    alt_text = Column(Text)
    # pending | recovered | not_archived | skipped | error
    recovery_status = Column(String(20), default="pending")
    source_archive = Column(String(60))          # NULL = the Wayback Machine
    memento_checked_at = Column(DateTime)
    wayback_timestamp = Column(String(14))
    wayback_url = Column(Text)
    mimetype = Column(String(100))
    local_path = Column(Text)
    file_size = Column(Integer)
    error_message = Column(Text)
    attempts = Column(Integer, default=0)
    # not_selected | pending | processed | skipped | error
    description_status = Column(String(20), default="not_selected")
    description = Column(Text)
    visible_text = Column(Text)
    model_used = Column(String(100))
    tokens_used = Column(Integer)
    raw_response = Column(Text)
    described_at = Column(DateTime)


class PageReading(Base):
    """A close reading of one article: the apparatus a historian works from.

    FileAnalysis answers "what kind of page is this and what is it about" across
    the whole corpus. That is triage, and it is what the site-wide aggregates
    are built from. This table answers the questions asked of a *document*:
    who acts in it, where it happens, when the events it reports took place (as
    distinct from when the page was published), what it asserts, and what it is
    trying to do.

    It is deliberately a separate table and a separate pass:

    - It runs only on article-shaped pages. Tag listings and pagination have no
      author, no place and no claim; close-reading them wastes budget and fills
      the aggregates with noise.
    - Its cost is per-document and its schema will keep changing as the
      questions sharpen. Re-running it must never put the triage analyses --
      which cost the whole corpus budget -- at risk.

    Duplicate captures are handled exactly as in FileAnalysis: one capture per
    URL is read, and ``source_file_id`` points at it from the siblings. Every
    aggregate MUST filter on ``source_file_id IS NULL``.
    """
    __tablename__ = "page_readings"
    __table_args__ = (
        Index("ix_page_readings_project", "project_id", "status"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False)
    file_id = Column(Integer, ForeignKey("archived_files.id"), nullable=False, unique=True)
    source_file_id = Column(Integer, ForeignKey("archived_files.id"), nullable=True)

    # pending | processed | error | skipped
    status = Column(String(20), default="pending")
    error_message = Column(Text)
    attempts = Column(Integer, default=0)

    title = Column(Text)
    # Journalistic/literary form (see reading.GENRES). Genre is evidence in its
    # own right: a site that starts in nota informativa and ends in cronica and
    # resena has changed what it is for.
    genre = Column(String(30))
    # Primary register of purpose (see reading.INTENTS): political, cultural...
    intent = Column(String(30))
    secondary_intents = Column(JSON)
    # Posture toward the subject (see reading.STANCES).
    stance = Column(String(30))

    # [{name, role: author|quoted|subject|mentioned, affiliation, note}]
    # The role is the point. Who is quoted and who is merely written about is a
    # measurable claim about whom a publication treats as a source.
    people = Column(JSON)
    # [{name, kind: city|state|country|region|neighborhood|venue|other, note}]
    places = Column(JSON)
    # [{name, kind: government|party|university|company|media|ngo|...}]
    organizations = Column(JSON)
    # [{what, when, where}] -- when the reported events happened.
    events = Column(JSON)
    # Verifiable assertions the page makes, in its own terms.
    facts = Column(JSON)
    # How the subject is construed ("impunidad", "identidad fronteriza"...).
    frames = Column(JSON)
    # [{speaker, text}] -- direct speech actually printed on the page.
    quotes = Column(JSON)

    # The page's own time. ``published_estimate`` is when it went out;
    # period_start/end bound what it talks about, which is frequently earlier
    # and is the date a historian actually needs.
    published_estimate = Column(String(50))
    period_start = Column(String(50))
    period_end = Column(String(50))
    # current_event | retrospective | forecast | timeless
    temporal_mode = Column(String(20))

    summary = Column(Text)

    model_used = Column(String(100))
    tokens_used = Column(Integer)
    raw_response = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)


class PageComment(Base):
    """One reader comment, read out of the page's own markup.

    The audience is the half of a publication that its own pages record and
    almost every web archive discards. Comments survive in the capture as
    ordinary markup, so they cost nothing to recover and nothing to verify:
    anyone can open the capture and check.

    Rows are rebuilt from scratch on every extraction run and deduplicated on
    ``(page_url, comment_ref)`` -- WordPress's own comment id -- because a
    thread captured five times is one thread. They are read from the latest
    in-scope capture of each URL, since comments accumulate after publication
    while the article text does not (see services/reception.py).
    """
    __tablename__ = "page_comments"
    __table_args__ = (
        # Keyed on the *normalized* page, not the raw URL. One article is
        # captured under several spellings -- with and without a trailing
        # slash, and with WordPress's own ?replytocom= links -- and keying on
        # the raw URL let the same comment in under each of them, inflating
        # every reception figure by a third.
        Index("ux_page_comments_page_ref", "project_id", "page_key", "comment_ref", unique=True),
        Index("ix_page_comments_project", "project_id"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False)
    # The capture the comment was read from, for citation.
    file_id = Column(Integer, ForeignKey("archived_files.id"), nullable=False)
    page_url = Column(Text, nullable=False)
    # page_url with scheme, www, default port, query and trailing slash removed.
    page_key = Column(Text, nullable=False)
    capture_timestamp = Column(String(14))

    comment_ref = Column(String(40), nullable=False)
    author = Column(Text)
    body = Column(Text)
    # Themes print "8 days ago" rather than a date; kept verbatim alongside the
    # date derived from it, so the derivation can be checked or redone.
    time_label = Column(Text)
    commented_on = Column(String(10))
    depth = Column(Integer, default=1)          # 1 = top level, 2+ = a reply
    by_site = Column(Boolean, default=False)    # the publication answering a reader


class PersonAlias(Base):
    """A historian's decision about two spellings of a name (services/people.py).

    kind 'merge': alias_key is the same person as canonical_key.
    kind 'separate': the two are different people, whatever the automatic
    rule thinks. Keys are folded names (people.key), not display spellings.
    """
    __tablename__ = "person_aliases"
    __table_args__ = (
        Index("ux_person_aliases", "project_id", "alias_key", "canonical_key", "kind", unique=True),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False)
    alias_key = Column(String(255), nullable=False)
    canonical_key = Column(String(255), nullable=False)
    kind = Column(String(10), nullable=False, default="merge")


# ===== Pydantic Schemas =====

# Project schemas
class ProjectCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    domain: str = Field(..., min_length=1, max_length=255)
    ollama_model: str = Field(default="llama3.2:3b")


class ProjectFilterSettings(BaseModel):
    filter_date_start: Optional[str] = None
    filter_date_end: Optional[str] = None
    filter_frequency: str = Field(default="all")  # all, yearly, monthly, weekly, daily
    filter_mimetypes: List[str] = Field(default=["text/html"])


class ProjectResponse(BaseModel):
    id: int
    name: str
    domain: str
    status: ProjectStatus
    total_snapshots: int
    date_range_start: Optional[str]
    date_range_end: Optional[str]
    filter_date_start: Optional[str]
    filter_date_end: Optional[str]
    filter_frequency: Optional[str]
    filter_mimetypes: Optional[List[str]]
    files_total: int
    files_downloaded: int
    files_processed: int
    ollama_model: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class ProjectProgress(BaseModel):
    status: ProjectStatus
    files_total: int
    files_downloaded: int
    files_processed: int
    current_operation: Optional[str] = None
    error_message: Optional[str] = None


# File schemas
class FileResponse(BaseModel):
    id: int
    original_url: str
    wayback_url: Optional[str]
    timestamp: Optional[str]
    mimetype: Optional[str]
    status: FileStatus
    file_size: Optional[int]
    downloaded_at: Optional[datetime]
    processed_at: Optional[datetime]
    error_message: Optional[str]

    class Config:
        from_attributes = True


# Analysis schemas
class AnalysisResponse(BaseModel):
    id: int
    file_id: int
    file_path: Optional[str]
    estimated_date: Optional[str]
    date_confidence: Optional[str]
    people_mentioned: Optional[List[str]]
    interaction_type: Optional[str]
    subjects: Optional[List[str]]
    sentiment: Optional[str]
    sentiment_score: Optional[float]
    content_type: Optional[str]
    tags: Optional[List[str]]
    description: Optional[str]
    model_used: Optional[str]
    created_at: datetime

    # Older rows stored people/subjects/tags as objects ({"name": ...}) because
    # nothing normalized the model output. Coerce on read so those rows do not
    # fail validation; new rows are normalized before they are written.
    @field_validator("people_mentioned", "subjects", "tags", mode="before")
    @classmethod
    def _coerce_str_list(cls, value):
        if value is None or isinstance(value, str):
            return value
        if not isinstance(value, (list, tuple)):
            return value
        out = []
        for item in value:
            if isinstance(item, dict):
                item = item.get("name") or item.get("person") or item.get("title") or ""
            text = str(item).strip()
            if text:
                out.append(text)
        return out

    class Config:
        from_attributes = True


class FileWithAnalysis(BaseModel):
    file: FileResponse
    analysis: Optional[AnalysisResponse]


# Search schemas
class SearchQuery(BaseModel):
    query: Optional[str] = None
    people: Optional[List[str]] = None
    subjects: Optional[List[str]] = None
    sentiment: Optional[str] = None
    content_type: Optional[str] = None
    tags: Optional[List[str]] = None
    date_start: Optional[str] = None
    date_end: Optional[str] = None
    page: int = Field(default=1, ge=1)
    per_page: int = Field(default=20, ge=1, le=100)


class SearchResults(BaseModel):
    total: int
    page: int
    per_page: int
    pages: int
    results: List[FileWithAnalysis]


# Wayback schemas
class WaybackSnapshot(BaseModel):
    timestamp: str
    original_url: str
    mimetype: str
    status_code: str
    digest: str
    length: int


class WaybackSearchResult(BaseModel):
    domain: str
    total_snapshots: int
    date_range_start: Optional[str]
    date_range_end: Optional[str]
    sample_urls: List[str]
    mimetypes: List[str]


# Archival card schemas
class ArchivalCardResponse(BaseModel):
    id: int
    project_id: int
    registrar: Optional[str]
    whois_created: Optional[str]
    whois_expires: Optional[str]
    whois_updated: Optional[str]
    nameservers: Optional[List[str]]
    whois_status: Optional[List[str]]
    registrant: Optional[dict]
    whois_source: Optional[str]
    whois_domain: Optional[str] = None
    tech_stack: Optional[List[dict]]
    timeline: Optional[dict]
    likely_people: Optional[List[dict]]
    llm_summary: Optional[str]
    llm_model_used: Optional[str]
    generated_at: Optional[datetime]

    class Config:
        from_attributes = True
