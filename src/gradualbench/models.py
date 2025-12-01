"""Database models for GradualBench."""

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Session, relationship, sessionmaker


def utcnow():
    """Return current UTC datetime."""
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    """Base class for all models."""

    pass


class Dataset(Base):
    """Represents a dataset that has been ingested."""

    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True)
    name = Column(String(255), unique=True, nullable=False)
    source = Column(String(512), nullable=False)  # e.g., "huggingface:dataset_name"
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=utcnow)

    prompts = relationship("Prompt", back_populates="dataset", cascade="all, delete-orphan")


class Prompt(Base):
    """Represents a single prompt/question from a dataset."""

    __tablename__ = "prompts"

    id = Column(Integer, primary_key=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=False)
    external_id = Column(String(255), nullable=True)  # ID from original dataset
    content = Column(Text, nullable=False)
    expected_completion = Column(Text, nullable=True)  # Ground truth/reference answer
    metadata_json = Column(Text, nullable=True)  # Additional metadata as JSON
    created_at = Column(DateTime, default=utcnow)

    dataset = relationship("Dataset", back_populates="prompts")
    completions = relationship("Completion", back_populates="prompt", cascade="all, delete-orphan")


class Model(Base):
    """Represents a model or checkpoint used for inference."""

    __tablename__ = "models"

    id = Column(Integer, primary_key=True)
    name = Column(String(255), unique=True, nullable=False)
    endpoint = Column(String(512), nullable=True)  # API endpoint if different from default
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=utcnow)

    completions = relationship("Completion", back_populates="model", cascade="all, delete-orphan")


class Completion(Base):
    """Represents a model's completion/response for a prompt."""

    __tablename__ = "completions"

    id = Column(Integer, primary_key=True)
    prompt_id = Column(Integer, ForeignKey("prompts.id"), nullable=False)
    model_id = Column(Integer, ForeignKey("models.id"), nullable=False)
    content = Column(Text, nullable=False)
    tokens_used = Column(Integer, nullable=True)
    latency_ms = Column(Float, nullable=True)
    metadata_json = Column(Text, nullable=True)  # Additional response metadata
    created_at = Column(DateTime, default=utcnow)

    prompt = relationship("Prompt", back_populates="completions")
    model = relationship("Model", back_populates="completions")
    evaluations = relationship("Evaluation", back_populates="completion", cascade="all, delete-orphan")


class Judge(Base):
    """Represents a judge model/configuration used for evaluation."""

    __tablename__ = "judges"

    id = Column(Integer, primary_key=True)
    name = Column(String(255), unique=True, nullable=False)
    model_name = Column(String(255), nullable=False)  # e.g., "gpt-4"
    prompt_template = Column(Text, nullable=True)  # Custom evaluation prompt
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=utcnow)

    evaluations = relationship("Evaluation", back_populates="judge", cascade="all, delete-orphan")


class Evaluation(Base):
    """Represents an evaluation result for a completion."""

    __tablename__ = "evaluations"

    id = Column(Integer, primary_key=True)
    completion_id = Column(Integer, ForeignKey("completions.id"), nullable=False)
    judge_id = Column(Integer, ForeignKey("judges.id"), nullable=False)
    score = Column(Float, nullable=True)  # Numeric score if applicable
    result = Column(Text, nullable=True)  # Full evaluation result/explanation
    passed = Column(Integer, nullable=True)  # 1 for pass, 0 for fail, null if not binary
    metadata_json = Column(Text, nullable=True)
    created_at = Column(DateTime, default=utcnow)

    completion = relationship("Completion", back_populates="evaluations")
    judge = relationship("Judge", back_populates="evaluations")


def get_engine(database_url: str = "sqlite:///gradualbench.db"):
    """Create and return a database engine."""
    return create_engine(database_url, echo=False)


def init_db(database_url: str = "sqlite:///gradualbench.db"):
    """Initialize the database with all tables."""
    engine = get_engine(database_url)
    Base.metadata.create_all(engine)
    return engine


def get_session(engine) -> Session:
    """Create and return a database session."""
    SessionLocal = sessionmaker(bind=engine)
    return SessionLocal()
