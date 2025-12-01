"""Tests for database models."""

import pytest
from sqlalchemy import create_engine

from gradualbench.models import (
    Base,
    Completion,
    Dataset,
    Evaluation,
    Judge,
    Model,
    Prompt,
    get_session,
    init_db,
)


@pytest.fixture
def db_session():
    """Create an in-memory database session for testing."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = get_session(engine)
    yield session
    session.close()


def test_create_dataset(db_session):
    """Test creating a dataset."""
    dataset = Dataset(
        name="test_dataset",
        source="huggingface:test",
        description="Test dataset",
    )
    db_session.add(dataset)
    db_session.commit()
    
    assert dataset.id is not None
    assert dataset.name == "test_dataset"


def test_create_prompt(db_session):
    """Test creating a prompt."""
    dataset = Dataset(name="test_dataset", source="test")
    db_session.add(dataset)
    db_session.commit()
    
    prompt = Prompt(
        dataset_id=dataset.id,
        content="What is 2+2?",
        expected_completion="4",
    )
    db_session.add(prompt)
    db_session.commit()
    
    assert prompt.id is not None
    assert prompt.dataset.name == "test_dataset"


def test_create_model(db_session):
    """Test creating a model."""
    model = Model(name="gpt-4", description="Test model")
    db_session.add(model)
    db_session.commit()
    
    assert model.id is not None
    assert model.name == "gpt-4"


def test_create_completion(db_session):
    """Test creating a completion."""
    dataset = Dataset(name="test_dataset", source="test")
    db_session.add(dataset)
    db_session.commit()
    
    prompt = Prompt(dataset_id=dataset.id, content="What is 2+2?")
    model = Model(name="gpt-4")
    db_session.add_all([prompt, model])
    db_session.commit()
    
    completion = Completion(
        prompt_id=prompt.id,
        model_id=model.id,
        content="4",
        tokens_used=10,
    )
    db_session.add(completion)
    db_session.commit()
    
    assert completion.id is not None
    assert completion.prompt.content == "What is 2+2?"
    assert completion.model.name == "gpt-4"


def test_create_judge(db_session):
    """Test creating a judge."""
    judge = Judge(name="default", model_name="gpt-4")
    db_session.add(judge)
    db_session.commit()
    
    assert judge.id is not None
    assert judge.name == "default"


def test_create_evaluation(db_session):
    """Test creating an evaluation."""
    dataset = Dataset(name="test_dataset", source="test")
    prompt = Prompt(dataset=dataset, content="What is 2+2?")
    model = Model(name="gpt-4")
    judge = Judge(name="default", model_name="gpt-4")
    db_session.add_all([dataset, prompt, model, judge])
    db_session.commit()
    
    completion = Completion(prompt_id=prompt.id, model_id=model.id, content="4")
    db_session.add(completion)
    db_session.commit()
    
    evaluation = Evaluation(
        completion_id=completion.id,
        judge_id=judge.id,
        score=1.0,
        passed=1,
        result="Correct answer",
    )
    db_session.add(evaluation)
    db_session.commit()
    
    assert evaluation.id is not None
    assert evaluation.score == 1.0
    assert evaluation.passed == 1


def test_cascade_delete(db_session):
    """Test cascade delete of prompts when dataset is deleted."""
    dataset = Dataset(name="test_dataset", source="test")
    prompt = Prompt(dataset=dataset, content="Test prompt")
    db_session.add_all([dataset, prompt])
    db_session.commit()
    
    prompt_id = prompt.id
    db_session.delete(dataset)
    db_session.commit()
    
    # Prompt should be deleted
    assert db_session.query(Prompt).filter(Prompt.id == prompt_id).first() is None


def test_init_db():
    """Test database initialization."""
    engine = init_db("sqlite:///:memory:")
    assert engine is not None
