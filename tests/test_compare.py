"""Tests for compare functionality."""

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
)
from gradualbench.compare import (
    compare_models,
    get_all_model_stats_for_dataset,
    get_model_stats,
)


@pytest.fixture
def db_session():
    """Create an in-memory database session for testing."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = get_session(engine)
    yield session
    session.close()


@pytest.fixture
def populated_db(db_session):
    """Create a populated database for comparison tests."""
    # Create dataset
    dataset = Dataset(name="test_dataset", source="test")
    db_session.add(dataset)
    db_session.commit()
    
    # Create prompts
    prompts = []
    for i in range(5):
        prompt = Prompt(
            dataset_id=dataset.id,
            content=f"Question {i}",
            expected_completion=f"Answer {i}",
        )
        prompts.append(prompt)
    db_session.add_all(prompts)
    db_session.commit()
    
    # Create models
    model_a = Model(name="model-a")
    model_b = Model(name="model-b")
    db_session.add_all([model_a, model_b])
    db_session.commit()
    
    # Create judge
    judge = Judge(name="default", model_name="gpt-4")
    db_session.add(judge)
    db_session.commit()
    
    # Create completions for model A
    completions_a = []
    for prompt in prompts:
        comp = Completion(
            prompt_id=prompt.id,
            model_id=model_a.id,
            content=f"Response A for {prompt.content}",
        )
        completions_a.append(comp)
    db_session.add_all(completions_a)
    db_session.commit()
    
    # Create completions for model B
    completions_b = []
    for prompt in prompts:
        comp = Completion(
            prompt_id=prompt.id,
            model_id=model_b.id,
            content=f"Response B for {prompt.content}",
        )
        completions_b.append(comp)
    db_session.add_all(completions_b)
    db_session.commit()
    
    # Create evaluations for model A (scores: 0.6, 0.7, 0.8, 0.5, 0.4)
    scores_a = [0.6, 0.7, 0.8, 0.5, 0.4]
    for comp, score in zip(completions_a, scores_a):
        eval_ = Evaluation(
            completion_id=comp.id,
            judge_id=judge.id,
            score=score,
            passed=1 if score >= 0.5 else 0,
        )
        db_session.add(eval_)
    
    # Create evaluations for model B (scores: 0.8, 0.9, 0.7, 0.6, 0.5)
    scores_b = [0.8, 0.9, 0.7, 0.6, 0.5]
    for comp, score in zip(completions_b, scores_b):
        eval_ = Evaluation(
            completion_id=comp.id,
            judge_id=judge.id,
            score=score,
            passed=1 if score >= 0.5 else 0,
        )
        db_session.add(eval_)
    
    db_session.commit()
    
    return db_session


def test_get_model_stats(populated_db):
    """Test getting model statistics."""
    stats = get_model_stats(populated_db, "model-a", "test_dataset", "default")
    
    assert stats.model_name == "model-a"
    assert stats.dataset_name == "test_dataset"
    assert stats.judge_name == "default"
    assert stats.total_completions == 5
    assert stats.total_evaluations == 5
    assert abs(stats.avg_score - 0.6) < 0.01  # Average of [0.6, 0.7, 0.8, 0.5, 0.4]
    assert stats.total_passed == 4  # 4 scores >= 0.5


def test_compare_models(populated_db):
    """Test comparing two models."""
    comparison = compare_models(
        populated_db, "model-a", "model-b", "test_dataset", "default"
    )
    
    assert comparison.model_a_name == "model-a"
    assert comparison.model_b_name == "model-b"
    assert comparison.score_diff > 0  # model-b should have higher scores
    # Model A: 4/5 pass (80%), Model B: 5/5 pass (100%), so diff is 20%
    assert comparison.pass_rate_diff == 20.0


def test_get_all_model_stats(populated_db):
    """Test getting stats for all models."""
    stats_list = get_all_model_stats_for_dataset(
        populated_db, "test_dataset", "default"
    )
    
    assert len(stats_list) == 2
    model_names = {s.model_name for s in stats_list}
    assert "model-a" in model_names
    assert "model-b" in model_names
