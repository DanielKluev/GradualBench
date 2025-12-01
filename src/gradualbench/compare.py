"""Comparison and tracking functionality for evaluations across models."""

from dataclasses import dataclass
from typing import Optional

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .models import Completion, Dataset, Evaluation, Judge, Model, Prompt


@dataclass
class ModelStats:
    """Statistics for a model's evaluation results."""
    
    model_name: str
    dataset_name: str
    judge_name: str
    total_completions: int
    total_evaluations: int
    avg_score: Optional[float]
    pass_rate: Optional[float]
    total_passed: int
    total_failed: int


def get_model_stats(
    session: Session,
    model_name: str,
    dataset_name: str,
    judge_name: str,
) -> ModelStats:
    """Get evaluation statistics for a model on a dataset with a specific judge."""
    from .ingest import get_dataset
    
    # Get model
    model = session.query(Model).filter(Model.name == model_name).first()
    if not model:
        raise ValueError(f"Model '{model_name}' not found")
    
    # Get dataset
    dataset = get_dataset(session, dataset_name)
    if not dataset:
        raise ValueError(f"Dataset '{dataset_name}' not found")
    
    # Get judge
    judge = session.query(Judge).filter(Judge.name == judge_name).first()
    if not judge:
        raise ValueError(f"Judge '{judge_name}' not found")
    
    # Get prompt IDs for this dataset
    prompt_ids_subq = select(Prompt.id).filter(Prompt.dataset_id == dataset.id).scalar_subquery()
    
    # Get completion count
    total_completions = (
        session.query(func.count(Completion.id))
        .filter(Completion.model_id == model.id)
        .filter(Completion.prompt_id.in_(select(Prompt.id).filter(Prompt.dataset_id == dataset.id)))
        .scalar()
    )
    
    # Get completion IDs for this model and dataset
    completion_ids_subq = (
        select(Completion.id)
        .filter(Completion.model_id == model.id)
        .filter(Completion.prompt_id.in_(select(Prompt.id).filter(Prompt.dataset_id == dataset.id)))
    )
    
    # Get evaluation statistics
    eval_stats = (
        session.query(
            func.count(Evaluation.id).label("total"),
            func.avg(Evaluation.score).label("avg_score"),
            func.sum(Evaluation.passed).label("total_passed"),
        )
        .filter(Evaluation.completion_id.in_(completion_ids_subq))
        .filter(Evaluation.judge_id == judge.id)
        .first()
    )
    
    total_evaluations = eval_stats[0] or 0
    avg_score = float(eval_stats[1]) if eval_stats[1] is not None else None
    total_passed = int(eval_stats[2] or 0)
    total_failed = total_evaluations - total_passed
    pass_rate = (total_passed / total_evaluations * 100) if total_evaluations > 0 else None
    
    return ModelStats(
        model_name=model_name,
        dataset_name=dataset_name,
        judge_name=judge_name,
        total_completions=total_completions,
        total_evaluations=total_evaluations,
        avg_score=avg_score,
        pass_rate=pass_rate,
        total_passed=total_passed,
        total_failed=total_failed,
    )


@dataclass
class ModelComparison:
    """Comparison results between two models."""
    
    model_a_name: str
    model_b_name: str
    dataset_name: str
    judge_name: str
    model_a_stats: ModelStats
    model_b_stats: ModelStats
    score_diff: Optional[float]  # model_b - model_a
    pass_rate_diff: Optional[float]  # model_b - model_a
    improvement_summary: str


def compare_models(
    session: Session,
    model_a_name: str,
    model_b_name: str,
    dataset_name: str,
    judge_name: str,
) -> ModelComparison:
    """Compare evaluation results between two models."""
    stats_a = get_model_stats(session, model_a_name, dataset_name, judge_name)
    stats_b = get_model_stats(session, model_b_name, dataset_name, judge_name)
    
    # Calculate differences
    score_diff = None
    if stats_a.avg_score is not None and stats_b.avg_score is not None:
        score_diff = stats_b.avg_score - stats_a.avg_score
    
    pass_rate_diff = None
    if stats_a.pass_rate is not None and stats_b.pass_rate is not None:
        pass_rate_diff = stats_b.pass_rate - stats_a.pass_rate
    
    # Generate summary
    summaries = []
    if score_diff is not None:
        if score_diff > 0:
            summaries.append(f"Score improved by {score_diff:.3f}")
        elif score_diff < 0:
            summaries.append(f"Score decreased by {abs(score_diff):.3f}")
        else:
            summaries.append("Score unchanged")
    
    if pass_rate_diff is not None:
        if pass_rate_diff > 0:
            summaries.append(f"Pass rate improved by {pass_rate_diff:.1f}%")
        elif pass_rate_diff < 0:
            summaries.append(f"Pass rate decreased by {abs(pass_rate_diff):.1f}%")
        else:
            summaries.append("Pass rate unchanged")
    
    improvement_summary = "; ".join(summaries) if summaries else "No comparison data available"
    
    return ModelComparison(
        model_a_name=model_a_name,
        model_b_name=model_b_name,
        dataset_name=dataset_name,
        judge_name=judge_name,
        model_a_stats=stats_a,
        model_b_stats=stats_b,
        score_diff=score_diff,
        pass_rate_diff=pass_rate_diff,
        improvement_summary=improvement_summary,
    )


def get_all_model_stats_for_dataset(
    session: Session,
    dataset_name: str,
    judge_name: str,
) -> list[ModelStats]:
    """Get statistics for all models evaluated on a dataset."""
    # Get all models that have completions for this dataset
    from .ingest import get_dataset
    
    dataset = get_dataset(session, dataset_name)
    if not dataset:
        raise ValueError(f"Dataset '{dataset_name}' not found")
    
    prompt_ids_subq = select(Prompt.id).filter(Prompt.dataset_id == dataset.id)
    
    model_ids = (
        session.query(Completion.model_id)
        .filter(Completion.prompt_id.in_(prompt_ids_subq))
        .distinct()
        .all()
    )
    
    stats_list = []
    for (model_id,) in model_ids:
        model = session.query(Model).filter(Model.id == model_id).first()
        if model:
            try:
                stats = get_model_stats(session, model.name, dataset_name, judge_name)
                stats_list.append(stats)
            except ValueError:
                continue
    
    return stats_list


def get_detailed_comparison(
    session: Session,
    model_a_name: str,
    model_b_name: str,
    dataset_name: str,
    judge_name: str,
) -> list[dict]:
    """Get detailed per-prompt comparison between two models."""
    from .ingest import get_dataset
    
    # Get entities
    model_a = session.query(Model).filter(Model.name == model_a_name).first()
    model_b = session.query(Model).filter(Model.name == model_b_name).first()
    dataset = get_dataset(session, dataset_name)
    judge = session.query(Judge).filter(Judge.name == judge_name).first()
    
    if not all([model_a, model_b, dataset, judge]):
        raise ValueError("One or more entities not found")
    
    # Get prompts
    prompts = session.query(Prompt).filter(Prompt.dataset_id == dataset.id).all()
    
    comparisons = []
    for prompt in prompts:
        # Get completion and evaluation for model A
        comp_a = (
            session.query(Completion)
            .filter(Completion.prompt_id == prompt.id)
            .filter(Completion.model_id == model_a.id)
            .first()
        )
        
        eval_a = None
        if comp_a:
            eval_a = (
                session.query(Evaluation)
                .filter(Evaluation.completion_id == comp_a.id)
                .filter(Evaluation.judge_id == judge.id)
                .first()
            )
        
        # Get completion and evaluation for model B
        comp_b = (
            session.query(Completion)
            .filter(Completion.prompt_id == prompt.id)
            .filter(Completion.model_id == model_b.id)
            .first()
        )
        
        eval_b = None
        if comp_b:
            eval_b = (
                session.query(Evaluation)
                .filter(Evaluation.completion_id == comp_b.id)
                .filter(Evaluation.judge_id == judge.id)
                .first()
            )
        
        comparisons.append({
            "prompt_id": prompt.id,
            "prompt_content": prompt.content[:100] + "..." if len(prompt.content) > 100 else prompt.content,
            "expected": prompt.expected_completion[:100] + "..." if prompt.expected_completion and len(prompt.expected_completion) > 100 else prompt.expected_completion,
            "model_a_completion": comp_a.content[:100] + "..." if comp_a and len(comp_a.content) > 100 else (comp_a.content if comp_a else None),
            "model_a_score": eval_a.score if eval_a else None,
            "model_a_passed": eval_a.passed if eval_a else None,
            "model_b_completion": comp_b.content[:100] + "..." if comp_b and len(comp_b.content) > 100 else (comp_b.content if comp_b else None),
            "model_b_score": eval_b.score if eval_b else None,
            "model_b_passed": eval_b.passed if eval_b else None,
            "score_change": (eval_b.score - eval_a.score) if (eval_a and eval_b and eval_a.score and eval_b.score) else None,
        })
    
    return comparisons
