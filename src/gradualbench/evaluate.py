"""Evaluation functionality for GradualBench using LLM judges."""

import json
from typing import Optional

from openai import OpenAI
from sqlalchemy.orm import Session

from .models import Completion, Evaluation, Judge, Model, Prompt


DEFAULT_JUDGE_PROMPT = """You are an expert evaluator. Your task is to evaluate the quality of an AI assistant's response to a given prompt.

Prompt: {prompt}

Expected Answer (if available): {expected}

AI Response: {response}

Please evaluate the response on the following criteria:
1. Correctness: Is the response factually accurate and addresses the prompt correctly?
2. Completeness: Does the response fully address all aspects of the prompt?
3. Clarity: Is the response clear and well-structured?

Provide your evaluation in the following JSON format:
{{
    "score": <float between 0 and 1>,
    "passed": <true/false>,
    "reasoning": "<brief explanation of your evaluation>"
}}

Only output the JSON, nothing else."""


def get_or_create_judge(
    session: Session,
    judge_name: str,
    model_name: str = "gpt-4",
    prompt_template: Optional[str] = None,
    description: Optional[str] = None,
) -> Judge:
    """Get an existing judge or create a new one."""
    judge = session.query(Judge).filter(Judge.name == judge_name).first()
    if not judge:
        judge = Judge(
            name=judge_name,
            model_name=model_name,
            prompt_template=prompt_template or DEFAULT_JUDGE_PROMPT,
            description=description,
        )
        session.add(judge)
        session.commit()
    return judge


def run_evaluation(
    session: Session,
    model_name: str,
    judge_name: str = "default",
    judge_model: str = "gpt-4",
    dataset_name: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    limit: Optional[int] = None,
    skip_existing: bool = True,
) -> list[Evaluation]:
    """
    Run evaluation on completions using an LLM judge.
    
    Args:
        session: Database session
        model_name: Name of the model whose completions to evaluate
        judge_name: Name for the judge configuration
        judge_model: Model to use as judge (e.g., "gpt-4")
        dataset_name: Optional dataset name to filter completions
        api_key: OpenAI API key
        base_url: Custom API base URL
        limit: Maximum number of completions to evaluate
        skip_existing: Skip completions that already have evaluations from this judge
        
    Returns:
        List of created Evaluation objects
    """
    from .ingest import get_dataset
    
    # Get or create judge
    judge = get_or_create_judge(session, judge_name, model_name=judge_model)
    
    # Get model
    model = session.query(Model).filter(Model.name == model_name).first()
    if not model:
        raise ValueError(f"Model '{model_name}' not found")
    
    # Build query for completions
    query = session.query(Completion).filter(Completion.model_id == model.id)
    
    if dataset_name:
        dataset = get_dataset(session, dataset_name)
        if not dataset:
            raise ValueError(f"Dataset '{dataset_name}' not found")
        prompt_ids = session.query(Prompt.id).filter(Prompt.dataset_id == dataset.id)
        query = query.filter(Completion.prompt_id.in_(prompt_ids))
    
    if skip_existing:
        existing_completion_ids = (
            session.query(Evaluation.completion_id)
            .filter(Evaluation.judge_id == judge.id)
            .subquery()
        )
        query = query.filter(~Completion.id.in_(existing_completion_ids))
    
    if limit:
        query = query.limit(limit)
    
    completions = query.all()
    
    if not completions:
        return []
    
    # Initialize OpenAI client
    client_kwargs = {}
    if api_key:
        client_kwargs["api_key"] = api_key
    if base_url:
        client_kwargs["base_url"] = base_url
    
    client = OpenAI(**client_kwargs)
    
    evaluations = []
    prompt_template = judge.prompt_template or DEFAULT_JUDGE_PROMPT
    
    for completion in completions:
        # Get the prompt for this completion
        prompt = session.query(Prompt).filter(Prompt.id == completion.prompt_id).first()
        
        # Build evaluation prompt
        eval_prompt = prompt_template.format(
            prompt=prompt.content,
            expected=prompt.expected_completion or "Not provided",
            response=completion.content,
        )
        
        # Call judge model
        response = client.chat.completions.create(
            model=judge_model,
            messages=[{"role": "user", "content": eval_prompt}],
            temperature=0.0,
            max_tokens=500,
        )
        
        result_text = response.choices[0].message.content
        
        # Parse JSON response
        try:
            result_data = json.loads(result_text)
            score = float(result_data.get("score", 0))
            passed = 1 if result_data.get("passed", False) else 0
        except (json.JSONDecodeError, ValueError) as e:
            # Log warning for debugging but continue with null values
            import logging
            logging.warning(
                f"Failed to parse judge response as JSON for completion {completion.id}: {e}. "
                f"Raw response: {result_text[:200]}..."
            )
            score = None
            passed = None
        
        # Store evaluation
        evaluation = Evaluation(
            completion_id=completion.id,
            judge_id=judge.id,
            score=score,
            result=result_text,
            passed=passed,
        )
        session.add(evaluation)
        evaluations.append(evaluation)
    
    session.commit()
    return evaluations


def list_judges(session: Session) -> list[Judge]:
    """List all judges."""
    return session.query(Judge).all()


def get_evaluations(
    session: Session,
    model_name: str,
    judge_name: Optional[str] = None,
    dataset_name: Optional[str] = None,
) -> list[Evaluation]:
    """Get evaluations for a model, optionally filtered by judge and dataset."""
    from .ingest import get_dataset
    
    model = session.query(Model).filter(Model.name == model_name).first()
    if not model:
        raise ValueError(f"Model '{model_name}' not found")
    
    # Get completion IDs for this model
    completion_query = session.query(Completion.id).filter(Completion.model_id == model.id)
    
    if dataset_name:
        dataset = get_dataset(session, dataset_name)
        if not dataset:
            raise ValueError(f"Dataset '{dataset_name}' not found")
        prompt_ids = session.query(Prompt.id).filter(Prompt.dataset_id == dataset.id)
        completion_query = completion_query.filter(Completion.prompt_id.in_(prompt_ids))
    
    query = session.query(Evaluation).filter(Evaluation.completion_id.in_(completion_query))
    
    if judge_name:
        judge = session.query(Judge).filter(Judge.name == judge_name).first()
        if not judge:
            raise ValueError(f"Judge '{judge_name}' not found")
        query = query.filter(Evaluation.judge_id == judge.id)
    
    return query.all()
