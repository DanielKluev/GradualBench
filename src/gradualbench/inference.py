"""Inference functionality for GradualBench."""

import json
import time
from typing import Optional

from openai import OpenAI
from sqlalchemy.orm import Session

from .models import Completion, Model, Prompt


def get_or_create_model(
    session: Session,
    model_name: str,
    endpoint: Optional[str] = None,
    description: Optional[str] = None,
) -> Model:
    """Get an existing model or create a new one."""
    model = session.query(Model).filter(Model.name == model_name).first()
    if not model:
        model = Model(
            name=model_name,
            endpoint=endpoint,
            description=description,
        )
        session.add(model)
        session.commit()
    return model


def run_inference(
    session: Session,
    dataset_name: str,
    model_name: str,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    system_prompt: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    limit: Optional[int] = None,
    skip_existing: bool = True,
) -> list[Completion]:
    """
    Run inference on prompts from a dataset using OpenAI API.
    
    Args:
        session: Database session
        dataset_name: Name of the dataset to run inference on
        model_name: Model identifier (e.g., "gpt-4", "gpt-3.5-turbo")
        api_key: OpenAI API key (uses OPENAI_API_KEY env var if not provided)
        base_url: Custom API base URL for compatible endpoints
        system_prompt: Optional system prompt to prepend
        temperature: Sampling temperature
        max_tokens: Maximum tokens in completion
        limit: Maximum number of prompts to process
        skip_existing: Skip prompts that already have completions for this model
        
    Returns:
        List of created Completion objects
    """
    from .ingest import get_dataset
    
    # Get or create model record
    model = get_or_create_model(session, model_name, endpoint=base_url)
    
    # Get dataset
    dataset = get_dataset(session, dataset_name)
    if not dataset:
        raise ValueError(f"Dataset '{dataset_name}' not found")
    
    # Get prompts
    prompts_query = session.query(Prompt).filter(Prompt.dataset_id == dataset.id)
    
    if skip_existing:
        # Exclude prompts that already have completions for this model
        existing_prompt_ids = (
            session.query(Completion.prompt_id)
            .filter(Completion.model_id == model.id)
            .subquery()
        )
        prompts_query = prompts_query.filter(~Prompt.id.in_(existing_prompt_ids))
    
    if limit:
        prompts_query = prompts_query.limit(limit)
    
    prompts = prompts_query.all()
    
    if not prompts:
        return []
    
    # Initialize OpenAI client
    client_kwargs = {}
    if api_key:
        client_kwargs["api_key"] = api_key
    if base_url:
        client_kwargs["base_url"] = base_url
    
    client = OpenAI(**client_kwargs)
    
    completions = []
    for prompt in prompts:
        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt.content})
        
        # Call API
        start_time = time.time()
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        latency_ms = (time.time() - start_time) * 1000
        
        # Extract response
        content = response.choices[0].message.content
        tokens_used = response.usage.total_tokens if response.usage else None
        
        # Store completion
        completion = Completion(
            prompt_id=prompt.id,
            model_id=model.id,
            content=content,
            tokens_used=tokens_used,
            latency_ms=latency_ms,
            metadata_json=json.dumps({
                "temperature": temperature,
                "max_tokens": max_tokens,
                "finish_reason": response.choices[0].finish_reason,
            }),
        )
        session.add(completion)
        completions.append(completion)
    
    session.commit()
    return completions


def list_models(session: Session) -> list[Model]:
    """List all models."""
    return session.query(Model).all()


def get_completions(
    session: Session,
    model_name: str,
    dataset_name: Optional[str] = None,
) -> list[Completion]:
    """Get completions for a model, optionally filtered by dataset."""
    from .ingest import get_dataset
    
    model = session.query(Model).filter(Model.name == model_name).first()
    if not model:
        raise ValueError(f"Model '{model_name}' not found")
    
    query = session.query(Completion).filter(Completion.model_id == model.id)
    
    if dataset_name:
        dataset = get_dataset(session, dataset_name)
        if not dataset:
            raise ValueError(f"Dataset '{dataset_name}' not found")
        prompt_ids = session.query(Prompt.id).filter(Prompt.dataset_id == dataset.id)
        query = query.filter(Completion.prompt_id.in_(prompt_ids))
    
    return query.all()
