"""Dataset ingestion functionality for GradualBench."""

import json
from typing import Any, Optional

from datasets import load_dataset
from sqlalchemy.orm import Session

from .models import Dataset, Prompt


def ingest_dataset(
    session: Session,
    dataset_name: str,
    source: str,
    prompt_field: str = "question",
    completion_field: Optional[str] = "answer",
    split: str = "train",
    subset: Optional[str] = None,
    limit: Optional[int] = None,
    description: Optional[str] = None,
) -> Dataset:
    """
    Ingest a dataset from Hugging Face datasets library.
    
    Args:
        session: Database session
        dataset_name: Name to store the dataset under in the database
        source: Hugging Face dataset identifier (e.g., "squad", "gsm8k")
        prompt_field: Field name containing the prompt/question
        completion_field: Field name containing the expected completion/answer
        split: Dataset split to use (train, test, validation)
        subset: Dataset subset/configuration if applicable
        limit: Maximum number of examples to ingest
        description: Optional description for the dataset
        
    Returns:
        The created Dataset object
    """
    # Check if dataset already exists
    existing = session.query(Dataset).filter(Dataset.name == dataset_name).first()
    if existing:
        raise ValueError(f"Dataset '{dataset_name}' already exists. Use a different name or delete the existing one.")
    
    # Load from Hugging Face
    if subset:
        hf_dataset = load_dataset(source, subset, split=split)
    else:
        hf_dataset = load_dataset(source, split=split)
    
    # Create dataset record
    dataset = Dataset(
        name=dataset_name,
        source=f"huggingface:{source}" + (f"/{subset}" if subset else ""),
        description=description or f"Ingested from {source}",
    )
    session.add(dataset)
    session.flush()  # Get the dataset ID
    
    # Ingest prompts
    count = 0
    for idx, item in enumerate(hf_dataset):
        if limit and count >= limit:
            break
            
        # Extract prompt content
        if prompt_field not in item:
            raise ValueError(f"Field '{prompt_field}' not found in dataset. Available fields: {list(item.keys())}")
        
        prompt_content = item[prompt_field]
        
        # Extract expected completion if field specified
        expected_completion = None
        if completion_field and completion_field in item:
            expected_completion = item[completion_field]
            if isinstance(expected_completion, list):
                expected_completion = expected_completion[0] if expected_completion else None
            elif not isinstance(expected_completion, str):
                expected_completion = str(expected_completion)
        
        # Store any additional metadata
        metadata = {k: v for k, v in item.items() if k not in [prompt_field, completion_field]}
        
        prompt = Prompt(
            dataset_id=dataset.id,
            external_id=str(idx),
            content=prompt_content if isinstance(prompt_content, str) else str(prompt_content),
            expected_completion=expected_completion,
            metadata_json=json.dumps(metadata, default=str) if metadata else None,
        )
        session.add(prompt)
        count += 1
    
    session.commit()
    return dataset


def list_datasets(session: Session) -> list[Dataset]:
    """List all ingested datasets."""
    return session.query(Dataset).all()


def get_dataset(session: Session, name: str) -> Optional[Dataset]:
    """Get a dataset by name."""
    return session.query(Dataset).filter(Dataset.name == name).first()


def delete_dataset(session: Session, name: str) -> bool:
    """Delete a dataset and all its prompts."""
    dataset = get_dataset(session, name)
    if dataset:
        session.delete(dataset)
        session.commit()
        return True
    return False


def get_dataset_prompts(session: Session, dataset_name: str, limit: Optional[int] = None) -> list[Prompt]:
    """Get prompts from a dataset."""
    dataset = get_dataset(session, dataset_name)
    if not dataset:
        raise ValueError(f"Dataset '{dataset_name}' not found")
    
    query = session.query(Prompt).filter(Prompt.dataset_id == dataset.id)
    if limit:
        query = query.limit(limit)
    return query.all()
