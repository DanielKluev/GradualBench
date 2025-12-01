# GradualBench

LLM evaluation framework and tools aimed to keep all artifacts persisted in DB and resumable at any point.

## Features

- **Dataset Ingestion**: Import any public eval/training dataset from Hugging Face, storing prompts and expected completions separately
- **Model Inference**: Run inference using OpenAI API (or compatible endpoints), storing generated completions
- **LLM Evaluation**: Evaluate completions using OpenAI API judges
- **Model Comparison**: Track and compare evaluation results across different models/checkpoints
- **Persistent Storage**: All data stored in SQLite database for resumability

## Installation

```bash
pip install -e .
```

For development:
```bash
pip install -e ".[dev]"
```

## Quick Start

### 1. Initialize the database

```bash
gradualbench init
```

### 2. Ingest a dataset

```bash
# Ingest from Hugging Face
gradualbench dataset ingest my_dataset gsm8k --prompt-field question --completion-field answer --limit 100

# List datasets
gradualbench dataset list

# Show prompts
gradualbench dataset show my_dataset --limit 5
```

### 3. Run inference

```bash
# Run inference with OpenAI model
export OPENAI_API_KEY=your_api_key
gradualbench inference run my_dataset gpt-4 --limit 10

# For custom endpoints (e.g., local models, vLLM, etc.)
gradualbench inference run my_dataset my-model --base-url http://localhost:8000/v1

# List models
gradualbench inference list-models

# Show completions
gradualbench inference show gpt-4 --dataset my_dataset
```

### 4. Run evaluation

```bash
# Evaluate completions using GPT-4 as judge
gradualbench evaluate run gpt-4 --judge-name default --judge-model gpt-4o

# List judges
gradualbench evaluate list-judges

# Show evaluations
gradualbench evaluate show gpt-4 --judge default
```

### 5. Compare models

```bash
# Compare two models
gradualbench compare models model-a model-b --dataset my_dataset --judge default

# Show all models on a dataset
gradualbench compare all --dataset my_dataset --judge default

# Detailed per-prompt comparison
gradualbench compare detailed model-a model-b --dataset my_dataset --judge default
```

## CLI Commands

### Database
- `gradualbench init` - Initialize the database

### Dataset Management
- `gradualbench dataset ingest NAME SOURCE` - Ingest a dataset from Hugging Face
- `gradualbench dataset list` - List all datasets
- `gradualbench dataset show NAME` - Show prompts from a dataset
- `gradualbench dataset delete NAME` - Delete a dataset

### Inference
- `gradualbench inference run DATASET MODEL` - Run inference on a dataset
- `gradualbench inference list-models` - List all models
- `gradualbench inference show MODEL` - Show completions from a model

### Evaluation
- `gradualbench evaluate run MODEL` - Evaluate completions using an LLM judge
- `gradualbench evaluate list-judges` - List all judges
- `gradualbench evaluate show MODEL` - Show evaluations for a model

### Comparison
- `gradualbench compare models MODEL_A MODEL_B` - Compare two models
- `gradualbench compare all` - Show statistics for all models
- `gradualbench compare detailed MODEL_A MODEL_B` - Per-prompt comparison

## Configuration

### Database Location

By default, the database is stored as `gradualbench.db` in the current directory. Override with:

```bash
# Environment variable
export GRADUALBENCH_DB=sqlite:///path/to/db.sqlite

# Or CLI option
gradualbench --db sqlite:///path/to/db.sqlite dataset list
```

### Custom API Endpoints

GradualBench supports any OpenAI-compatible API endpoint:

```bash
# vLLM
gradualbench inference run my_dataset my-model --base-url http://localhost:8000/v1

# Azure OpenAI
gradualbench inference run my_dataset gpt-4 --base-url https://your-resource.openai.azure.com/
```

## Python API

```python
from gradualbench.models import init_db, get_session
from gradualbench.ingest import ingest_dataset, list_datasets
from gradualbench.inference import run_inference
from gradualbench.evaluate import run_evaluation
from gradualbench.compare import compare_models, get_model_stats

# Initialize database
engine = init_db("sqlite:///gradualbench.db")
session = get_session(engine)

# Ingest a dataset
dataset = ingest_dataset(
    session,
    dataset_name="my_dataset",
    source="gsm8k",
    prompt_field="question",
    completion_field="answer",
    limit=100,
)

# Run inference
completions = run_inference(
    session,
    dataset_name="my_dataset",
    model_name="gpt-4",
)

# Run evaluation
evaluations = run_evaluation(
    session,
    model_name="gpt-4",
    judge_model="gpt-4o",
)

# Get statistics
stats = get_model_stats(session, "gpt-4", "my_dataset", "default")
print(f"Pass rate: {stats.pass_rate}%")

# Compare models
comparison = compare_models(session, "gpt-3.5-turbo", "gpt-4", "my_dataset", "default")
print(f"Score improvement: {comparison.score_diff}")

session.close()
```

## License

MIT License - see [LICENSE](LICENSE) for details.

