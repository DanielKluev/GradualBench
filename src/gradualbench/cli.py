"""Command-line interface for GradualBench."""

import os
from typing import Optional

import click
from rich.console import Console
from rich.table import Table

from .models import get_session, init_db

console = Console()

DEFAULT_DB_PATH = "gradualbench.db"


def get_database_url():
    """Get database URL from environment or default."""
    return os.environ.get("GRADUALBENCH_DB", f"sqlite:///{DEFAULT_DB_PATH}")


@click.group()
@click.option("--db", envvar="GRADUALBENCH_DB", default=None, help="Database URL")
@click.pass_context
def cli(ctx, db):
    """GradualBench - LLM evaluation framework with persistent storage."""
    ctx.ensure_object(dict)
    ctx.obj["db_url"] = db or get_database_url()


@cli.command()
@click.pass_context
def init(ctx):
    """Initialize the database."""
    db_url = ctx.obj["db_url"]
    init_db(db_url)
    console.print(f"[green]Database initialized at {db_url}[/green]")


# ============== Dataset Commands ==============

@cli.group()
def dataset():
    """Manage datasets."""
    pass


@dataset.command("ingest")
@click.argument("name")
@click.argument("source")
@click.option("--prompt-field", default="question", help="Field containing prompts")
@click.option("--completion-field", default="answer", help="Field containing expected completions")
@click.option("--split", default="train", help="Dataset split to use")
@click.option("--subset", default=None, help="Dataset subset/configuration")
@click.option("--limit", default=None, type=int, help="Maximum number of examples")
@click.option("--description", default=None, help="Dataset description")
@click.pass_context
def ingest_dataset(ctx, name, source, prompt_field, completion_field, split, subset, limit, description):
    """Ingest a dataset from Hugging Face.
    
    NAME: Name to store the dataset under
    SOURCE: Hugging Face dataset identifier (e.g., 'gsm8k', 'squad')
    """
    from .ingest import ingest_dataset as _ingest_dataset
    
    db_url = ctx.obj["db_url"]
    engine = init_db(db_url)
    session = get_session(engine)
    
    try:
        with console.status(f"[bold blue]Ingesting dataset '{source}'...[/bold blue]"):
            dataset = _ingest_dataset(
                session=session,
                dataset_name=name,
                source=source,
                prompt_field=prompt_field,
                completion_field=completion_field,
                split=split,
                subset=subset,
                limit=limit,
                description=description,
            )
        
        prompt_count = len(dataset.prompts)
        console.print(f"[green]Successfully ingested {prompt_count} prompts from '{source}' as '{name}'[/green]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise click.Abort()
    finally:
        session.close()


@dataset.command("list")
@click.pass_context
def list_datasets(ctx):
    """List all ingested datasets."""
    from .ingest import list_datasets as _list_datasets
    
    db_url = ctx.obj["db_url"]
    engine = init_db(db_url)
    session = get_session(engine)
    
    try:
        datasets = _list_datasets(session)
        
        if not datasets:
            console.print("[yellow]No datasets found.[/yellow]")
            return
        
        table = Table(title="Datasets")
        table.add_column("Name", style="cyan")
        table.add_column("Source", style="magenta")
        table.add_column("Prompts", justify="right")
        table.add_column("Created", style="green")
        
        for ds in datasets:
            table.add_row(
                ds.name,
                ds.source,
                str(len(ds.prompts)),
                ds.created_at.strftime("%Y-%m-%d %H:%M"),
            )
        
        console.print(table)
    finally:
        session.close()


@dataset.command("delete")
@click.argument("name")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation")
@click.pass_context
def delete_dataset(ctx, name, yes):
    """Delete a dataset and all its prompts."""
    from .ingest import delete_dataset as _delete_dataset
    
    if not yes:
        if not click.confirm(f"Are you sure you want to delete dataset '{name}'?"):
            console.print("[yellow]Cancelled.[/yellow]")
            return
    
    db_url = ctx.obj["db_url"]
    engine = init_db(db_url)
    session = get_session(engine)
    
    try:
        if _delete_dataset(session, name):
            console.print(f"[green]Dataset '{name}' deleted.[/green]")
        else:
            console.print(f"[red]Dataset '{name}' not found.[/red]")
    finally:
        session.close()


@dataset.command("show")
@click.argument("name")
@click.option("--limit", default=10, type=int, help="Number of prompts to show")
@click.pass_context
def show_dataset(ctx, name, limit):
    """Show prompts from a dataset."""
    from .ingest import get_dataset_prompts
    
    db_url = ctx.obj["db_url"]
    engine = init_db(db_url)
    session = get_session(engine)
    
    try:
        prompts = get_dataset_prompts(session, name, limit=limit)
        
        if not prompts:
            console.print(f"[yellow]No prompts found in dataset '{name}'.[/yellow]")
            return
        
        table = Table(title=f"Prompts from '{name}' (showing {len(prompts)})")
        table.add_column("ID", style="cyan")
        table.add_column("Prompt", style="white", max_width=60)
        table.add_column("Expected", style="green", max_width=40)
        
        for prompt in prompts:
            table.add_row(
                str(prompt.id),
                prompt.content[:100] + "..." if len(prompt.content) > 100 else prompt.content,
                (prompt.expected_completion[:80] + "..." if prompt.expected_completion and len(prompt.expected_completion) > 80 else prompt.expected_completion) or "-",
            )
        
        console.print(table)
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
    finally:
        session.close()


# ============== Inference Commands ==============

@cli.group()
def inference():
    """Run model inference."""
    pass


@inference.command("run")
@click.argument("dataset_name")
@click.argument("model_name")
@click.option("--api-key", envvar="OPENAI_API_KEY", help="OpenAI API key")
@click.option("--base-url", default=None, help="Custom API base URL")
@click.option("--system-prompt", default=None, help="System prompt to use")
@click.option("--temperature", default=0.0, type=float, help="Sampling temperature")
@click.option("--max-tokens", default=1024, type=int, help="Max tokens in response")
@click.option("--limit", default=None, type=int, help="Max prompts to process")
@click.option("--no-skip", is_flag=True, help="Don't skip existing completions")
@click.pass_context
def run_inference(ctx, dataset_name, model_name, api_key, base_url, system_prompt, temperature, max_tokens, limit, no_skip):
    """Run inference on a dataset using a model.
    
    DATASET_NAME: Name of the dataset to run inference on
    MODEL_NAME: Model identifier (e.g., 'gpt-4', 'gpt-3.5-turbo')
    """
    from .inference import run_inference as _run_inference
    
    db_url = ctx.obj["db_url"]
    engine = init_db(db_url)
    session = get_session(engine)
    
    try:
        with console.status(f"[bold blue]Running inference with '{model_name}'...[/bold blue]"):
            completions = _run_inference(
                session=session,
                dataset_name=dataset_name,
                model_name=model_name,
                api_key=api_key,
                base_url=base_url,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                limit=limit,
                skip_existing=not no_skip,
            )
        
        console.print(f"[green]Generated {len(completions)} completions[/green]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise click.Abort()
    finally:
        session.close()


@inference.command("list-models")
@click.pass_context
def list_models(ctx):
    """List all models with completions."""
    from .inference import list_models as _list_models
    
    db_url = ctx.obj["db_url"]
    engine = init_db(db_url)
    session = get_session(engine)
    
    try:
        models = _list_models(session)
        
        if not models:
            console.print("[yellow]No models found.[/yellow]")
            return
        
        table = Table(title="Models")
        table.add_column("Name", style="cyan")
        table.add_column("Endpoint", style="magenta")
        table.add_column("Completions", justify="right")
        table.add_column("Created", style="green")
        
        for model in models:
            table.add_row(
                model.name,
                model.endpoint or "default",
                str(len(model.completions)),
                model.created_at.strftime("%Y-%m-%d %H:%M"),
            )
        
        console.print(table)
    finally:
        session.close()


@inference.command("show")
@click.argument("model_name")
@click.option("--dataset", default=None, help="Filter by dataset name")
@click.option("--limit", default=10, type=int, help="Number of completions to show")
@click.pass_context
def show_completions(ctx, model_name, dataset, limit):
    """Show completions from a model."""
    from .inference import get_completions
    
    db_url = ctx.obj["db_url"]
    engine = init_db(db_url)
    session = get_session(engine)
    
    try:
        completions = get_completions(session, model_name, dataset_name=dataset)[:limit]
        
        if not completions:
            console.print(f"[yellow]No completions found for model '{model_name}'.[/yellow]")
            return
        
        table = Table(title=f"Completions from '{model_name}' (showing {len(completions)})")
        table.add_column("ID", style="cyan")
        table.add_column("Prompt", style="white", max_width=40)
        table.add_column("Completion", style="green", max_width=50)
        table.add_column("Tokens", justify="right")
        
        for comp in completions:
            prompt = comp.prompt
            table.add_row(
                str(comp.id),
                prompt.content[:60] + "..." if len(prompt.content) > 60 else prompt.content,
                comp.content[:80] + "..." if len(comp.content) > 80 else comp.content,
                str(comp.tokens_used or "-"),
            )
        
        console.print(table)
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
    finally:
        session.close()


# ============== Evaluation Commands ==============

@cli.group()
def evaluate():
    """Run evaluations with LLM judges."""
    pass


@evaluate.command("run")
@click.argument("model_name")
@click.option("--judge-name", default="default", help="Judge configuration name")
@click.option("--judge-model", default="gpt-4", help="Model to use as judge")
@click.option("--dataset", default=None, help="Filter by dataset name")
@click.option("--api-key", envvar="OPENAI_API_KEY", help="OpenAI API key")
@click.option("--base-url", default=None, help="Custom API base URL")
@click.option("--limit", default=None, type=int, help="Max completions to evaluate")
@click.option("--no-skip", is_flag=True, help="Don't skip existing evaluations")
@click.pass_context
def run_evaluation(ctx, model_name, judge_name, judge_model, dataset, api_key, base_url, limit, no_skip):
    """Run evaluation on completions using an LLM judge.
    
    MODEL_NAME: Name of the model whose completions to evaluate
    """
    from .evaluate import run_evaluation as _run_evaluation
    
    db_url = ctx.obj["db_url"]
    engine = init_db(db_url)
    session = get_session(engine)
    
    try:
        with console.status(f"[bold blue]Running evaluation with judge '{judge_model}'...[/bold blue]"):
            evaluations = _run_evaluation(
                session=session,
                model_name=model_name,
                judge_name=judge_name,
                judge_model=judge_model,
                dataset_name=dataset,
                api_key=api_key,
                base_url=base_url,
                limit=limit,
                skip_existing=not no_skip,
            )
        
        # Calculate summary
        total = len(evaluations)
        passed = sum(1 for e in evaluations if e.passed == 1)
        avg_score = sum(e.score for e in evaluations if e.score is not None) / total if total > 0 else 0
        
        console.print(f"[green]Evaluated {total} completions[/green]")
        console.print(f"  Pass rate: {passed}/{total} ({passed/total*100:.1f}%)" if total > 0 else "")
        console.print(f"  Avg score: {avg_score:.3f}" if total > 0 else "")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise click.Abort()
    finally:
        session.close()


@evaluate.command("list-judges")
@click.pass_context
def list_judges(ctx):
    """List all judge configurations."""
    from .evaluate import list_judges as _list_judges
    
    db_url = ctx.obj["db_url"]
    engine = init_db(db_url)
    session = get_session(engine)
    
    try:
        judges = _list_judges(session)
        
        if not judges:
            console.print("[yellow]No judges found.[/yellow]")
            return
        
        table = Table(title="Judges")
        table.add_column("Name", style="cyan")
        table.add_column("Model", style="magenta")
        table.add_column("Evaluations", justify="right")
        table.add_column("Created", style="green")
        
        for judge in judges:
            table.add_row(
                judge.name,
                judge.model_name,
                str(len(judge.evaluations)),
                judge.created_at.strftime("%Y-%m-%d %H:%M"),
            )
        
        console.print(table)
    finally:
        session.close()


@evaluate.command("show")
@click.argument("model_name")
@click.option("--judge", default=None, help="Filter by judge name")
@click.option("--dataset", default=None, help="Filter by dataset name")
@click.option("--limit", default=10, type=int, help="Number of evaluations to show")
@click.pass_context
def show_evaluations(ctx, model_name, judge, dataset, limit):
    """Show evaluations for a model."""
    from .evaluate import get_evaluations
    
    db_url = ctx.obj["db_url"]
    engine = init_db(db_url)
    session = get_session(engine)
    
    try:
        evaluations = get_evaluations(session, model_name, judge_name=judge, dataset_name=dataset)[:limit]
        
        if not evaluations:
            console.print(f"[yellow]No evaluations found for model '{model_name}'.[/yellow]")
            return
        
        table = Table(title=f"Evaluations for '{model_name}' (showing {len(evaluations)})")
        table.add_column("ID", style="cyan")
        table.add_column("Judge", style="magenta")
        table.add_column("Score", justify="right")
        table.add_column("Passed", style="green")
        table.add_column("Result", max_width=50)
        
        for eval_ in evaluations:
            table.add_row(
                str(eval_.id),
                eval_.judge.name,
                f"{eval_.score:.3f}" if eval_.score is not None else "-",
                "✓" if eval_.passed == 1 else "✗" if eval_.passed == 0 else "-",
                eval_.result[:80] + "..." if eval_.result and len(eval_.result) > 80 else (eval_.result or "-"),
            )
        
        console.print(table)
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
    finally:
        session.close()


# ============== Compare Commands ==============

@cli.group()
def compare():
    """Compare and track evaluation changes."""
    pass


@compare.command("models")
@click.argument("model_a")
@click.argument("model_b")
@click.option("--dataset", required=True, help="Dataset to compare on")
@click.option("--judge", default="default", help="Judge to use for comparison")
@click.pass_context
def compare_models(ctx, model_a, model_b, dataset, judge):
    """Compare evaluation results between two models.
    
    MODEL_A: First model (baseline)
    MODEL_B: Second model (to compare against baseline)
    """
    from .compare import compare_models as _compare_models
    
    db_url = ctx.obj["db_url"]
    engine = init_db(db_url)
    session = get_session(engine)
    
    try:
        comparison = _compare_models(session, model_a, model_b, dataset, judge)
        
        # Display comparison results
        console.print(f"\n[bold]Model Comparison: {model_a} → {model_b}[/bold]")
        console.print(f"Dataset: {dataset}")
        console.print(f"Judge: {judge}\n")
        
        table = Table(title="Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column(model_a, justify="right")
        table.add_column(model_b, justify="right")
        table.add_column("Change", justify="right")
        
        stats_a = comparison.model_a_stats
        stats_b = comparison.model_b_stats
        
        table.add_row(
            "Completions",
            str(stats_a.total_completions),
            str(stats_b.total_completions),
            "-",
        )
        table.add_row(
            "Evaluations",
            str(stats_a.total_evaluations),
            str(stats_b.total_evaluations),
            "-",
        )
        table.add_row(
            "Avg Score",
            f"{stats_a.avg_score:.3f}" if stats_a.avg_score else "-",
            f"{stats_b.avg_score:.3f}" if stats_b.avg_score else "-",
            f"{comparison.score_diff:+.3f}" if comparison.score_diff else "-",
        )
        table.add_row(
            "Pass Rate",
            f"{stats_a.pass_rate:.1f}%" if stats_a.pass_rate else "-",
            f"{stats_b.pass_rate:.1f}%" if stats_b.pass_rate else "-",
            f"{comparison.pass_rate_diff:+.1f}%" if comparison.pass_rate_diff else "-",
        )
        table.add_row(
            "Passed",
            str(stats_a.total_passed),
            str(stats_b.total_passed),
            f"{stats_b.total_passed - stats_a.total_passed:+d}",
        )
        
        console.print(table)
        console.print(f"\n[bold]Summary:[/bold] {comparison.improvement_summary}")
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
    finally:
        session.close()


@compare.command("all")
@click.option("--dataset", required=True, help="Dataset to compare on")
@click.option("--judge", default="default", help="Judge to use for comparison")
@click.pass_context
def compare_all_models(ctx, dataset, judge):
    """Show statistics for all models on a dataset."""
    from .compare import get_all_model_stats_for_dataset
    
    db_url = ctx.obj["db_url"]
    engine = init_db(db_url)
    session = get_session(engine)
    
    try:
        stats_list = get_all_model_stats_for_dataset(session, dataset, judge)
        
        if not stats_list:
            console.print(f"[yellow]No evaluation data found for dataset '{dataset}' with judge '{judge}'.[/yellow]")
            return
        
        # Sort by avg score descending
        stats_list.sort(key=lambda s: s.avg_score or 0, reverse=True)
        
        table = Table(title=f"All Models on '{dataset}' (Judge: {judge})")
        table.add_column("Rank", style="cyan", justify="right")
        table.add_column("Model", style="white")
        table.add_column("Evaluations", justify="right")
        table.add_column("Avg Score", justify="right")
        table.add_column("Pass Rate", justify="right")
        table.add_column("Passed/Failed", justify="right")
        
        for rank, stats in enumerate(stats_list, 1):
            table.add_row(
                str(rank),
                stats.model_name,
                str(stats.total_evaluations),
                f"{stats.avg_score:.3f}" if stats.avg_score else "-",
                f"{stats.pass_rate:.1f}%" if stats.pass_rate else "-",
                f"{stats.total_passed}/{stats.total_failed}",
            )
        
        console.print(table)
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
    finally:
        session.close()


@compare.command("detailed")
@click.argument("model_a")
@click.argument("model_b")
@click.option("--dataset", required=True, help="Dataset to compare on")
@click.option("--judge", default="default", help="Judge to use for comparison")
@click.option("--limit", default=20, type=int, help="Number of prompts to show")
@click.pass_context
def detailed_comparison(ctx, model_a, model_b, dataset, judge, limit):
    """Show detailed per-prompt comparison between models."""
    from .compare import get_detailed_comparison
    
    db_url = ctx.obj["db_url"]
    engine = init_db(db_url)
    session = get_session(engine)
    
    try:
        comparisons = get_detailed_comparison(session, model_a, model_b, dataset, judge)[:limit]
        
        if not comparisons:
            console.print(f"[yellow]No comparison data found.[/yellow]")
            return
        
        table = Table(title=f"Detailed Comparison: {model_a} vs {model_b}")
        table.add_column("Prompt", style="white", max_width=40)
        table.add_column(f"{model_a} Score", justify="right")
        table.add_column(f"{model_b} Score", justify="right")
        table.add_column("Change", justify="right")
        
        for comp in comparisons:
            score_a = f"{comp['model_a_score']:.3f}" if comp['model_a_score'] is not None else "-"
            score_b = f"{comp['model_b_score']:.3f}" if comp['model_b_score'] is not None else "-"
            change = f"{comp['score_change']:+.3f}" if comp['score_change'] is not None else "-"
            
            table.add_row(
                comp['prompt_content'],
                score_a,
                score_b,
                change,
            )
        
        console.print(table)
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
    finally:
        session.close()


if __name__ == "__main__":
    cli()
