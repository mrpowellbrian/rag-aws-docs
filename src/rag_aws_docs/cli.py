"""CLI entry point.

Commands
--------
rag-aws-docs ingest          Clone/pull corpus repos, chunk, embed, upsert.
rag-aws-docs query "<text>"  Retrieve + generate an answer.
rag-aws-docs metrics         Print a summary of the query log.
rag-aws-docs clear           Drop the Chroma collection (forces re-ingest).
"""

import logging
import sys
import time
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from rag_aws_docs.config import settings
from rag_aws_docs.observability.logging import log_query
from rag_aws_docs.observability.metrics import compute_metrics, format_summary

app = typer.Typer(
    name="rag-aws-docs",
    help="RAG system over AWS documentation.",
    add_completion=False,
)
console = Console()
err_console = Console(stderr=True)

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s %(name)s: %(message)s",
    stream=sys.stderr,
)


def _verbose_logger(verbose: bool) -> None:
    if verbose:
        logging.getLogger("rag_aws_docs").setLevel(logging.DEBUG)


@app.command()
def ingest(
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
    force: Annotated[
        bool,
        typer.Option("--force", help="Drop existing collection before ingesting."),
    ] = False,
) -> None:
    """Clone or pull corpus repos, chunk documents, embed, and store in Chroma."""
    _verbose_logger(verbose)

    from rag_aws_docs.embeddings.provider import get_embedder
    from rag_aws_docs.ingest.chunker import chunk_documents
    from rag_aws_docs.ingest.loader import load_corpus
    from rag_aws_docs.storage.chroma import VectorStore

    store = VectorStore()

    if force:
        console.print("[yellow]dropping existing collection[/yellow]")
        store.delete_collection()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("fetching corpus...", total=None)

        docs = load_corpus()
        progress.update(task, description=f"loaded {len(docs)} documents — chunking...")

        chunks = chunk_documents(docs)
        progress.update(
            task,
            description=f"chunked into {len(chunks)} chunks — embedding...",
        )

        embedder = get_embedder()
        texts = [c.content for c in chunks]

        # Embed in batches of 256 to show incremental progress.
        batch_size = 256
        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            all_embeddings.extend(embedder.embed(batch))
            progress.update(
                task,
                description=f"embedded {min(i + batch_size, len(texts))}/{len(texts)} chunks...",
            )

        progress.update(task, description="upserting to Chroma...")
        store.upsert(chunks, all_embeddings)

    console.print(
        f"[green]done.[/green] {len(chunks)} chunks from {len(docs)} documents "
        f"stored in [bold]{settings.chroma_path}[/bold]"
    )


@app.command()
def query(
    question: Annotated[str, typer.Argument(help="Question to answer.")],
    top_k: Annotated[
        Optional[int],
        typer.Option("--top-k", "-k", help="Number of chunks to retrieve."),
    ] = None,
    show_sources: Annotated[
        bool,
        typer.Option("--sources", help="Print retrieved source chunks after the answer."),
    ] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
    repo_filter: Annotated[
        Optional[str],
        typer.Option("--repo", help="Restrict retrieval to a specific repo slug."),
    ] = None,
) -> None:
    """Retrieve relevant documentation chunks and generate an answer."""
    _verbose_logger(verbose)

    from rag_aws_docs.embeddings.provider import get_embedder
    from rag_aws_docs.generation.provider import get_generator
    from rag_aws_docs.storage.chroma import VectorStore

    embedder = get_embedder()
    store = VectorStore()
    generator = get_generator()

    # ── Retrieval ─────────────────────────────────────────────────────────────
    t_retrieval_start = time.perf_counter()

    query_embedding = embedder.embed([question])[0]
    where = {"repo": repo_filter} if repo_filter else None
    chunks = store.query(query_embedding, top_k=top_k, where=where)

    retrieval_latency = time.perf_counter() - t_retrieval_start

    if not chunks:
        err_console.print(
            "[red]No results found.[/red] The collection may be empty — "
            "run [bold]rag-aws-docs ingest[/bold] first."
        )
        raise typer.Exit(1)

    # ── Generation ────────────────────────────────────────────────────────────
    t_generation_start = time.perf_counter()
    result = generator.generate(question, chunks)
    generation_latency = time.perf_counter() - t_generation_start

    # ── Output ────────────────────────────────────────────────────────────────
    console.print(result.answer)

    if show_sources:
        console.print("\n[dim]── sources ──────────────────────────────────────[/dim]")
        for i, chunk in enumerate(chunks, 1):
            console.print(
                f"[dim][{i}] {chunk.source_path}  score={chunk.score:.3f}[/dim]"
            )

    console.print(
        f"\n[dim]tokens: {result.input_tokens} in / {result.output_tokens} out  "
        f"cost: ${result.cost_usd:.4f}  "
        f"retrieval: {retrieval_latency:.2f}s  "
        f"generation: {generation_latency:.2f}s[/dim]"
    )

    # ── Log ───────────────────────────────────────────────────────────────────
    log_query(
        query=question,
        chunks=chunks,
        result=result,
        retrieval_latency=retrieval_latency,
        generation_latency=generation_latency,
    )


@app.command()
def metrics(
    log_file: Annotated[
        Optional[Path],
        typer.Option("--log-file", help="Path to query log. Defaults to settings.log_file."),
    ] = None,
) -> None:
    """Print a summary of cost and quality metrics from the query log."""
    summary = compute_metrics(log_file)
    console.print(format_summary(summary))


@app.command()
def clear(
    yes: Annotated[
        bool,
        typer.Option("--yes", "-y", help="Skip confirmation prompt."),
    ] = False,
) -> None:
    """Drop the Chroma collection. You will need to re-ingest afterwards."""
    if not yes:
        typer.confirm(
            f"Drop collection '{settings.chroma_collection}' at {settings.chroma_path}?",
            abort=True,
        )

    from rag_aws_docs.storage.chroma import VectorStore

    VectorStore().delete_collection()
    console.print("[green]Collection dropped.[/green] Run [bold]rag-aws-docs ingest[/bold] to rebuild.")
