"""Loads AWS documentation from cloned awsdocs GitHub repositories.

Each entry in the corpus is a (repo_slug, subfolder) pair. The loader
clones the repo on first run and does a fast-forward pull on subsequent
runs. Only .rst and .md files are yielded; binary and generated files
are skipped.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path

import git

from rag_aws_docs.config import DEFAULT_CORPUS, settings

logger = logging.getLogger(__name__)

GITHUB_BASE = "https://github.com"
# File extensions treated as documentation source.
DOC_SUFFIXES = {".md", ".rst"}


@dataclass(frozen=True)
class Document:
    """A single source document before chunking."""

    content: str
    # Relative path within the repo, e.g. "doc_source/lambda-intro.md"
    source_path: str
    # GitHub repo slug, e.g. "awsdocs/aws-lambda-developer-guide"
    repo: str
    metadata: dict[str, str] = field(default_factory=dict)


def _clone_or_pull(repo_slug: str, target: Path) -> None:
    """Clone repo if absent, otherwise pull latest changes."""
    url = f"{GITHUB_BASE}/{repo_slug}.git"

    if target.exists():
        logger.info("pulling %s", repo_slug)
        repo = git.Repo(target)
        origin = repo.remotes.origin
        origin.pull(ff_only=True)
    else:
        logger.info("cloning %s → %s", repo_slug, target)
        target.parent.mkdir(parents=True, exist_ok=True)
        git.Repo.clone_from(url, target, depth=1, single_branch=True)


def _iter_doc_files(root: Path) -> list[Path]:
    """Return all documentation files under root, sorted for determinism."""
    return sorted(p for p in root.rglob("*") if p.suffix in DOC_SUFFIXES and p.is_file())


def load_corpus(
    corpus: list[tuple[str, str]] | None = None,
    data_path: Path | None = None,
) -> list[Document]:
    """Clone/pull each corpus repo and return all documents.

    Args:
        corpus: List of (repo_slug, subfolder) pairs. Defaults to
                DEFAULT_CORPUS from config.
        data_path: Local root for cloned repos. Defaults to settings.data_path.

    Returns:
        Flat list of Document objects, one per source file.
    """
    corpus = corpus or DEFAULT_CORPUS
    data_path = data_path or settings.data_path

    documents: list[Document] = []

    for repo_slug, subfolder in corpus:
        repo_name = repo_slug.split("/")[-1]
        repo_dir = data_path / repo_name
        doc_dir = repo_dir / subfolder

        _clone_or_pull(repo_slug, repo_dir)

        if not doc_dir.exists():
            logger.warning("subfolder %s not found in %s, skipping", subfolder, repo_slug)
            continue

        files = _iter_doc_files(doc_dir)
        logger.info("found %d doc files in %s/%s", len(files), repo_slug, subfolder)

        for file_path in files:
            try:
                content = file_path.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                logger.warning("skipping %s: %s", file_path, exc)
                continue

            if not content.strip():
                continue

            rel_path = file_path.relative_to(repo_dir)
            documents.append(
                Document(
                    content=content,
                    source_path=str(rel_path),
                    repo=repo_slug,
                    metadata={
                        "repo": repo_slug,
                        "source_path": str(rel_path),
                        "filename": file_path.name,
                    },
                )
            )

    logger.info("loaded %d documents total", len(documents))
    return documents
