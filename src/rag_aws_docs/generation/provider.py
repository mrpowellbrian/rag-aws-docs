"""LLM generation provider abstraction.

Providers
---------
anthropic   Direct Anthropic API. Requires ANTHROPIC_API_KEY.
            Default model: claude-haiku-4-5-20251001.
bedrock     AWS Bedrock via boto3. Requires AWS credentials and the
            [bedrock] extra. Uses the Bedrock Converse API so the same
            message format works across all supported models.

GenerationResult carries the answer text and exact token counts from the
API response (not estimates). Cost is calculated from those counts and the
pricing constants in settings, so the observability layer always has
accurate numbers to log.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

from rag_aws_docs.config import LLMProvider, settings
from rag_aws_docs.storage.chroma import RetrievedChunk

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a precise technical assistant. Answer the user's question using only
the AWS documentation excerpts provided below. If the excerpts do not contain
enough information to answer the question, say so explicitly rather than
speculating. Cite the source file for each claim you make.\
"""


@dataclass(frozen=True)
class GenerationResult:
    answer: str
    input_tokens: int
    output_tokens: int
    cost_usd: float


def _build_context(chunks: list[RetrievedChunk]) -> str:
    parts: list[str] = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(f"[{i}] {chunk.source_path}\n{chunk.content}")
    return "\n\n---\n\n".join(parts)


def _calculate_cost(input_tokens: int, output_tokens: int) -> float:
    input_cost = (input_tokens / 1_000_000) * settings.llm_input_cost_per_mtok
    output_cost = (output_tokens / 1_000_000) * settings.llm_output_cost_per_mtok
    return round(input_cost + output_cost, 6)


class BaseGenerator(ABC):
    @abstractmethod
    def generate(self, query: str, chunks: list[RetrievedChunk]) -> GenerationResult:
        """Generate an answer grounded in the retrieved chunks."""
        ...


class AnthropicGenerator(BaseGenerator):
    def __init__(self, model: str | None = None) -> None:
        import anthropic

        self._model = model or settings.llm_model
        if not settings.anthropic_api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Add it to your .env file or environment."
            )
        self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        logger.info("using Anthropic model: %s", self._model)

    def generate(self, query: str, chunks: list[RetrievedChunk]) -> GenerationResult:
        context = _build_context(chunks)
        user_message = f"Documentation excerpts:\n\n{context}\n\nQuestion: {query}"

        response = self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )

        answer = response.content[0].text
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        cost = _calculate_cost(input_tokens, output_tokens)

        logger.debug(
            "generation complete: input_tokens=%d output_tokens=%d cost_usd=%.6f",
            input_tokens,
            output_tokens,
            cost,
        )
        return GenerationResult(
            answer=answer,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
        )


class BedrockGenerator(BaseGenerator):
    """AWS Bedrock generator using the Converse API.

    Requires the [bedrock] extra and AWS credentials resolvable by boto3
    (IAM role, environment variables, or ~/.aws/credentials).
    """

    def __init__(self, model: str | None = None) -> None:
        try:
            import boto3
        except ImportError as e:
            raise ImportError(
                "Bedrock generation requires the [bedrock] extra: "
                "pip install rag-aws-docs[bedrock]"
            ) from e

        self._model = model or settings.llm_model
        self._client = boto3.client("bedrock-runtime")
        logger.info("using Bedrock model: %s", self._model)

    def generate(self, query: str, chunks: list[RetrievedChunk]) -> GenerationResult:
        import boto3  # already confirmed importable in __init__

        context = _build_context(chunks)
        user_message = f"Documentation excerpts:\n\n{context}\n\nQuestion: {query}"

        response = self._client.converse(
            modelId=self._model,
            system=[{"text": SYSTEM_PROMPT}],
            messages=[{"role": "user", "content": [{"text": user_message}]}],
            inferenceConfig={"maxTokens": 1024},
        )

        answer = response["output"]["message"]["content"][0]["text"]
        usage = response["usage"]
        input_tokens = usage["inputTokens"]
        output_tokens = usage["outputTokens"]
        cost = _calculate_cost(input_tokens, output_tokens)

        return GenerationResult(
            answer=answer,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
        )


# ── Factory ───────────────────────────────────────────────────────────────────

_instance: BaseGenerator | None = None


def get_generator(provider: LLMProvider | None = None) -> BaseGenerator:
    """Return the configured generator, constructing it once per process."""
    global _instance
    if _instance is not None:
        return _instance

    resolved = provider or settings.llm_provider

    if resolved == LLMProvider.ANTHROPIC:
        _instance = AnthropicGenerator()
    elif resolved == LLMProvider.BEDROCK:
        _instance = BedrockGenerator()
    else:
        raise ValueError(f"unknown LLM provider: {resolved}")

    return _instance
