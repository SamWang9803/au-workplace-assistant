"""OpenAI calls used by the pipeline: embeddings and strict-JSON chat, plus cost accounting."""
import json
import time
from dataclasses import dataclass
from functools import cache

from openai import OpenAI

from rag.settings import settings


@dataclass
class ChatReply:
    """A parsed chat reply (None when the model returned no usable JSON) with its token usage and latency."""

    data: dict | None
    prompt_tokens: int
    completion_tokens: int
    latency_ms: int


@cache
def _client() -> OpenAI:
    """Create the shared client; the SDK retries 429 and 5xx responses three times with exponential backoff."""
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is not set: copy .env.example to .env and fill it in.")
    return OpenAI(api_key=settings.openai_api_key, max_retries=3, timeout=60)


def embed(texts: list[str]) -> tuple[list[list[float]], int]:
    """Embed a batch of texts; return one vector per text, in input order, and the tokens used."""
    response = _client().embeddings.create(model=settings.embed_model, input=texts)
    vectors = [item.embedding for item in sorted(response.data, key=lambda item: item.index)]
    return vectors, response.usage.total_tokens


def chat_json(system: str, user: str, schema: dict) -> ChatReply:
    """Ask the chat model, at temperature 0, for a reply that must follow the given JSON schema."""
    start = time.perf_counter()
    response = _client().chat.completions.create(
        model=settings.chat_model,
        temperature=0,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        response_format={"type": "json_schema", "json_schema": {"name": "reply", "strict": True, "schema": schema}},
    )
    latency_ms = round((time.perf_counter() - start) * 1000)
    try:
        data = json.loads(response.choices[0].message.content or "")
    except json.JSONDecodeError:
        data = None  # a safety refusal or a truncated reply: the schema could not be honoured
    return ChatReply(data, response.usage.prompt_tokens, response.usage.completion_tokens, latency_ms)


def cost_usd(prompt_tokens: int, completion_tokens: int, embed_tokens: int) -> float:
    """Convert token counts to US dollars at the list prices in settings."""
    return (
        prompt_tokens * settings.price_chat_input
        + completion_tokens * settings.price_chat_output
        + embed_tokens * settings.price_embed
    ) / 1_000_000
