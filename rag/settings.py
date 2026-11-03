"""Configuration read from .env, plus the corpus list and the price table."""
import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

# The eleven Fair Work Ombudsman fact sheets, keyed by the file name used in data/raw/.
CORPUS_BASE_URL = "https://www.fairwork.gov.au/sites/default/files/migration/"
CORPUS_URLS = {
    path.rsplit("/", 1)[1]: CORPUS_BASE_URL + path
    for path in [
        "723/Introduction-to-the-national-employment-standards.pdf",
        "723/Maximum-weekly-hours.pdf",
        "723/Annual-leave.pdf",
        "723/Public-holidays.pdf",
        "723/Notice-of-termination-and-redundancy-pay.pdf",
        "723/Minimum-wages.pdf",
        "723/Long-service-leave.pdf",
        "723/Community-service-leave.pdf",
        "723/Unpaid-work.pdf",
        "723/Record-keeping-and-pay-slips.pdf",
        "724/Fair-Work-Information-Statement.pdf",
    ]
}


@dataclass(frozen=True)
class Settings:
    """Runtime settings: values from the environment plus fixed chunking and pricing constants."""

    openai_api_key: str
    chat_model: str
    embed_model: str
    database_url: str
    rag_min_similarity: float
    chunk_size: int = 1800
    chunk_overlap: int = 200
    # Approximate list prices in USD per 1M tokens (November 2026); update them if the models change.
    price_chat_input: float = 0.40
    price_chat_output: float = 1.60
    price_embed: float = 0.02


settings = Settings(
    openai_api_key=os.getenv("OPENAI_API_KEY", ""),
    chat_model=os.getenv("CHAT_MODEL", "gpt-4.1-mini"),
    embed_model=os.getenv("EMBED_MODEL", "text-embedding-3-small"),
    database_url=os.getenv("DATABASE_URL", "postgresql://rag:rag@127.0.0.1:5433/rag"),
    rag_min_similarity=float(os.getenv("RAG_MIN_SIMILARITY", "0.30")),
)
