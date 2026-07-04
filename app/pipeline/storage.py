"""
Persistence layer: stores JD extractions and CV extractions in Postgres
with pgvector for embedding columns. Same function signatures as the old
JSON-file version — routers are untouched.
"""

import json
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional

from helpers.database import get_pool

def _parse_embedding(value) -> list:
    """Convert stored vector string back to a list of floats."""
    if value is None:
        return None
    if isinstance(value, list):
        return value
    # asyncpg returns it as a string like '[-0.03, 0.12, ...]'
    import json
    return json.loads(str(value).replace("(", "[").replace(")", "]"))

# -----------------------------
# Jobs (JD)
# -----------------------------

async def save_job(
    jd_text: str,
    jd_extracted: Dict,
    jd_query: str,
    jd_embedding: Optional[list] = None,
) -> str:
    job_id = uuid.uuid4().hex[:12]
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO jobs (job_id, jd_text, extracted, query, jd_embedding, created_at)
            VALUES ($1, $2, $3::jsonb, $4, $5::vector, $6)
            """,
            job_id,
            jd_text,
            json.dumps(jd_extracted),
            jd_query,
            str(jd_embedding) if jd_embedding is not None else None,
            datetime.now(timezone.utc),
        )
    return job_id


async def get_job(job_id: str) -> Optional[Dict]:
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM jobs WHERE job_id = $1", job_id
        )
    if row is None:
        return None
    return _job_row_to_dict(row)


async def list_jobs() -> List[Dict]:
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch("SELECT * FROM jobs ORDER BY created_at")
    return [_job_row_to_dict(r) for r in rows]


def _job_row_to_dict(row) -> Dict:
    d = dict(row)
    if isinstance(d.get("extracted"), str):
        d["extracted"] = json.loads(d["extracted"])
    if d.get("created_at"):
        d["created_at"] = d["created_at"].isoformat()
    d["jd_embedding"] = _parse_embedding(d.get("jd_embedding"))
    return d


# -----------------------------
# Candidates (CV)
# -----------------------------

async def save_candidate(
    file_id: str,
    filename: str,
    cv_parsed: Dict,
    cv_embedding: Optional[list] = None,
) -> str:
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO candidates
                (candidate_id, file_id, filename, parsed, cv_embedding, created_at)
            VALUES ($1, $2, $3, $4::jsonb, $5::vector, $6)
            ON CONFLICT (candidate_id) DO NOTHING
            """,
            cv_parsed["id"],
            file_id,
            filename,
            json.dumps(cv_parsed),
            str(cv_embedding) if cv_embedding is not None else None,
            datetime.now(timezone.utc),
        )
    return cv_parsed["id"]


async def get_candidate(candidate_id: str) -> Optional[Dict]:
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM candidates WHERE candidate_id = $1", candidate_id
        )
    if row is None:
        return None
    return _candidate_row_to_dict(row)


async def list_candidates() -> List[Dict]:
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch("SELECT * FROM candidates ORDER BY created_at")
    return [_candidate_row_to_dict(r) for r in rows]


async def get_candidates_with_embeddings() -> List[Dict]:
    """
    Returns only candidates that already have a stored embedding.
    Used by the ranker's fallback path (explicit candidate_ids) instead of
    re-embedding on every match request.
    """
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT * FROM candidates WHERE cv_embedding IS NOT NULL ORDER BY created_at"
        )
    return [_candidate_row_to_dict(r) for r in rows]


async def get_top_candidates_by_similarity(jd_embedding: list, limit: int) -> List[Dict]:
    """
    The "reduce the number of CVs" step: ask Postgres/pgvector directly
    for the `limit` candidates with the highest cosine similarity to the
    JD embedding, using the HNSW index (idx_candidates_cv_embedding) —
    this scales to a large candidate pool without pulling every row (and
    every embedding) into Python first.

    `cv_embedding <=> $1` is pgvector's cosine *distance* (0 = identical,
    2 = opposite), so `1 - distance` gives the cosine *similarity* score.
    """
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT *, 1 - (cv_embedding <=> $1::vector) AS semantic_score
            FROM candidates
            WHERE cv_embedding IS NOT NULL
            ORDER BY cv_embedding <=> $1::vector
            LIMIT $2
            """,
            str(jd_embedding),
            limit,
        )
    results = []
    for row in rows:
        d = _candidate_row_to_dict(row)
        d["semantic_score"] = float(row["semantic_score"])
        results.append(d)
    return results


def _candidate_row_to_dict(row) -> Dict:
    d = dict(row)
    if isinstance(d.get("parsed"), str):
        d["parsed"] = json.loads(d["parsed"])
    if d.get("created_at"):
        d["created_at"] = d["created_at"].isoformat()
    d["cv_embedding"] = _parse_embedding(d.get("cv_embedding"))
    return d