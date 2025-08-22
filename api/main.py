import os
import json
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import psycopg
from psycopg.rows import dict_row
from pgvector.psycopg import register_vector

from openai import OpenAI


# ============================================================
#                   ENV & GLOBAL CLIENTS
# ============================================================

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
DATABASE_URL = os.environ.get("DATABASE_URL")

# Frontend origins:
# - FRONTEND_ORIGINS can be comma-separated list, e.g. "http://localhost:5173,https://myapp.vercel.app"
# - If unset, we'll allow localhost and any *.vercel.app via regex.
FRONTEND_ORIGINS = [
    o.strip()
    for o in os.environ.get("FRONTEND_ORIGINS", "").split(",")
    if o.strip()
]
FRONTEND_ALLOW_VERCEL_WILDCARD = os.environ.get("FRONTEND_ALLOW_VERCEL_WILDCARD", "true").lower() in ("1", "true", "yes")

# Embedding/Gen models:
# Use "text-embedding-3-small" (1536-d) by default to match common DB schema.
EMBED_MODEL = os.environ.get("EMBED_MODEL", "text-embedding-3-small")  # 1536
GEN_MODEL = os.environ.get("GEN_MODEL", "gpt-4o-mini")

# Optional: override embed dimension explicitly (otherwise inferred from model)
EMBED_DIM_ENV = os.environ.get("EMBED_DIM")  # e.g., "1536" or "3072"
EMBED_DIM = int(EMBED_DIM_ENV) if EMBED_DIM_ENV else (1536 if "small" in EMBED_MODEL else 3072)

if not OPENAI_API_KEY or not DATABASE_URL:
    raise RuntimeError("Missing OPENAI_API_KEY or DATABASE_URL")

# OpenAI client with a timeout to avoid hanging requests
client = OpenAI(api_key=OPENAI_API_KEY, timeout=45.0)


# ============================================================
#                         APP
# ============================================================

app = FastAPI(title="Happy Face Ads API", version="1.0.0")

# CORS: prefer explicit origins; also allow *.vercel.app if flag is true
cors_kwargs: Dict[str, Any] = dict(
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,  # flip to True only if you use cookies
)
if FRONTEND_ORIGINS:
    cors_kwargs["allow_origins"] = FRONTEND_ORIGINS
else:
    cors_kwargs["allow_origins"] = ["http://localhost:5173", "http://localhost:3000"]

if FRONTEND_ALLOW_VERCEL_WILDCARD:
    cors_kwargs["allow_origin_regex"] = r"^https://.*\.vercel\.app$"

app.add_middleware(CORSMiddleware, **cors_kwargs)


# ============================================================
#                      DB HELPERS / INIT
# ============================================================

def get_conn():
    """
    Returns a psycopg connection with dict rows and pgvector registered.
    """
    conn = psycopg.connect(DATABASE_URL, row_factory=dict_row)
    register_vector(conn)  # make Python list <-> pgvector work
    return conn


DB_BOOTSTRAP_SQL = f"""
-- Extensions
create extension if not exists vector;
create extension if not exists pgcrypto;

-- Documents table
create table if not exists public.documents (
  id uuid primary key default gen_random_uuid(),
  brand_id uuid null,
  source_type text not null check (source_type in ('internal','inspo')),
  title text not null,
  url text null,
  text_content text not null,
  created_at timestamptz default now()
);

-- Embeddings table (uses EMBED_DIM)
create table if not exists public.embeddings (
  id bigserial primary key,
  doc_id uuid not null references public.documents(id) on delete cascade,
  chunk_idx int not null,
  text_chunk text not null,
  embedding vector({EMBED_DIM}) not null,
  created_at timestamptz default now()
);

-- Briefs table
create table if not exists public.briefs (
  id bigserial primary key,
  inputs jsonb not null,
  output_md jsonb not null,
  created_at timestamptz default now()
);

-- Fast vector search index (HNSW; available on PG 16/Supabase). If unavailable, switch to ivfflat.
do $$
begin
  begin
    create index if not exists embeddings_embedding_hnsw
      on public.embeddings using hnsw (embedding vector_l2_ops);
  exception when others then
    -- fallback try IVF if HNSW isn't available
    begin
      create index if not exists embeddings_embedding_ivf
        on public.embeddings using ivfflat (embedding vector_l2_ops) with (lists = 100);
    exception when others then
      -- ignore if neither is available
      null;
    end;
  end;
end $$;
"""


@app.on_event("startup")
def bootstrap_db():
    """
    Create required extensions/tables if they don't exist.
    This is idempotent and safe to run on each startup.
    """
    try:
        with get_conn() as conn:
            conn.execute(DB_BOOTSTRAP_SQL)
    except Exception as e:
        # If bootstrap fails, it's still useful to start the app; errors will surface on first use.
        # But we raise here to make it obvious in Render logs.
        raise RuntimeError(f"DB bootstrap failed: {e}") from e


# ============================================================
#                     OPENAI / EMBEDDINGS
# ============================================================

def embed(texts: List[str]) -> List[List[float]]:
    """
    Returns list of embeddings (list[float]) for the given texts.
    """
    if not texts:
        return []
    try:
        resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
        vectors = [d.embedding for d in resp.data]
        # guard dimension
        for v in vectors:
            if len(v) != EMBED_DIM:
                raise HTTPException(
                    status_code=500,
                    detail=f"Embedding dimension mismatch: got {len(v)}, expected {EMBED_DIM}. "
                           f"Model={EMBED_MODEL}. Consider setting EMBED_DIM env to override."
                )
        return vectors
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Embedding error: {str(e)[:250]}")


# ============================================================
#                         MODELS
# ============================================================

class GenerateReq(BaseModel):
    project_type: str            # "product" | "service"
    ad_type: str                 # "static" | "video"
    industry: str
    product_name: Optional[str] = None
    what_it_does: Optional[str] = None
    service: Optional[str] = None
    audience: str
    platforms: List[str] = []    # TikTok, Reels, Shorts...
    content_style: List[str] = []# UGC, Talking Head, Faceless Video, Skit, Parody
    levers: List[str] = []       # comedic levers
    n: int = 12
    brand_id: Optional[str] = None

class RewriteReq(BaseModel):
    concept_text: Dict[str, Any]  # {hook,premise,escalations[],cta}
    duration_s: int = 30
    platform: str = "TikTok"
    style: str = "Faceless Video"

class IngestText(BaseModel):
    title: str
    text: str
    source_type: str              # 'internal' | 'inspo'
    brand_id: Optional[str] = None
    url: Optional[str] = None


# ============================================================
#                       SMALL UTILS
# ============================================================

def chunk_text(txt: str, max_chars: int = 2500, overlap: int = 250) -> List[str]:
    txt = (txt or "").strip()
    if not txt:
        return []
    if len(txt) <= max_chars:
        return [txt]
    chunks, start = [], 0
    while start < len(txt):
        end = min(len(txt), start + max_chars)
        chunks.append(txt[start:end])
        start = max(0, end - overlap)
    return chunks


def _build_inputs_summary(req: GenerateReq) -> str:
    parts = [
        f"Project Type: {req.project_type}",
        f"Ad Type: {req.ad_type}",
        f"Industry: {req.industry}",
        f"Product Name: {req.product_name or ''}",
        f"What It Does: {req.what_it_does or ''}",
        f"Service: {req.service or ''}",
        f"Audience: {req.audience}",
        f"Platforms: {', '.join(req.platforms) if req.platforms else ''}",
        f"Content Style: {', '.join(req.content_style) if req.content_style else ''}",
        f"Funny Levers (HARD 10/10): {', '.join(req.levers) if req.levers else ''}",
    ]
    return "\n".join([p for p in parts if p.strip()])


PITCH_SYSTEM = (
    "You are Happy Face Ads’ writers-room engine.\n"
    "- Swing HUGE (10/10 absurdity) while staying producible for social ads.\n"
    "- Obey the brief literally (industry, audience, ad_type, platforms, content_style).\n"
    "- Keep each concept tight and shootable.\n"
)

JSON_INSTRUCTIONS = (
    "Return STRICT JSON with this shape:\n"
    "{\n"
    '  "concepts": [\n'
    '    {\n'
    '      "hook": "string",\n'
    '      "premise": "string",\n'
    '      "escalations": ["beat 1","beat 2","beat 3"],\n'
    '      "cta": "string"\n'
    "    }\n"
    "  ]\n"
    "}\n"
    "NO markdown, no commentary—JSON ONLY."
)


def _call_for_concepts(context: str, req: GenerateReq, needed: int) -> List[Dict[str, Any]]:
    user = (
        f"CONTEXT (summarized training snippets):\n{context[:12000]}\n\n"
        f"BRIEF (use these details exactly):\n{_build_inputs_summary(req)}\n\n"
        f"TASK: Create exactly {needed} distinct concepts. "
        "Each must match ad_type and content_style. Platforms should feel native. "
        "Escalations must be three crisp beats that heighten the bit.\n\n"
        + JSON_INSTRUCTIONS
    )

    try:
        chat = client.chat.completions.create(
            model=GEN_MODEL,
            temperature=0.7,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": PITCH_SYSTEM},
                {"role": "user",   "content": user},
            ],
        )
        raw = chat.choices[0].message.content or "{}"
        parsed = json.loads(raw)
        concepts = parsed.get("concepts", [])
        clean: List[Dict[str, Any]] = []
        for c in concepts:
            hook = (c.get("hook") or "").strip()
            prem = (c.get("premise") or "").strip()
            esc = c.get("escalations") or []
            cta = (c.get("cta") or "").strip()
            if hook and prem and isinstance(esc, list) and len(esc) >= 3 and cta:
                clean.append({"hook": hook, "premise": prem, "escalations": esc[:3], "cta": cta})
        return clean
    except Exception as e:
        # Return empty so caller can attempt a top-up call
        return []


# ============================================================
#                         ROUTES
# ============================================================

@app.get("/health")
def health():
    return {"ok": True, "embed_model": EMBED_MODEL, "embed_dim": EMBED_DIM}


@app.post("/ingest/text")
def ingest_text(req: IngestText):
    if req.source_type not in ("internal", "inspo"):
        raise HTTPException(status_code=400, detail="source_type must be 'internal' or 'inspo'")

    chunks = chunk_text(req.text)
    vecs = embed(chunks)

    try:
        with get_conn() as conn, conn.transaction():
            doc = conn.execute(
                """
                insert into documents(brand_id, source_type, title, url, text_content)
                values (%s,%s,%s,%s,%s)
                returning id
                """,
                (req.brand_id, req.source_type, req.title, req.url, req.text),
            ).fetchone()

            for i, (chunk, v) in enumerate(zip(chunks, vecs)):
                conn.execute(
                    "insert into embeddings(doc_id, chunk_idx, text_chunk, embedding) values (%s,%s,%s,%s)",
                    (doc["id"], i, chunk, v),  # pgvector adapter handles Python list
                )

        return {"status": "ok", "chunks": len(chunks), "doc_id": doc["id"]}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingest failed: {str(e)[:300]}")


@app.post("/generate/concepts")
def generate_concepts(req: GenerateReq):
    try:
        # 1) Build retrieval query from inputs
        fields = [
            req.project_type, req.ad_type, req.industry, req.audience,
            " ".join(req.platforms or []),
            " ".join(req.content_style or []),
            " ".join(req.levers or []),
            (req.product_name or ""), (req.what_it_does or ""), (req.service or "")
        ]
        q = " | ".join([f for f in fields if f])
        q_vec_list = embed([q])
        if not q_vec_list:
            raise HTTPException(status_code=400, detail="Could not embed query.")
        q_vec = q_vec_list[0]

        # 2) Pull nearest training chunks (internal then inspo)
        with get_conn() as conn:
            internal = conn.execute(
                f"""
                select text_chunk from embeddings e
                join documents d on d.id = e.doc_id
                where d.source_type = 'internal'
                order by e.embedding <=> %s::vector
                limit 10
                """,
                (q_vec,),
            ).fetchall()

            inspo = conn.execute(
                f"""
                select text_chunk from embeddings e
                join documents d on d.id = e.doc_id
                where d.source_type = 'inspo'
                order by e.embedding <=> %s::vector
                limit 6
                """,
                (q_vec,),
            ).fetchall()

        ctx = "\n\n".join([r["text_chunk"] for r in (internal + inspo)]) or "No prior docs yet."

        # 3) Call model for exactly n concepts; top-up if short
        concepts: List[Dict[str, Any]] = _call_for_concepts(ctx, req, req.n)
        if len(concepts) < req.n:
            missing = req.n - len(concepts)
            more = _call_for_concepts(ctx, req, missing)
            concepts.extend(more)

        # Last resort: avoid empty UI by duplicating if we got at least one
        while len(concepts) < req.n and concepts:
            concepts.append(concepts[len(concepts) % len(concepts)])

        # 4) Persist brief
        with get_conn() as conn, conn.transaction():
            brief = conn.execute(
                "insert into briefs(inputs, output_md) values (%s,%s) returning id",
                (json.dumps(req.dict()), json.dumps({"concepts": concepts})),
            ).fetchone()

        return {"brief_id": brief["id"], "concepts": concepts}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"generate_concepts failed: {str(e)[:350]}")


@app.post("/generate/rewrite")
def generate_rewrite(req: RewriteReq):
    prompt = {
        "concept": req.concept_text,
        "instructions": f"Rewrite into a {req.duration_s}s {req.platform} script with sections: Hook, Premise, Escalations (3 beats), CTA. "
                        f"Style: {req.style}. JSON only."
    }
    try:
        chat = client.chat.completions.create(
            model=GEN_MODEL,
            temperature=0.7,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You are Happy Face Ads’ rewrite engine. Keep it punchy and producible."},
                {"role": "user", "content": json.dumps(prompt)}
            ]
        )
        data = json.loads(chat.choices[0].message.content or "{}")
    except Exception:
        data = {"hook": "", "premise": "", "escalations": [], "cta": ""}

    return {"script": data}
