import os
import json
import time
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl

import psycopg
from psycopg.rows import dict_row
from pgvector.psycopg import register_vector

from openai import OpenAI
from twelvelabs import TwelveLabs


# ============================================================
#                   ENV & GLOBAL CLIENTS
# ============================================================

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
DATABASE_URL = os.environ.get("DATABASE_URL")

# Frontend origins (comma-separated). If unset, allow localhost and *.vercel.app via regex.
FRONTEND_ORIGINS = [
    o.strip()
    for o in os.environ.get("FRONTEND_ORIGINS", "").split(",")
    if o.strip()
]
FRONTEND_ALLOW_VERCEL_WILDCARD = os.environ.get(
    "FRONTEND_ALLOW_VERCEL_WILDCARD", "true"
).lower() in ("1", "true", "yes")

# Embedding/Gen models:
# Default to 1536-d to match common DB schema.
EMBED_MODEL = os.environ.get("EMBED_MODEL", "text-embedding-3-small")  # 1536
GEN_MODEL = os.environ.get("GEN_MODEL", "gpt-4o-mini")

# Optional dimension override (else inferred from model)
EMBED_DIM_ENV = os.environ.get("EMBED_DIM")  # "1536" or "3072"
EMBED_DIM = int(EMBED_DIM_ENV) if EMBED_DIM_ENV else (1536 if "small" in EMBED_MODEL else 3072)

if not OPENAI_API_KEY or not DATABASE_URL:
    raise RuntimeError("Missing OPENAI_API_KEY or DATABASE_URL")

# OpenAI client (timeouts prevent hangs)
client = OpenAI(api_key=OPENAI_API_KEY, timeout=45.0)

# Twelve Labs (required for video features)
TWELVELABS_API_KEY = os.environ.get("TWELVELABS_API_KEY")
if not TWELVELABS_API_KEY:
    raise RuntimeError("Missing TWELVELABS_API_KEY")
tl_client = TwelveLabs(api_key=TWELVELABS_API_KEY)


# ============================================================
#                         APP
# ============================================================

app = FastAPI(title="Happy Face Ads API", version="1.1.0")

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
    """Psycopg connection with dict rows + pgvector adapter."""
    conn = psycopg.connect(DATABASE_URL, row_factory=dict_row)
    register_vector(conn)  # python list <-> pgvector
    return conn


DB_BOOTSTRAP_SQL = f"""
-- Extensions
create extension if not exists vector;
create extension if not exists pgcrypto;

-- Documents: your knowledge store
create table if not exists public.documents (
  id uuid primary key default gen_random_uuid(),
  brand_id uuid null,
  source_type text not null check (source_type in ('internal','inspo')),
  title text not null,
  url text null,
  text_content text not null,
  created_at timestamptz default now()
);

-- Embeddings: chunked text vectors
create table if not exists public.embeddings (
  id bigserial primary key,
  doc_id uuid not null references public.documents(id) on delete cascade,
  chunk_idx int not null,
  text_chunk text not null,
  embedding vector({EMBED_DIM}) not null,
  created_at timestamptz default now()
);

-- Briefs: generated outputs
create table if not exists public.briefs (
  id bigserial primary key,
  inputs jsonb not null,
  output_md jsonb not null,
  created_at timestamptz default now()
);

-- Videos: Twelve Labs ingestion tracking
create table if not exists public.videos (
  id uuid primary key default gen_random_uuid(),
  source_url text not null,
  index_id text not null,
  video_id text not null,
  status text not null default 'processing', -- processing|ready|failed|timeout
  created_at timestamptz default now()
);

-- Video analyses: raw JSON results from TL
create table if not exists public.video_analyses (
  id bigserial primary key,
  video_id_ref uuid not null references public.videos(id) on delete cascade,
  analysis_type text not null,    -- 'open-ended' | 'gist' | 'summary'
  prompt text,
  output_json jsonb not null,
  created_at timestamptz default now()
);

-- Vector index (try HNSW, fallback to IVF)
do $$
begin
  begin
    create index if not exists embeddings_embedding_hnsw
      on public.embeddings using hnsw (embedding vector_l2_ops);
  exception when others then
    begin
      create index if not exists embeddings_embedding_ivf
        on public.embeddings using ivfflat (embedding vector_l2_ops) with (lists = 100);
    exception when others then
      null;
    end;
  end;
end $$;
"""

@app.on_event("startup")
def bootstrap_db():
    with get_conn() as conn:
        conn.execute(DB_BOOTSTRAP_SQL)


# ============================================================
#                     OPENAI / EMBEDDINGS
# ============================================================

def embed(texts: List[str]) -> List[List[float]]:
    """Return embeddings for texts; enforce dimension sanity."""
    if not texts:
        return []
    try:
        resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
        vectors = [d.embedding for d in resp.data]
        for v in vectors:
            if len(v) != EMBED_DIM:
                raise HTTPException(
                    status_code=500,
                    detail=f"Embedding dimension mismatch: got {len(v)}, expected {EMBED_DIM} (model={EMBED_MODEL})."
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

# Twelve Labs requests
class VideoIngestReq(BaseModel):
    url: HttpUrl
    index_name: Optional[str] = "happyface-default"
    visual: bool = True
    audio: bool = True

class VideoStatusReq(BaseModel):
    task_id: str
    video_row_id: str

class VideoAnalyzeReq(BaseModel):
    video_row_id: str
    mode: str = "open-ended"  # 'open-ended' | 'gist' | 'summary'
    prompt: Optional[str] = None  # required if open-ended


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


# ============================================================
#               OPENAI CONCEPT GENERATION HELPERS
# ============================================================

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
    except Exception:
        return []


# ============================================================
#                   TWELVE LABS HELPERS
# ============================================================

def tl_get_or_create_index(index_name: str = "happyface-default", use_visual=True, use_audio=True) -> str:
    """
    Create (or reuse) a TwelveLabs index and return its id.
    """
    models = []
    if use_visual: models.append("visual")
    if use_audio:  models.append("audio")
    try:
        idx = tl_client.indexes.create(index_name=index_name, models=models)
        return idx.id
    except Exception as e:
        # If it already exists and SDK raises, surface a clear hint.
        raise HTTPException(status_code=500, detail="Index creation failed. Create once in the TL console or store INDEX_ID.")

def tl_upload_video(index_id: str, video_url: str) -> dict:
    task = tl_client.tasks.create(index_id=index_id, video_url=video_url)
    return {"task_id": task.id, "video_id": task.video_id}

def tl_wait_until_ready(task_id: str, sleep_sec: int = 5, timeout_sec: int = 1800) -> str:
    start = time.time()
    while True:
        res = tl_client.tasks.wait_for_done(task_id=task_id, sleep_interval=sleep_sec)
        status = getattr(res, "status", None) or (res.get("status") if isinstance(res, dict) else None)
        if status == "ready":
            return "ready"
        if time.time() - start > timeout_sec:
            return status or "timeout"


# ============================================================
#                         ROUTES
# ============================================================

@app.get("/health")
def health():
    return {
        "ok": True,
        "embed_model": EMBED_MODEL,
        "embed_dim": EMBED_DIM
    }


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
                    (doc["id"], i, chunk, v),
                )
        return {"status": "ok", "chunks": len(chunks), "doc_id": doc["id"]}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingest failed: {str(e)[:300]}")


@app.post("/generate/concepts")
def generate_concepts(req: GenerateReq):
    try:
        # Build retrieval query
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

        # Retrieve nearest chunks
        with get_conn() as conn:
            internal = conn.execute(
                """
                select text_chunk from embeddings e
                join documents d on d.id = e.doc_id
                where d.source_type = 'internal'
                order by e.embedding <=> %s::vector
                limit 10
                """,
                (q_vec,),
            ).fetchall()

            inspo = conn.execute(
                """
                select text_chunk from embeddings e
                join documents d on d.id = e.doc_id
                where d.source_type = 'inspo'
                order by e.embedding <=> %s::vector
                limit 6
                """,
                (q_vec,),
            ).fetchall()

        ctx = "\n\n".join([r["text_chunk"] for r in (internal + inspo)]) or "No prior docs yet."

        # Generate exactly n concepts; top-up if short
        concepts: List[Dict[str, Any]] = _call_for_concepts(ctx, req, req.n)
        if len(concepts) < req.n:
            missing = req.n - len(concepts)
            more = _call_for_concepts(ctx, req, missing)
            concepts.extend(more)

        while len(concepts) < req.n and concepts:
            concepts.append(concepts[len(concepts) % len(concepts)])

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


# --------------------- Twelve Labs: Video Pipeline ----------------------

@app.post("/videos/ingest")
def videos_ingest(req: VideoIngestReq):
    """Create/reuse index, upload video URL, store tracking row."""
    try:
        index_id = tl_get_or_create_index(req.index_name, req.visual, req.audio)
        ids = tl_upload_video(index_id, str(req.url))
        with get_conn() as conn, conn.transaction():
            row = conn.execute(
                """
                insert into videos (source_url, index_id, video_id, status)
                values (%s, %s, %s, %s)
                returning id
                """,
                (str(req.url), index_id, ids["video_id"], "processing"),
            ).fetchone()
        return {"video_row_id": row["id"], "index_id": index_id, "video_id": ids["video_id"], "task_id": ids["task_id"]}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Video ingest failed: {str(e)[:300]}")


@app.post("/videos/status")
def videos_status(req: VideoStatusReq):
    """Poll TL until 'ready'; mark the DB row when done."""
    try:
        status = tl_wait_until_ready(req.task_id)
        if status == "ready":
            with get_conn() as conn, conn.transaction():
                conn.execute("update videos set status='ready' where id=%s", (req.video_row_id,))
        elif status in ("failed", "timeout"):
            with get_conn() as conn, conn.transaction():
                conn.execute("update videos set status=%s where id=%s", (status, req.video_row_id))
        return {"status": status}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)[:300]}")


@app.post("/videos/analyze")
def videos_analyze(req: VideoAnalyzeReq):
    """
    Analyze an indexed video via Twelve Labs, save raw JSON,
    then flatten text into documents/embeddings so it fuels your RAG.
    """
    try:
        # Load video row
        with get_conn() as conn:
            v = conn.execute("select * from videos where id=%s", (req.video_row_id,)).fetchone()
            if not v:
                raise HTTPException(status_code=404, detail="Video not found")
            if v["status"] != "ready":
                raise HTTPException(status_code=400, detail=f"Video status is {v['status']}, not ready")

        video_id = v["video_id"]
        analysis_type = req.mode
        output_json: Dict[str, Any]

        if req.mode == "gist":
            # Titles / topics / hashtags
            res = tl_client.gist(video_id=video_id, types=["title", "topic", "hashtag"])
            output_json = res if isinstance(res, dict) else json.loads(json.dumps(res, default=lambda o: o.__dict__))
        elif req.mode == "summary":
            # Summaries / chapters / highlights
            res = tl_client.summarize(video_id=video_id, types=["summary", "chapters", "highlights"])
            output_json = res if isinstance(res, dict) else json.loads(json.dumps(res, default=lambda o: o.__dict__))
        else:
            # Open-ended (non-streaming)
            if not (req.prompt and req.prompt.strip()):
                raise HTTPException(status_code=400, detail="prompt is required for open-ended mode")
            res = tl_client.analyze(video_id=video_id, prompt=req.prompt, temperature=0.4)
            output_json = res if isinstance(res, dict) else json.loads(json.dumps(res, default=lambda o: o.__dict__))

        # Save raw analysis
        with get_conn() as conn, conn.transaction():
            ana = conn.execute(
                """
                insert into video_analyses (video_id_ref, analysis_type, prompt, output_json)
                values (%s,%s,%s,%s)
                returning id
                """,
                (req.video_row_id, analysis_type, req.prompt, json.dumps(output_json)),
            ).fetchone()

        # Flatten JSON → readable text
        def _flatten_text(obj):
            if obj is None:
                return ""
            if isinstance(obj, str):
                return obj
            if isinstance(obj, (int, float)):
                return str(obj)
            if isinstance(obj, list):
                return "\n".join(_flatten_text(x) for x in obj)
            if isinstance(obj, dict):
                return "\n".join(_flatten_text(v) for v in obj.values())
            return ""

        readable = _flatten_text(output_json).strip()

        # Store readable text into documents/embeddings so /generate/concepts can retrieve it
        if readable:
            chunks = chunk_text(readable)
            vecs = embed(chunks)
            with get_conn() as conn, conn.transaction():
                doc = conn.execute(
                    """
                    insert into documents(brand_id, source_type, title, url, text_content)
                    values (%s,%s,%s,%s,%s)
                    returning id
                    """,
                    (None, "internal", f"Video Analysis {ana['id']}", v["source_url"], readable),
                ).fetchone()
                for i, (chunk, vec) in enumerate(zip(chunks, vecs)):
                    conn.execute(
                        "insert into embeddings(doc_id, chunk_idx, text_chunk, embedding) values (%s,%s,%s,%s)",
                        (doc["id"], i, chunk, vec),
                    )

        return {"analysis_id": ana["id"], "added_to_rag": bool(readable)}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Video analyze failed: {str(e)[:350]}")
