import os, json
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import psycopg
from psycopg.rows import dict_row
from openai import OpenAI

# Env setup
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
DATABASE_URL = os.environ.get("DATABASE_URL")
FRONTEND_ORIGIN = os.environ.get("FRONTEND_ORIGIN", "*")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "text-embedding-3-large")
GEN_MODEL = os.environ.get("GEN_MODEL", "gpt-4o-mini")

if not (OPENAI_API_KEY and DATABASE_URL):
    raise RuntimeError("Missing OPENAI_API_KEY or DATABASE_URL env var")

client = OpenAI(api_key=OPENAI_API_KEY)
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_ORIGIN, "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_conn():
    return psycopg.connect(DATABASE_URL, row_factory=dict_row)

# ====== Request bodies ======
class IngestText(BaseModel):
    title: str
    text: str
    source_type: str  # 'internal' | 'inspo'
    brand_id: Optional[str] = None
    url: Optional[str] = None
    # NEW
    style_tags: Optional[List[str]] = []
    hook_rate: Optional[float] = None
    cpm: Optional[float] = None
    hold_rate: Optional[float] = None
    cpl: Optional[float] = None
    link_clicks: Optional[int] = None

class GenerateReq(BaseModel):
    brand_id: Optional[str] = None
    industry: str
    service: str
    audience: str
    platforms: List[str] = []
    levers: List[str] = []
    n: int = 12

class RewriteReq(BaseModel):
    concept_text: str
    duration_s: int = 30
    platform: str = "TikTok"
    style: str = "faceless"

# ====== Helpers ======
def chunk_text(txt: str, max_chars: int = 2500, overlap: int = 250):
    txt = txt.strip()
    if len(txt) <= max_chars:
        return [txt]
    chunks, start = [], 0
    while start < len(txt):
        end = min(len(txt), start + max_chars)
        chunks.append(txt[start:end])
        start = max(0, end - overlap)
    return chunks

def embed(texts: List[str]):
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]

PITCH_SYSTEM_PROMPT = "You are Happy Face Ads’ writers-room engine. Bold, absurd, against-the-grain, comedic."
CONCEPT_FORMAT = "Produce {n} numbered concepts. Each with Hook, Premise, 3 Escalations, CTA."

# ====== Routes ======
@app.get("/health")
def health():
    return {"ok": True}

@app.post("/ingest/text")
def ingest_text(req: IngestText):
    if req.source_type not in ("internal", "inspo"):
        raise HTTPException(400, "source_type must be 'internal' or 'inspo'")

    # Build metadata (drop any None metrics)
    metrics = {
        k: v for k, v in {
            "hook_rate": req.hook_rate,
            "cpm": req.cpm,
            "hold_rate": req.hold_rate,
            "cpl": req.cpl,
            "link_clicks": req.link_clicks,
        }.items() if v is not None
    }
    meta = {"style_tags": req.style_tags or [], "metrics": metrics}

    chunks = chunk_text(req.text)
    vecs = embed(chunks)

    with get_conn() as conn, conn.transaction():
        doc = conn.execute(
            "insert into documents(brand_id, source_type, title, url, text_content, metadata) "
            "values (%s,%s,%s,%s,%s,%s) returning id",
            (req.brand_id, req.source_type, req.title, req.url, req.text, json.dumps(meta)),
        ).fetchone()

        for i, (chunk, v) in enumerate(zip(chunks, vecs)):
            conn.execute(
                "insert into embeddings(doc_id, chunk_idx, text_chunk, embedding, metadata) "
                "values (%s,%s,%s,%s,%s)",
                (doc["id"], i, chunk, v, json.dumps({"style_tags": req.style_tags or []})),
            )
    return {"status": "ok", "chunks": len(chunks), "metadata_saved": meta}

@app.post("/generate/concepts")
def generate_concepts(req: GenerateReq):
    q = f"Industry: {req.industry}\nService: {req.service}\nAudience: {req.audience}\nLevers: {', '.join(req.levers)}"
    q_vec = embed([q])[0]

    with get_conn() as conn:
        internal = conn.execute(
            """
            select text_chunk from embeddings e
            join documents d on d.id = e.doc_id
            where d.source_type = 'internal'
            order by e.embedding <=> %s
            limit 8
            """,
            (q_vec,),
        ).fetchall()
        inspo = conn.execute(
            """
            select text_chunk from embeddings e
            join documents d on d.id = e.doc_id
            where d.source_type = 'inspo'
            order by e.embedding <=> %s
            limit 4
            """,
            (q_vec,),
        ).fetchall()

    ctx = "\n\n".join([r["text_chunk"] for r in (internal + inspo)])
    user_prompt = (
        f"CONTEXT:\n{ctx}\nINPUTS:\nIndustry: {req.industry}\nService: {req.service}\n"
        f"Audience: {req.audience}\nLevers: {req.levers}\n"
        + CONCEPT_FORMAT.format(n=req.n)
    )

    chat = client.chat.completions.create(
        model=GEN_MODEL,
        messages=[
            {"role": "system", "content": PITCH_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.9,
    )
    text = chat.choices[0].message.content

    with get_conn() as conn, conn.transaction():
        brief = conn.execute(
            "insert into briefs(inputs, output_md) values (%s,%s) returning id",
            (json.dumps(req.dict()), text),
        ).fetchone()

    return {"brief_id": brief["id"], "markdown": text}

@app.post("/generate/rewrite")
def generate_rewrite(req: RewriteReq):
    prompt = (
        f"Take the concept below and expand it into a {req.duration_s}-second ad script for {req.platform}. "
        f"Style: {req.style}. Concept:\n{req.concept_text}"
    )
    chat = client.chat.completions.create(
        model=GEN_MODEL,
        messages=[
            {"role": "system", "content": "You are Happy Face Ads’ rewrite engine."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.8,
    )
    return {"script": chat.choices[0].message.content}
