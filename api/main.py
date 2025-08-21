import os, json, subprocess, tempfile
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import psycopg
from psycopg.rows import dict_row
from pgvector.psycopg import register_vector
from openai import OpenAI

# ===== ENV =====
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
DATABASE_URL = os.environ.get("DATABASE_URL")
FRONTEND_ORIGIN = os.environ.get("FRONTEND_ORIGIN", "*")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "text-embedding-3-large")
GEN_MODEL = os.environ.get("GEN_MODEL", "gpt-4o-mini")

if not (OPENAI_API_KEY and DATABASE_URL):
    raise RuntimeError("Missing OPENAI_API_KEY or DATABASE_URL env var")

client = OpenAI(api_key=OPENAI_API_KEY)

# ===== APP =====
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_ORIGIN, "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== DB =====
def get_conn():
    conn = psycopg.connect(DATABASE_URL, row_factory=dict_row)
    register_vector(conn)  # so lists can be sent as pgvector/arrays
    return conn

# ===== MODELS =====
class IngestText(BaseModel):
    title: str
    text: str
    source_type: str  # 'internal' | 'inspo'
    brand_id: Optional[str] = None
    url: Optional[str] = None
    style_tags: List[str] = []

class IngestImageURL(BaseModel):
    title: str
    url: str
    source_type: str  # 'internal' | 'inspo'
    brand_id: Optional[str] = None
    style_hint: List[str] = []

class GenerateReq(BaseModel):
    brand_id: Optional[str] = None
    industry: str
    service: str
    audience: str
    platforms: List[str] = []
    levers: List[str] = []
    prefer_styles: List[str] = []   # NEW: bias toward these styles
    n: int = 6

class Concept(BaseModel):
    title: Optional[str] = None
    hook: str
    premise: str
    escalations: List[str]
    cta: str

class RewriteReq(BaseModel):
    concept: Concept
    duration_s: int = 30
    platform: str = "TikTok"
    style: str = "faceless"

class RewriteBatchReq(BaseModel):
    items: List[RewriteReq]

class MetricsIngest(BaseModel):
    doc_id: str  # link to a row in documents
    hook_rate: Optional[float] = None
    hold_rate: Optional[float] = None
    clicks: Optional[int] = None
    conversions: Optional[int] = None
    frequency: Optional[float] = None
    cpm: Optional[float] = None
    cost_per_lead: Optional[float] = None
    meta: Dict[str, Any] = {}

# ===== HELPERS =====
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

def vision_describe_image(url: str, extra_hints: List[str]) -> Dict[str, Any]:
    """Use OpenAI vision to tag an image and summarize style."""
    system = "You are a creative director. Return compact JSON: {tags: [..], description: '...'}"
    user = (
        "Look at this ad image. List high-signal style tags "
        "(tone, pacing, humor device, color/lighting, framing, editing notes). "
        "Prefer 3-8 tags. Then 1-2 sentence description."
        + (f" Style hints: {', '.join(extra_hints)}." if extra_hints else "")
    )
    chat = client.chat.completions.create(
        model=GEN_MODEL,
        response_format={"type": "json_object"},
        messages=[{"role":"system","content":system},
                  {"role":"user","content":[
                      {"type":"input_text","text":user},
                      {"type":"input_image","image_url": url}
                  ]}],
        temperature=0.2,
    )
    try:
        return json.loads(chat.choices[0].message.content)
    except Exception:
        return {"tags": extra_hints, "description": "visual analysis unavailable"}

# ===== PROMPTS =====
PITCH_SYSTEM_PROMPT = "Return only valid JSON. You are Happy Face Ads’ writers-room engine—bold, absurd, against-the-grain, big swings, often faceless."
SCHEMA_HINT = {
    "concepts":[
        {"title":"string","hook":"string","premise":"string","escalations":["string","string","string"],"cta":"string"}
    ]
}

# ===== ROUTES =====
@app.get("/health")
def health():
    return {"ok": True}

# ---- Ingest plain text (scripts, notes, etc.)
@app.post("/ingest/text")
def ingest_text(req: IngestText):
    if req.source_type not in ("internal", "inspo"):
        raise HTTPException(400, "source_type must be 'internal' or 'inspo'")
    chunks = chunk_text(req.text)
    vecs = embed(chunks)
    with get_conn() as conn, conn.transaction():
        doc = conn.execute(
            """
            insert into documents(brand_id, source_type, title, url, text_content, style_tags)
            values (%s,%s,%s,%s,%s,%s)
            returning id
            """,
            (req.brand_id, req.source_type, req.title, req.url, req.text, req.style_tags),
        ).fetchone()
        for i, (chunk, v) in enumerate(zip(chunks, vecs)):
            conn.execute(
                "insert into embeddings(doc_id, chunk_idx, text_chunk, embedding) values (%s,%s,%s,%s)",
                (doc["id"], i, chunk, v),
            )
    return {"status": "ok", "chunks": len(chunks)}

# ---- Ingest an IMAGE by URL (model “sees” it)
@app.post("/ingest/image-url")
def ingest_image_url(req: IngestImageURL):
    if req.source_type not in ("internal", "inspo"):
        raise HTTPException(400, "source_type must be 'internal' or 'inspo'")
    analysis = vision_describe_image(req.url, req.style_hint)
    tags = list(dict.fromkeys((analysis.get("tags") or []) + (req.style_hint or [])))  # unique
    text = f"IMAGE ANALYSIS\nTags: {', '.join(tags)}\nDescription: {analysis.get('description','')}\nURL: {req.url}"
    chunks = chunk_text(text)
    vecs = embed(chunks)
    with get_conn() as conn, conn.transaction():
        doc = conn.execute(
            """
            insert into documents(brand_id, source_type, title, url, text_content, style_tags, media_path, metadata)
            values (%s,%s,%s,%s,%s,%s,%s,%s)
            returning id
            """,
            (req.brand_id, req.source_type, req.title, req.url, text, tags, req.url, json.dumps({"image_url": req.url, "analysis": analysis})),
        ).fetchone()
        for i, (chunk, v) in enumerate(zip(chunks, vecs)):
            conn.execute(
                "insert into embeddings(doc_id, chunk_idx, text_chunk, embedding) values (%s,%s,%s,%s)",
                (doc["id"], i, chunk, v),
            )
    return {"status":"ok","doc_style_tags":tags,"chunks":len(chunks)}

# ---- Generate concepts (biased by prefer_styles if provided)
@app.post("/generate/concepts")
def generate_concepts(req: GenerateReq):
    q = (
        f"Industry: {req.industry}\n"
        f"Service: {req.service}\n"
        f"Audience: {req.audience}\n"
        f"Levers: {', '.join(req.levers)}\n"
        f"Prefer styles: {', '.join(req.prefer_styles)}"
    )
    q_vec = embed([q])[0]

    with get_conn() as conn:
        if req.prefer_styles:
            internal = conn.execute(
                """
                select text_chunk
                from embeddings e
                join documents d on d.id = e.doc_id
                where d.source_type = 'internal'
                order by
                  case when d.style_tags && %s::text[] then 0 else 1 end,
                  e.embedding <=> %s::vector
                limit 8
                """,
                (req.prefer_styles, q_vec),
            ).fetchall()
            inspo = conn.execute(
                """
                select text_chunk
                from embeddings e
                join documents d on d.id = e.doc_id
                where d.source_type = 'inspo'
                order by
                  case when d.style_tags && %s::text[] then 0 else 1 end,
                  e.embedding <=> %s::vector
                limit 4
                """,
                (req.prefer_styles, q_vec),
            ).fetchall()
        else:
            internal = conn.execute(
                """
                select text_chunk
                from embeddings e
                join documents d on d.id = e.doc_id
                where d.source_type = 'internal'
                order by e.embedding <=> %s::vector
                limit 8
                """,
                (q_vec,),
            ).fetchall()
            inspo = conn.execute(
                """
                select text_chunk
                from embeddings e
                join documents d on d.id = e.doc_id
                where d.source_type = 'inspo'
                order by e.embedding <=> %s::vector
                limit 4
                """,
                (q_vec,),
            ).fetchall()

    ctx = "\n\n".join([r["text_chunk"] for r in (internal + inspo)])

    user_prompt = (
        "Style: bold, absurd, against-the-grain. Return EXACTLY this JSON shape with "
        f"{req.n} items:\n" + json.dumps(SCHEMA_HINT, indent=2) +
        "\n\nCONTEXT:\n" + ctx +
        f"\n\nINPUTS:\nIndustry: {req.industry}\nService: {req.service}\nAudience: {req.audience}\n"
        f"Levers: {', '.join(req.levers)}\nPrefer styles: {', '.join(req.prefer_styles)}"
    )

    chat = client.chat.completions.create(
        model=GEN_MODEL,
        messages=[{"role":"system","content":PITCH_SYSTEM_PROMPT},
                  {"role":"user","content":user_prompt}],
        temperature=0.9,
        response_format={"type":"json_object"},
    )
    raw = chat.choices[0].message.content
    try:
        data = json.loads(raw)
        concepts = data.get("concepts", [])
    except Exception:
        concepts = []

    # Also store a markdown copy
    def to_md(c, i):
        title = c.get("title") or f"Concept {i+1}"
        esc = "\n".join([f"- {e}" for e in c.get("escalations", [])])
        return f"### {title}\n1. **Hook**: {c.get('hook','')}\n2. **Premise**: {c.get('premise','')}\n3. **Escalations**:\n{esc}\n4. **CTA**: {c.get('cta','')}\n"
    markdown = "\n\n".join([to_md(c,i) for i,c in enumerate(concepts)])

    with get_conn() as conn, conn.transaction():
        brief = conn.execute(
            "insert into briefs(inputs, output_md) values (%s,%s) returning id",
            (json.dumps(req.dict()), markdown),
        ).fetchone()

    return {"brief_id": brief["id"], "concepts": concepts, "markdown": markdown}

# ---- Rewrite one concept
@app.post("/generate/rewrite")
def generate_rewrite(req: RewriteReq):
    c = req.concept
    concept_text = (
        f"Title: {c.title or ''}\nHook: {c.hook}\nPremise: {c.premise}\n"
        f"Escalations:\n- " + "\n- ".join(c.escalations) + f"\nCTA: {c.cta}"
    )
    prompt = (
        f"Expand into a {req.duration_s}-second ad script for {req.platform}. "
        f"Style: {req.style}. Write beat-by-beat shot list and VO/captions. "
        f"Keep it producible and punchy.\n\n{concept_text}"
    )
    chat = client.chat.completions.create(
        model=GEN_MODEL,
        messages=[{"role":"system","content":"You are Happy Face Ads’ rewrite engine."},
                  {"role":"user","content":prompt}],
        temperature=0.8,
    )
    return {"script": chat.choices[0].message.content}

# ---- Rewrite multiple concepts at once
@app.post("/generate/rewrite-batch")
def generate_rewrite_batch(req: RewriteBatchReq):
    outputs = []
    for item in req.items:
        out = generate_rewrite(item)  # reuse single route logic
        outputs.append(out["script"])
    return {"scripts": outputs}

# ---- Ingest performance metrics
@app.post("/ingest/metrics")
def ingest_metrics(req: MetricsIngest):
    with get_conn() as conn, conn.transaction():
        conn.execute(
            """
            insert into ad_metrics(doc_id, hook_rate, hold_rate, clicks, conversions, frequency, cpm, cost_per_lead, meta)
            values (%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """,
            (req.doc_id, req.hook_rate, req.hold_rate, req.clicks, req.conversions,
             req.frequency, req.cpm, req.cost_per_lead, json.dumps(req.meta)),
        )
    return {"status": "ok"}

# ---- Style counts (for dashboard)
@app.get("/stats/styles")
def stats_styles():
    with get_conn() as conn:
        rows = conn.execute("select * from style_counts").fetchall()
    return {"styles": rows}
