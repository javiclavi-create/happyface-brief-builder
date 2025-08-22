import os, io, json, uuid
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import psycopg
from psycopg.rows import dict_row
from pgvector.psycopg import register_vector, Vector   # <-- IMPORTANT
from openai import OpenAI

# --- Env ---
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
DATABASE_URL = os.environ.get("DATABASE_URL")
FRONTEND_ORIGIN = os.environ.get("FRONTEND_ORIGIN", "*")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "text-embedding-3-large")
GEN_MODEL = os.environ.get("GEN_MODEL", "gpt-4o-mini")  # vision-capable
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
BUCKET = "ads"

if not (OPENAI_API_KEY and DATABASE_URL):
    raise RuntimeError("Missing OPENAI_API_KEY or DATABASE_URL env var")

# --- Clients ---
client = OpenAI(api_key=OPENAI_API_KEY)

supabase = None
if SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY:
    from supabase import create_client
    supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

# --- App ---
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_ORIGIN, "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- DB helper ---
def get_conn():
    conn = psycopg.connect(DATABASE_URL, row_factory=dict_row)
    register_vector(conn)  # register pgvector adapter for this connection
    return conn

# --- Small utils ---
def chunk_text(txt: str, max_chars: int = 2500, overlap: int = 250):
    txt = (txt or "").strip()
    if not txt:
        return []
    if len(txt) <= max_chars:
        return [txt]
    chunks, start = [], 0
    while start < len(txt):
        end = min(len(txt), start + max_chars)
        chunks.append(txt[start:end])
        start = end - overlap
        if start < 0: start = 0
    return chunks

def embed(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]

def dict_clean(d: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in d.items() if v not in (None, "", [], {})}

# --- Models for generation ---
class GenerateReq(BaseModel):
    project_type: str  # "product" or "service"
    ad_type: str       # "static" or "video"
    industry: str
    product_name: Optional[str] = None
    what_it_does: Optional[str] = None
    service: Optional[str] = None
    audience: str
    platforms: List[str] = []
    content_style: List[str] = []  # UGC, Talking Head, Faceless Video, Skit, Parody
    levers: List[str] = []         # funny levers
    n: int = 12
    brand_id: Optional[str] = None

class RewriteReq(BaseModel):
    concept_text: str
    duration_s: int = 30
    platform: str = "TikTok"
    style: str = "Faceless Video"

class RewriteBatchReq(BaseModel):
    concepts: List[Dict[str, Any]]
    duration_s: int = 30
    platform: str = "TikTok"
    style: str = "Faceless Video"

# --- Vision helper for single image URL ---
def describe_image(public_url: str) -> str:
    try:
        msg = client.chat.completions.create(
            model=GEN_MODEL,
            temperature=0.2,
            messages=[
                {"role": "system", "content": "You are a sharp creative director describing ad visuals."},
                {"role": "user", "content": [
                    {"type": "text", "text": "Describe the visual style, tone, on-screen text, props, layout, and color palette."},
                    {"type": "image_url", "image_url": {"url": public_url}}
                ]}
            ]
        )
        return (msg.choices[0].message.content or "").strip()
    except Exception:
        return ""

# --- Routes ---
@app.get("/health")
def health():
    return {"ok": True}

# 1) Ingest plain text (no file)
@app.post("/ingest/text")
def ingest_text(
    title: str = Form(...),
    text: str = Form(...),
    source_type: str = Form(...),  # 'internal' | 'inspo'
    media_kind: Optional[str] = Form(None),  # 'static' | 'video'
    brand_id: Optional[str] = Form(None),
    url: Optional[str] = Form(None),
    content_style: Optional[str] = Form(None),
    hook_rate: Optional[float] = Form(None),
    hold_rate: Optional[float] = Form(None),
    cpm: Optional[float] = Form(None),
    cpl: Optional[float] = Form(None),
    link_clicks: Optional[int] = Form(None),
    spend: Optional[float] = Form(None),
):
    if source_type not in ("internal", "inspo"):
        raise HTTPException(400, "source_type must be 'internal' or 'inspo'")

    metrics = dict_clean({
        "hook_rate": hook_rate, "hold_rate": hold_rate, "cpm": cpm,
        "cpl": cpl, "link_clicks": link_clicks, "spend": spend
    })

    full_text = f"TITLE: {title}\nURL: {url or ''}\nMEDIA_KIND: {media_kind or ''}\nCONTENT_STYLE: {content_style or ''}\nMETRICS: {json.dumps(metrics)}\n\n{text}".strip()
    chunks = chunk_text(full_text)
    vecs = embed(chunks)

    with get_conn() as conn, conn.transaction():
        doc = conn.execute(
            """insert into documents(brand_id, source_type, title, url, text_content, media_kind, metadata)
               values (%s,%s,%s,%s,%s,%s,%s) returning id""",
            (brand_id, source_type, title, url, full_text, media_kind, json.dumps(dict_clean({"content_style":content_style, "metrics":metrics})))
        ).fetchone()
        for i, (chunk, v) in enumerate(zip(chunks, vecs)):
            conn.execute(
                "insert into embeddings(doc_id, chunk_idx, text_chunk, embedding) values (%s,%s,%s,%s)",
                (doc["id"], i, chunk, Vector(v))   # <-- IMPORTANT
            )
    return {"status": "ok", "chunks": len(chunks)}

# 2) Ingest with optional file (image or video) + optional link + metrics
@app.post("/ingest/media")
async def ingest_media(
    title: str = Form(...),
    source_type: str = Form(...),    # 'internal' | 'inspo'
    media_kind: str = Form(...),     # 'static' | 'video'
    brand_id: Optional[str] = Form(None),
    url: Optional[str] = Form(None),
    content_style: Optional[str] = Form(None),
    hook_rate: Optional[float] = Form(None),
    hold_rate: Optional[float] = Form(None),
    cpm: Optional[float] = Form(None),
    cpl: Optional[float] = Form(None),
    link_clicks: Optional[int] = Form(None),
    spend: Optional[float] = Form(None),
    file: UploadFile = File(None),
):
    if source_type not in ("internal", "inspo"):
        raise HTTPException(400, "source_type must be 'internal' or 'inspo'")
    if media_kind not in ("static", "video"):
        raise HTTPException(400, "media_kind must be 'static' or 'video'")

    public_url = None
    media_mime = None

    if file and supabase:
        media_mime = file.content_type or ""
        path = f"uploads/{uuid.uuid4()}/{file.filename}"
        data = await file.read()
        upload = supabase.storage.from_(BUCKET).upload(path, data, {"content-type": media_mime, "upsert": False})
        if upload is not None and getattr(upload, "error", None):
            raise HTTPException(500, f"Storage error: {upload.error}")
        public_url = supabase.storage.from_(BUCKET).get_public_url(path)
    else:
        public_url = url

    visual_desc = ""
    if media_kind == "static" and public_url:
        visual_desc = describe_image(public_url)

    metrics = dict_clean({
        "hook_rate": hook_rate, "hold_rate": hold_rate, "cpm": cpm,
        "cpl": cpl, "link_clicks": link_clicks, "spend": spend
    })

    base_lines = [
        f"TITLE: {title}",
        f"MEDIA_KIND: {media_kind}",
        f"CONTENT_STYLE: {content_style or ''}",
        f"PUBLIC_URL: {public_url or ''}",
        f"VISUAL_DESC: {visual_desc}" if visual_desc else "",
        f"METRICS: {json.dumps(metrics)}",
    ]
    full_text = "\n".join([x for x in base_lines if x]).strip()

    chunks = chunk_text(full_text)
    vecs = embed(chunks)

    with get_conn() as conn, conn.transaction():
        doc = conn.execute(
            """insert into documents(brand_id, source_type, title, url, text_content, media_kind, media_mime, media_url, metadata)
               values (%s,%s,%s,%s,%s,%s,%s,%s,%s) returning id""",
            (brand_id, source_type, title, public_url, full_text, media_kind, media_mime, public_url,
             json.dumps(dict_clean({"content_style": content_style, "metrics": metrics})))
        ).fetchone()
        for i, (chunk, v) in enumerate(zip(chunks, vecs)):
            conn.execute(
                "insert into embeddings(doc_id, chunk_idx, text_chunk, embedding) values (%s,%s,%s,%s)",
                (doc["id"], i, chunk, Vector(v))   # <-- IMPORTANT
            )

    return {"status": "ok", "chunks": len(chunks), "url": public_url}

# 3) Generate concepts (BIG swings, 10/10 funny)
@app.post("/generate/concepts")
def generate_concepts(req: GenerateReq):
    fields = [
        req.project_type, req.ad_type, req.industry, req.audience,
        " ".join(req.platforms or []),
        " ".join(req.content_style or []),
        " ".join(req.levers or []),
        (req.product_name or ""), (req.what_it_does or ""), (req.service or "")
    ]
    q = " | ".join([f for f in fields if f])

    q_vec = embed([q])[0]

    with get_conn() as conn:
        internal = conn.execute(
            """select text_chunk from embeddings e
               join documents d on d.id = e.doc_id
               where d.source_type = 'internal'
               order by e.embedding <=> %s
               limit 8""",
            (Vector(q_vec),)   # <-- IMPORTANT
        ).fetchall()
        inspo = conn.execute(
            """select text_chunk from embeddings e
               join documents d on d.id = e.doc_id
               where d.source_type = 'inspo'
               order by e.embedding <=> %s
               limit 4""",
            (Vector(q_vec),)   # <-- IMPORTANT
        ).fetchall()

    ctx = "\n\n".join([r["text_chunk"] for r in (internal + inspo)]) or "No prior docs yet."

    system = (
        "You are Happy Face Ads’ writers-room engine. Always swing HUGE (10/10 absurdity), "
        "against the grain, bold, deadpan where funny. Keep ideas producible."
    )

    user = {
        "context": ctx[:12000],
        "inputs": req.dict(),
        "format": "Return JSON with key 'concepts' as an array of objects: {hook, premise, escalations(array of 3), cta}. NO markdown."
    }

    chat = client.chat.completions.create(
        model=GEN_MODEL,
        temperature=0.95,
        messages=[
            {"role":"system","content": system},
            {"role":"user","content": json.dumps(user)}
        ]
    )

    raw = chat.choices[0].message.content or "{}"
    try:
        parsed = json.loads(raw)
        concepts = parsed.get("concepts", [])
    except Exception:
        concepts = [{
            "hook":"A totally unexpected cold open that subverts the product category",
            "premise":"Lean into an absurd premise that still showcases the product/service benefit",
            "escalations":["Beat 1 gets weirder","Beat 2 doubles the bit","Beat 3 flips POV"],
            "cta":"Direct, punchy ask tailored to platform"
        }]

    with get_conn() as conn, conn.transaction():
        brief = conn.execute(
            "insert into briefs(inputs, output_md) values (%s,%s) returning id",
            (json.dumps(req.dict()), raw)
        ).fetchone()

    return {"brief_id": brief["id"], "concepts": concepts, "markdown": raw}

# 4) Rewrite single
@app.post("/generate/rewrite")
def generate_rewrite(req: RewriteReq):
    prompt = {
        "concept": req.concept_text,
        "instructions": f"Rewrite into a {req.duration_s}s {req.platform} script in card format sections: Hook, Premise, Escalations (3 beats), CTA. Style: {req.style}."
    }
    chat = client.chat.completions.create(
        model=GEN_MODEL,
        temperature=0.8,
        messages=[
            {"role":"system","content":"You are Happy Face Ads’ rewrite engine. Keep it tight, punchy, shootable."},
            {"role":"user","content": json.dumps(prompt)}
        ]
    )
    return {"script": chat.choices[0].message.content}

# 5) Rewrite batch
@app.post("/generate/rewrite-batch")
def generate_rewrite_batch(req: RewriteBatchReq):
    scripts = []
    for c in req.concepts:
        concept_text = json.dumps(c)
        prompt = {
            "concept": concept_text,
            "instructions": f"Rewrite into a {req.duration_s}s {req.platform} script in card format sections: Hook, Premise, Escalations (3 beats), CTA. Style: {req.style}."
        }
        chat = client.chat.completions.create(
            model=GEN_MODEL,
            temperature=0.8,
            messages=[
                {"role":"system","content":"You are Happy Face Ads’ rewrite engine. Keep it tight, punchy, shootable."},
                {"role":"user","content": json.dumps(prompt)}
            ]
        )
        scripts.append(chat.choices[0].message.content)
    return {"scripts": scripts}
