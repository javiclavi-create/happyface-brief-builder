import os, json
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import psycopg
from psycopg.rows import dict_row
from pgvector.psycopg import register_vector
from openai import OpenAI

# -------- Env --------
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
DATABASE_URL    = os.environ.get("DATABASE_URL")
FRONTEND_ORIGIN = os.environ.get("FRONTEND_ORIGIN", "*")
EMBED_MODEL     = os.environ.get("EMBED_MODEL", "text-embedding-3-large")
GEN_MODEL       = os.environ.get("GEN_MODEL", "gpt-4o-mini")

if not (OPENAI_API_KEY and DATABASE_URL):
    raise RuntimeError("Missing OPENAI_API_KEY or DATABASE_URL")

client = OpenAI(api_key=OPENAI_API_KEY)

# -------- App --------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_ORIGIN, "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------- DB helpers --------
def get_conn():
    conn = psycopg.connect(DATABASE_URL, row_factory=dict_row)
    # Make Python lists <-> pgvector work for this connection
    register_vector(conn)
    return conn

def embed(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]

# -------- Models --------
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

# -------- Small utils --------
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
        start = max(0, end - overlap)
    return chunks

# -------- Routes --------
@app.get("/health")
def health():
    return {"ok": True}

@app.post("/ingest/text")
def ingest_text(req: IngestText):
    if req.source_type not in ("internal", "inspo"):
        raise HTTPException(400, "source_type must be 'internal' or 'inspo'")

    chunks = chunk_text(req.text)
    vecs   = embed(chunks)

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
                (doc["id"], i, chunk, v),   # thanks to register_vector(conn)
            )

    return {"status": "ok", "chunks": len(chunks)}

# ---- Generation core (strict JSON, exactly n) ----
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

    # Ask for JSON only
    chat = client.chat.completions.create(
        model=GEN_MODEL,
        temperature=0.7,                 # tighter adherence
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": PITCH_SYSTEM},
            {"role": "user",   "content": user},
        ],
    )

    raw = chat.choices[0].message.content or "{}"
    try:
        parsed = json.loads(raw)
        concepts = parsed.get("concepts", [])
        # ensure correct shape
        clean: List[Dict[str, Any]] = []
        for c in concepts:
            hook = (c.get("hook") or "").strip()
            prem = (c.get("premise") or "").strip()
            esc  = c.get("escalations") or []
            cta  = (c.get("cta") or "").strip()
            if hook and prem and isinstance(esc, list) and len(esc) >= 3 and cta:
                clean.append({"hook":hook, "premise":prem, "escalations":esc[:3], "cta":cta})
        return clean
    except Exception:
        return []

@app.post("/generate/concepts")
def generate_concepts(req: GenerateReq):
    # Build a retrieval query from inputs
    fields = [
        req.project_type, req.ad_type, req.industry, req.audience,
        " ".join(req.platforms or []),
        " ".join(req.content_style or []),
        " ".join(req.levers or []),
        (req.product_name or ""), (req.what_it_does or ""), (req.service or "")
    ]
    q = " | ".join([f for f in fields if f])
    q_vec = embed([q])[0]

    # Pull nearest training chunks (internal priority)
    with get_conn() as conn:
        internal = conn.execute(
            """
            select text_chunk from embeddings e
            join documents d on d.id = e.doc_id
            where d.source_type = 'internal'
            order by e.embedding <=> %s
            limit 10
            """,
            (q_vec,),
        ).fetchall()

        inspo = conn.execute(
            """
            select text_chunk from embeddings e
            join documents d on d.id = e.doc_id
            where d.source_type = 'inspo'
            order by e.embedding <=> %s
            limit 6
            """,
            (q_vec,),
        ).fetchall()

    ctx = "\n\n".join([r["text_chunk"] for r in (internal + inspo)]) or "No prior docs yet."

    # First ask for exactly n concepts
    concepts: List[Dict[str, Any]] = _call_for_concepts(ctx, req, req.n)

    # If the model returns fewer, top up with a second call for the missing amount
    if len(concepts) < req.n:
        missing = req.n - len(concepts)
        more = _call_for_concepts(ctx, req, missing)
        concepts.extend(more)

    # Still short? As a last resort, duplicate with small tags so the UI isn’t empty
    while len(concepts) < req.n and concepts:
        concepts.append(concepts[len(concepts) % len(concepts)])

    # Persist brief
    with get_conn() as conn, conn.transaction():
        brief = conn.execute(
            "insert into briefs(inputs, output_md) values (%s,%s) returning id",
            (json.dumps(req.dict()), json.dumps({"concepts": concepts})),
        ).fetchone()

    return {"brief_id": brief["id"], "concepts": concepts}

@app.post("/generate/rewrite")
def generate_rewrite(req: RewriteReq):
    # JSON card back for clean rendering
    prompt = {
        "concept": req.concept_text,
        "instructions": f"Rewrite into a {req.duration_s}s {req.platform} script with sections: Hook, Premise, Escalations (3 beats), CTA. Style: {req.style}. JSON only."
    }
    chat = client.chat.completions.create(
        model=GEN_MODEL,
        temperature=0.7,
        response_format={"type":"json_object"},
        messages=[
            {"role":"system","content":"You are Happy Face Ads’ rewrite engine. Keep it punchy and producible."},
            {"role":"user","content": json.dumps(prompt)}
        ]
    )
    try:
        data = json.loads(chat.choices[0].message.content or "{}")
    except Exception:
        data = {"hook":"", "premise":"", "escalations": [], "cta":""}
    return {"script": data}
