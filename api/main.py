import os, json
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import psycopg
from psycopg.rows import dict_row
from openai import OpenAI


# ----- Settings (we'll set these in Render) -----
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
DATABASE_URL = os.environ.get("DATABASE_URL")
FRONTEND_ORIGIN = os.environ.get("FRONTEND_ORIGIN", "*")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "text-embedding-3-small") # 1536 dims
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


# ----- Database helper -----


def get_conn():
return psycopg.connect(DATABASE_URL, row_factory=dict_row)


# ----- Input shapes from the website -----


class IngestText(BaseModel):
title: str
text: str
source_type: str # 'internal' | 'inspo'
brand_id: Optional[str] = None
url: Optional[str] = None


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
style: str = "faceless" # faceless | UGC | actor
# ----- Helpers -----


def chunk_text(txt: str, max_chars: int = 2500, overlap: int = 250):
txt = txt.strip()
if len(txt) <= max_chars:
return [txt]
chunks, start = [], 0
while start < len(txt):
end = min(len(txt), start + max_chars)
chunks.append(txt[start:end])
start = end - overlap
if start < 0:
start = 0
return chunks




def embed(texts: List[str]):
resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
return [d.embedding for d in resp.data]


PITCH_SYSTEM_PROMPT = (
"You are Happy Face Ads’ writers-room engine. You pitch bold, absurd, against-the-grain, comedic ad concepts that are faceless-friendly but can scale to actors. Lean on juxtaposition, pattern breaks, deadpan, absurd escalation, and clear CTAs. Keep hooks ≤12 words."
)


CONCEPT_FORMAT = (
"Produce {n} numbered concepts. Each concept must have:\n"
"Hook: <≤12 words>\nPremise: <2–3 lines>\nEscalation: 1) … 2) … 3) …\nCTA: …\nTags: [lever1][lever2]...\n"
)


REWRITE_SYSTEM_PROMPT = (
"You turn an approved ad concept into a production-ready short-form ad script in Happy Face tone: disruptive, funny, big swings, faceless-friendly; clear CTAs; no banned claims."
)

# ----- Routes the website will call -----
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
f"CONTEXT (internal first, then inspo):\n{ctx}\n\n"
f"INPUTS:\nIndustry: {req.industry}\nService: {req.service}\nAudience: {req.audience}\nPlatforms: {', '.join(req.platforms)}\nLevers: {', '.join(req.levers)}\n\n"
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
prompt = f"""
Take the concept below and expand it into a {req.duration_s}-second ad script for {req.platform}.


Style: {req.style}


Output in this structure (Markdown):
- VO (voice over) lines
- On-screen text lines
- B-roll / props list
- 3 alternate hooks
- 3 alternate CTAs


Concept:
{req.concept_text}
"""


chat = client.chat.completions.create(
model=GEN_MODEL,
messages=[
{"role": "system", "content": REWRITE_SYSTEM_PROMPT},
{"role": "user", "content": prompt},
],
temperature=0.8,
)
return {"script": chat.choices[0].message.content}
