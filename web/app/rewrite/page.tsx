'use client';
import { useEffect, useState } from 'react';
const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8000';

type Concept = { title?: string; hook: string; premise: string; escalations: string[]; cta: string; };

type RewriteItem = {
  concept: Concept;
  duration_s: number;
  platform: string;
  style: string;
  script?: string;
  generating?: boolean;
}

export default function RewritePage(){
  const [items, setItems] = useState<RewriteItem[]>([]);

  useEffect(()=>{
    const raw = localStorage.getItem('rewrite_queue');
    if(raw){
      const concepts: Concept[] = JSON.parse(raw);
      setItems(concepts.map(c=>({
        concept: c,
        duration_s: 30,
        platform: 'TikTok',
        style: 'faceless',
      })));
    }
  },[]);

  async function genOne(idx: number){
    setItems(arr => arr.map((it,i)=> i===idx ? {...it, generating:true } : it));
    const it = items[idx];
    const res = await fetch(`${API_BASE}/generate/rewrite`,{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ concept: it.concept, duration_s: it.duration_s, platform: it.platform, style: it.style })
    });
    const json = await res.json();
    setItems(arr => arr.map((it,i)=> i===idx ? {...it, script: json.script, generating:false } : it));
  }

  async function genAll(){
    const payload = items.map(it=>({ concept: it.concept, duration_s: it.duration_s, platform: it.platform, style: it.style }));
    const res = await fetch(`${API_BASE}/generate/rewrite-batch`,{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ items: payload })
    });
    const json = await res.json();
    setItems(arr => arr.map((it,i)=> ({...it, script: json.scripts[i]})));
  }

  return (
    <main className="space-y-6">
      <section className="card-pro">
        <div className="flex items-center justify-between">
          <h2 className="text-xl font-medium">Rewrite Studio</h2>
          <div className="flex gap-2">
            <button className="btn" onClick={()=>genAll()} disabled={items.length===0}>Generate all</button>
          </div>
        </div>
      </section>

      {items.length===0 && (
        <div className="text-sm text-neutral-400">No concepts yet. Go back and pick “Rewrite this” or “Rewrite selected”.</div>
      )}

      <div className="grid md:grid-cols-2 gap-4">
        {items.map((it, idx)=>(
          <section key={idx} className="card-pro space-y-3">
            <header className="flex items-center justify-between">
              <h3 className="font-semibold">{it.concept.title || `Concept ${idx+1}`}</h3>
              <button className="btn" onClick={()=>genOne(idx)} disabled={it.generating}>
                {it.generating ? 'Generating…' : 'Generate'}
              </button>
            </header>

            <div className="text-sm space-y-2">
              <div><span className="label">Hook</span><p className="mt-1">{it.concept.hook}</p></div>
              <div><span className="label">Premise</span><p className="mt-1">{it.concept.premise}</p></div>
              <div><span className="label">Escalations</span>
                <ul className="mt-1 list-disc pl-5">{it.concept.escalations.map((e,i)=><li key={i}>{e}</li>)}</ul>
              </div>
              <div><span className="label">CTA</span><p className="mt-1">{it.concept.cta}</p></div>
            </div>

            <div className="grid grid-cols-3 gap-2">
              <div>
                <label className="label">Duration (s)</label>
                <input className="input" type="number" value={it.duration_s}
                       onChange={e=>setItems(arr=>arr.map((x,i)=>i===idx?{...x,duration_s:Number(e.target.value)}:x))}/>
              </div>
              <div>
                <label className="label">Platform</label>
                <input className="input" value={it.platform}
                       onChange={e=>setItems(arr=>arr.map((x,i)=>i===idx?{...x,platform:e.target.value}:x))}/>
              </div>
              <div>
                <label className="label">Style</label>
                <input className="input" value={it.style}
                       onChange={e=>setItems(arr=>arr.map((x,i)=>i===idx?{...x,style:e.target.value}:x))}/>
              </div>
            </div>

            {it.script && (
              <div>
                <div className="label">Script</div>
                <pre className="whitespace-pre-wrap text-sm mt-1">{it.script}</pre>
              </div>
            )}
          </section>
        ))}
      </div>
    </main>
  );
}
