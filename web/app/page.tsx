'use client';
import { useMemo, useState } from 'react';
const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8000';

// very light parser that looks for section labels
function parseConcepts(md: string){
  const blocks = md.split('### ').filter(Boolean);
  const out: any[] = [];
  blocks.forEach((b, idx) => {
    const lines = b.split('\n');
    const title = lines[0]?.trim() || `Concept ${idx+1}`;
    const data: any = { id: `c${idx+1}`, title, hook:'', premise:'', escalations:[], cta:'', raw: b.trim() };
    let section = '';
    for(let i=1;i<lines.length;i++){
      const L = lines[i].trim();
      const low = L.toLowerCase();
      if(low.startsWith('**hook') || low.startsWith('hook')){ section='hook'; data.hook = L.replaceAll('*','').split(':').slice(1).join(':').trim(); continue; }
      if(low.startsWith('**premise') || low.startsWith('premise')){ section='premise'; const content = L.replaceAll('*','').split(':').slice(1).join(':').trim(); if(content) data.premise = content; continue; }
      if(low.startsWith('**escalations') || low.startsWith('escalations')){ section='esc'; continue; }
      if(low.startsWith('**cta') || low.startsWith('cta')){ section='cta'; data.cta = L.replaceAll('*','').split(':').slice(1).join(':').trim(); continue; }
      if(section==='esc'){
        let s = L;
        if (s.startsWith('- ')) s = s.slice(2);
        else if (s.startsWith('– ')) s = s.slice(2);
        else if (s.startsWith('* ')) s = s.slice(2);
        if (s) data.escalations.push(s);
      } else if(section==='premise' && L){
        data.premise += (data.premise ? ' ' : '') + L;
      }
    }
    out.push(data);
  });
  return out;
}

export default function Home() {
  const [form, setForm] = useState({
    industry: '', service: '', audience: '', platforms: 'TikTok, Reels',
    levers: ['juxtaposition','absurd escalation','deadpan'], n: 12
  });
  const [loading, setLoading] = useState(false);
  const [markdown, setMarkdown] = useState<string | null>(null);
  const [selected, setSelected] = useState<string[]>([]);

  const concepts = useMemo(() => (markdown ? parseConcepts(markdown) : []), [markdown]);

  const toggleLever = (l: string) => {
    setForm(f => ({...f, levers: f.levers.includes(l) ? f.levers.filter(x=>x!==l) : [...f.levers, l]}));
  };

  async function onSubmit(e:any){
    e.preventDefault();
    setLoading(true); setMarkdown(null); setSelected([]);
    const res = await fetch(`${API_BASE}/generate/concepts`,{
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({
        industry: form.industry,
        service: form.service,
        audience: form.audience,
        platforms: form.platforms.split(',').map(s=>s.trim()),
        levers: form.levers, n: form.n
      })
    });
    const json = await res.json();
    setMarkdown(json.markdown || '');
    setLoading(false);
  }

  function togglePick(id: string){
    setSelected(prev => prev.includes(id) ? prev.filter(x=>x!==id) : [...prev, id]);
  }

  async function sendSelectedToRewrite(){
    const picks = concepts.filter(c=>selected.includes(c.id));
    const results: any[] = [];
    for (const c of picks){
      const res = await fetch(`${API_BASE}/generate/rewrite`,{
        method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({ concept_text: c.raw, duration_s: 30, platform: 'TikTok', style: 'faceless' })
      });
      const json = await res.json();
      results.push({ title: c.title, script: json.script });
    }
    localStorage.setItem('rewrites', JSON.stringify(results));
    window.location.href = '/rewrite';
  }

  return (
    <main className="space-y-6">
      <section className="card space-y-4">
        <h2 className="text-xl font-medium">New Brief</h2>
        <form className="grid grid-cols-1 md:grid-cols-2 gap-4" onSubmit={onSubmit}>
          <div><label className="label">Industry</label><input className="input" value={form.industry} onChange={e=>setForm({...form, industry:e.target.value})}/></div>
          <div><label className="label">Service</label><input className="input" value={form.service} onChange={e=>setForm({...form, service:e.target.value})}/></div>
          <div><label className="label">Audience</label><input className="input" value={form.audience} onChange={e=>setForm({...form, audience:e.target.value})}/></div>
          <div><label className="label">Platforms</label><input className="input" value={form.platforms} onChange={e=>setForm({...form, platforms:e.target.value})}/></div>
          <div className="md:col-span-2">
            <label className="label">Funny levers</label>
            <div className="flex flex-wrap gap-2 mt-2">
              {['juxtaposition','absurd escalation','deadpan','pattern break'].map(l=>{
                const picked = form.levers.includes(l);
                return (
                  <button
                    key={l}
                    type="button"
                    className={`btn ${picked ? 'btn-selected' : ''}`}
                    onClick={()=>toggleLever(l)}
                  >{l}</button>
                );
              })}
            </div>
          </div>
          <div className="md:col-span-2 flex items-center gap-3">
            <label className="label"># Concepts</label>
            <input type="number" className="input w-24" value={form.n} onChange={e=>setForm({...form, n:Number(e.target.value)})}/>
            <button className="btn" type="submit" disabled={loading}>{loading?'Generating…':'Generate'}</button>
          </div>
        </form>
      </section>

      {concepts.length>0 && (
        <section className="space-y-3">
          <div className="flex items-center justify-between">
            <h3 className="text-lg font-medium">Pitch Room Output</h3>
            <button className="btn btn-selected" disabled={!selected.length} onClick={sendSelectedToRewrite}>
              Send {selected.length||''} to Rewrite →
            </button>
          </div>
          <div className="grid md:grid-cols-2 gap-4">
            {concepts.map(c=> (
              <div key={c.id} className={`card space-y-2 ${selected.includes(c.id)?'ring-2 ring-[var(--happy-yellow)]':''}`}>
                <div className="flex items-center justify-between">
                  <h4 className="font-semibold">{c.title}</h4>
                  <label className="badge">
                    <input type="checkbox" className="mr-2" checked={selected.includes(c.id)} onChange={()=>togglePick(c.id)} /> pick
                  </label>
                </div>
                {c.hook && <p><span className="font-semibold">Hook: </span>{c.hook}</p>}
                {c.premise && <p><span className="font-semibold">Premise: </span>{c.premise}</p>}
                {c.escalations?.length>0 && (
                  <div>
                    <div className="font-semibold">Escalations</div>
                    <ul className="list-disc list-inside text-sm space-y-1">
                      {c.escalations.map((e: string, i: number) => (<li key={i}>{e}</li>))}
                    </ul>
                  </div>
                )}
                {c.cta && <p><span className="font-semibold">CTA: </span>{c.cta}</p>}
                <div className="pt-2">
                  <button className="btn" onClick={async()=>{ setSelected([c.id]); await sendSelectedToRewrite(); }}>
                    Send to Rewrite
                  </button>
                </div>
              </div>
            ))}
          </div>
        </section>
      )}
    </main>
  );
}
