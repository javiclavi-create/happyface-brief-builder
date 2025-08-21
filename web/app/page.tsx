'use client';
import { useState } from 'react';
const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8000';

type Concept = {
  title?: string;
  hook: string;
  premise: string;
  escalations: string[];
  cta: string;
};

export default function Home() {
  const [form, setForm] = useState({
    industry: '',
    service: '',
    audience: '',
    platforms: 'TikTok, Reels',
    levers: ['juxtaposition','absurd escalation','deadpan'],
    prefer_styles: ['deadpan'], // NEW
    n: 6
  });
  const [loading, setLoading] = useState(false);
  const [concepts, setConcepts] = useState<Concept[]>([]);
  const [selected, setSelected] = useState<Set<number>>(new Set());
  const [error, setError] = useState<string | null>(null);

  const toggleLever = (l: string) => {
    setForm(f => ({
      ...f,
      levers: f.levers.includes(l) ? f.levers.filter(x=>x!==l) : [...f.levers, l]
    }));
  }
  const toggleStyle = (s: string) => {
    setForm(f => ({
      ...f,
      prefer_styles: f.prefer_styles.includes(s) ? f.prefer_styles.filter(x=>x!==s) : [...f.prefer_styles, s]
    }));
  }
  const toggleSelect = (i:number) => {
    setSelected(prev=>{
      const n = new Set(prev);
      n.has(i) ? n.delete(i) : n.add(i);
      return n;
    });
  }

  async function onSubmit(e:any){
    e.preventDefault();
    setLoading(true);
    setError(null);
    setConcepts([]);
    setSelected(new Set());
    try{
      const res = await fetch(`${API_BASE}/generate/concepts`,{
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body: JSON.stringify({
          industry: form.industry,
          service: form.service,
          audience: form.audience,
          platforms: form.platforms.split(',').map(s=>s.trim()),
          levers: form.levers,
          prefer_styles: form.prefer_styles,
          n: form.n
        })
      });
      const json = await res.json();
      if (json.concepts && Array.isArray(json.concepts)) {
        setConcepts(json.concepts);
      } else {
        setError('Unexpected response. Try again.');
      }
    } catch(err:any){
      setError(err?.message || 'Something went wrong.');
    } finally {
      setLoading(false);
    }
  }

  function sendToRewrite(items: Concept[]) {
    localStorage.setItem('rewrite_queue', JSON.stringify(items));
    window.location.href = '/rewrite';
  }

  const STYLE_OPTIONS = ['deadpan','absurd','mockumentary','surreal','infomercial-parody','slice-of-life'];

  return (
    <main className="space-y-6">
      {/* Form */}
      <section className="card-pro space-y-4">
        <div className="flex items-center justify-between">
          <h2 className="text-xl font-medium">New Brief</h2>
        </div>

        <form className="grid grid-cols-1 md:grid-cols-2 gap-4" onSubmit={onSubmit}>
          <div><label className="label">Industry</label><input className="input" value={form.industry} onChange={e=>setForm({...form, industry:e.target.value})}/></div>
          <div><label className="label">Service</label><input className="input" value={form.service} onChange={e=>setForm({...form, service:e.target.value})}/></div>
          <div><label className="label">Audience</label><input className="input" value={form.audience} onChange={e=>setForm({...form, audience:e.target.value})}/></div>
          <div><label className="label">Platforms</label><input className="input" value={form.platforms} onChange={e=>setForm({...form, platforms:e.target.value})}/></div>

          <div className="md:col-span-2">
            <label className="label">Funny levers</label>
            <div className="flex flex-wrap gap-2 mt-2">
              {['juxtaposition','absurd escalation','deadpan','pattern break','surreal reveal'].map(l=>(
                <button key={l} type="button"
                        className={`btn ${form.levers.includes(l)?'bg-neutral-800':''}`}
                        onClick={()=>toggleLever(l)}>{l}</button>
              ))}
            </div>
          </div>

          <div className="md:col-span-2">
            <label className="label">Preferred styles (bias training)</label>
            <div className="flex flex-wrap gap-2 mt-2">
              {STYLE_OPTIONS.map(s=>(
                <button key={s} type="button"
                        className={`btn ${form.prefer_styles.includes(s)?'bg-neutral-800':''}`}
                        onClick={()=>toggleStyle(s)}>{s}</button>
              ))}
            </div>
          </div>

          <div className="md:col-span-2 flex items-center gap-3">
            <label className="label"># Concepts</label>
            <input type="number" className="input w-24" value={form.n}
                   onChange={e=>setForm({...form, n:Number(e.target.value)})}/>
            <button className="btn" type="submit" disabled={loading}>
              {loading ? 'Generating…' : 'Generate'}
            </button>
          </div>
        </form>

        {error && <div className="text-sm text-red-400">{error}</div>}
      </section>

      {/* Results */}
      {concepts.length > 0 && (
        <section className="space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-lg font-medium">Pitch Room Output</h3>
            <div className="flex gap-2">
              <button className="btn"
                onClick={()=>sendToRewrite(concepts.filter((_,i)=>selected.has(i)))}
                disabled={selected.size===0}
              >Rewrite selected ({selected.size})</button>
              <button className="btn"
                onClick={()=>sendToRewrite(concepts)}
              >Rewrite all</button>
            </div>
          </div>

          <div className="grid md:grid-cols-2 gap-4">
            {concepts.map((c, i) => {
              const isSel = selected.has(i);
              return (
                <article key={i} className={`card-pro space-y-3 ${isSel ? 'ring-2 ring-indigo-400' : ''}`}>
                  <header className="flex items-start justify-between gap-3">
                    <div className="flex items-center gap-3">
                      <input type="checkbox" checked={isSel} onChange={()=>toggleSelect(i)} />
                      <h4 className="text-base font-semibold">
                        {c.title || `Concept ${i+1}`}
                      </h4>
                    </div>
                    <div className="flex gap-2">
                      <button className="btn" onClick={()=>sendToRewrite([c])}>Rewrite this</button>
                      <button className="btn" onClick={()=>{
                        const text = `Title: ${c.title || `Concept ${i+1}`}\nHook: ${c.hook}\nPremise: ${c.premise}\nEscalations:\n- ${c.escalations.join('\n- ')}\nCTA: ${c.cta}`;
                        navigator.clipboard.writeText(text);
                      }}>Copy</button>
                    </div>
                  </header>

                  <div className="space-y-2 text-sm">
                    <div><span className="label">Hook</span><p className="mt-1">{c.hook}</p></div>
                    <div><span className="label">Premise</span><p className="mt-1">{c.premise}</p></div>
                    <div>
                      <span className="label">Escalations</span>
                      <ul className="mt-1 list-disc pl-5 space-y-1">
                        {c.escalations.map((e, j)=>(<li key={j}>{e}</li>))}
                      </ul>
                    </div>
                    <div><span className="label">CTA</span><p className="mt-1">{c.cta}</p></div>
                  </div>
                </article>
              );
            })}
          </div>
        </section>
      )}
    </main>
  );
}
