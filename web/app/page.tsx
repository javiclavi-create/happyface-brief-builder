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
    n: 6
  });
  const [loading, setLoading] = useState(false);
  const [concepts, setConcepts] = useState<Concept[]>([]);
  const [error, setError] = useState<string | null>(null);

  const toggleLever = (l: string) => {
    setForm(f => ({
      ...f,
      levers: f.levers.includes(l) ? f.levers.filter(x=>x!==l) : [...f.levers, l]
    }));
  }

  async function onSubmit(e:any){
    e.preventDefault();
    setLoading(true);
    setError(null);
    setConcepts([]);
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
          n: form.n
        })
      });
      const json = await res.json();
      if (json.concepts && Array.isArray(json.concepts)) {
        setConcepts(json.concepts);
      } else {
        setError('Got an unexpected response. Try again.');
      }
    } catch(err:any){
      setError(err?.message || 'Something went wrong.');
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="space-y-6">
      {/* Form */}
      <section className="card-pro space-y-4">
        <div className="flex items-center justify-between">
          <h2 className="text-xl font-medium">New Brief</h2>
          <div className="hidden md:flex gap-2">
            <span className="badge">absurd</span>
            <span className="badge">big swings</span>
            <span className="badge">faceless-friendly</span>
          </div>
        </div>

        <form className="grid grid-cols-1 md:grid-cols-2 gap-4" onSubmit={onSubmit}>
          <div>
            <label className="label">Industry</label>
            <input className="input" value={form.industry} onChange={e=>setForm({...form, industry:e.target.value})}/>
          </div>
          <div>
            <label className="label">Service</label>
            <input className="input" value={form.service} onChange={e=>setForm({...form, service:e.target.value})}/>
          </div>
          <div>
            <label className="label">Audience</label>
            <input className="input" value={form.audience} onChange={e=>setForm({...form, audience:e.target.value})}/>
          </div>
          <div>
            <label className="label">Platforms</label>
            <input className="input" value={form.platforms} onChange={e=>setForm({...form, platforms:e.target.value})}/>
          </div>

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
          <h3 className="text-lg font-medium">Pitch Room Output</h3>

          <div className="grid md:grid-cols-2 gap-4">
            {concepts.map((c, i) => (
              <article key={i} className="card-pro space-y-3">
                <header className="flex items-start justify-between gap-3">
                  <h4 className="text-base font-semibold">
                    {c.title || `Concept ${i+1}`}
                  </h4>
                  <button
                    className="btn"
                    onClick={()=>{
                      const text = `Title: ${c.title || `Concept ${i+1}`}\nHook: ${c.hook}\nPremise: ${c.premise}\nEscalations:\n- ${c.escalations.join('\n- ')}\nCTA: ${c.cta}`;
                      navigator.clipboard.writeText(text);
                    }}
                  >Copy</button>
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
            ))}
          </div>
        </section>
      )}
    </main>
  );
}
