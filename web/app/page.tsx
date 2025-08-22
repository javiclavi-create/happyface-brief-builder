'use client';
import { useMemo, useState } from 'react';
const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8000';

type Concept = { hook: string; premise: string; escalations: string[]; cta: string };

const PLATFORMS = ["TikTok","Reels","Shorts"];
const CONTENT_STYLES = ["UGC","Talking Head","Faceless Video","Skit","Parody"];
const LEVERS = ["juxtaposition","absurd escalation","deadpan","pattern break"];

export default function Home() {
  const [projectType, setProjectType] = useState<'product'|'service'>('product');
  const [adType, setAdType] = useState<'static'|'video'>('video');

  // form fields
  const [industry, setIndustry] = useState('');
  const [productName, setProductName] = useState('');
  const [whatItDoes, setWhatItDoes] = useState('');
  const [service, setService] = useState('');
  const [audience, setAudience] = useState('');
  const [platforms, setPlatforms] = useState<string[]>(["TikTok","Reels"]);
  const [contentStyle, setContentStyle] = useState<string[]>(["Faceless Video"]);
  const [levers, setLevers] = useState<string[]>([...LEVERS]); // always max intensity
  const [n, setN] = useState(12);

  const [loading, setLoading] = useState(false);
  const [concepts, setConcepts] = useState<Concept[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<number[]>([]);
  const [rewriting, setRewriting] = useState(false);
  const [rewriteResults, setRewriteResults] = useState<string[]>([]);

  const toggle = (arr: string[], set: (v: string[])=>void, item: string) => {
    if (arr.includes(item)) set(arr.filter(x => x !== item));
    else set([...arr, item]);
  };

  const canSubmit = useMemo(() => {
    if (!industry || !audience) return false;
    if (projectType === 'product' && (!productName || !whatItDoes)) return false;
    if (projectType === 'service' && !service) return false;
    return true;
  }, [industry, audience, projectType, productName, whatItDoes, service]);

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!canSubmit) return;
    setLoading(true);
    setError(null);
    setConcepts([]);
    setSelected([]);
    try {
      const body = {
        project_type: projectType,
        ad_type: adType,
        industry,
        product_name: projectType === 'product' ? productName : undefined,
        what_it_does: projectType === 'product' ? whatItDoes : undefined,
        service: projectType === 'service' ? service : undefined,
        audience,
        platforms,
        content_style: contentStyle,
        levers,
        n
      };
      const res = await fetch(`${API_BASE}/generate/concepts`, {
        method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(body)
      });
      if (!res.ok) {
        const t = await res.text();
        throw new Error(t || `Server error ${res.status}`);
      }
      const json = await res.json();
      const list: Concept[] = json.concepts || [];
      setConcepts(list);
    } catch (err:any) {
      setError(err.message || 'Something went wrong');
    } finally {
      setLoading(false);
    }
  }

  async function rewriteOne(idx: number) {
    const c = concepts[idx];
    setRewriting(true);
    try {
      const res = await fetch(`${API_BASE}/generate/rewrite`, {
        method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({
          concept_text: JSON.stringify(c),
          duration_s: 30, platform: (platforms[0] || 'TikTok'),
          style: (contentStyle[0] || 'Faceless Video')
        })
      });
      const json = await res.json();
      setRewriteResults([json.script]);
    } catch (e:any) {
      setError(e.message || 'Rewrite failed');
    } finally {
      setRewriting(false);
    }
  }

  async function rewriteBatch() {
    if (selected.length === 0) return;
    setRewriting(true);
    try {
      const chosen = selected.map(i => concepts[i]);
      const res = await fetch(`${API_BASE}/generate/rewrite-batch`, {
        method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({
          concepts: chosen, duration_s: 30,
          platform: (platforms[0] || 'TikTok'),
          style: (contentStyle[0] || 'Faceless Video')
        })
      });
      const json = await res.json();
      setRewriteResults(json.scripts || []);
    } catch (e:any) {
      setError(e.message || 'Batch rewrite failed');
    } finally {
      setRewriting(false);
    }
  }

  return (
    <main className="space-y-6">
      {/* FORM */}
      <section className="card space-y-4">
        <h2 className="text-xl font-medium">Pitch Room</h2>

        {/* toggles */}
        <div className="flex flex-wrap gap-3">
          <div className="flex items-center gap-2">
            <span className="label">Type:</span>
            <button type="button" className={`chip ${projectType==='product'?'chip-selected':''}`} onClick={()=>setProjectType('product')}>Product</button>
            <button type="button" className={`chip ${projectType==='service'?'chip-selected':''}`} onClick={()=>setProjectType('service')}>Service</button>
          </div>
          <div className="flex items-center gap-2">
            <span className="label">Ad:</span>
            <button type="button" className={`chip ${adType==='static'?'chip-selected':''}`} onClick={()=>setAdType('static')}>Static</button>
            <button type="button" className={`chip ${adType==='video'?'chip-selected':''}`} onClick={()=>setAdType('video')}>Video</button>
          </div>
        </div>

        <form className="grid grid-cols-1 md:grid-cols-2 gap-4" onSubmit={onSubmit}>
          <div><label className="label">Industry</label><input className="input" value={industry} onChange={e=>setIndustry(e.target.value)} /></div>
          <div><label className="label">Audience</label><input className="input" value={audience} onChange={e=>setAudience(e.target.value)} /></div>

          {projectType==='product' && (
            <>
              <div><label className="label">Product name</label><input className="input" value={productName} onChange={e=>setProductName(e.target.value)} /></div>
              <div><label className="label">What it does</label><input className="input" value={whatItDoes} onChange={e=>setWhatItDoes(e.target.value)} /></div>
            </>
          )}
          {projectType==='service' && (
            <div className="md:col-span-2"><label className="label">Service</label><input className="input" value={service} onChange={e=>setService(e.target.value)} /></div>
          )}

          {/* Platforms */}
          <div className="md:col-span-2">
            <label className="label">Platforms</label>
            <div className="flex flex-wrap gap-2 mt-2">
              {PLATFORMS.map(p => (
                <button key={p} type="button"
                  className={`chip ${platforms.includes(p)?'chip-selected':''}`}
                  onClick={()=>toggle(platforms, setPlatforms, p)}>{p}</button>
              ))}
            </div>
          </div>

          {/* Content Style */}
          <div className="md:col-span-2">
            <label className="label">Content Style</label>
            <div className="flex flex-wrap gap-2 mt-2">
              {CONTENT_STYLES.map(s => (
                <button key={s} type="button"
                  className={`chip ${contentStyle.includes(s)?'chip-selected':''}`}
                  onClick={()=>toggle(contentStyle, setContentStyle, s)}>{s}</button>
              ))}
            </div>
          </div>

          {/* Funny Levers (always max) */}
          <div className="md:col-span-2">
            <label className="label">Funny Levers (always 10/10)</label>
            <div className="flex flex-wrap gap-2 mt-2">
              {LEVERS.map(l => (
                <button key={l} type="button"
                  className={`chip ${levers.includes(l)?'chip-selected':''}`}
                  onClick={()=>toggle(levers, setLevers, l)}>{l}</button>
              ))}
            </div>
          </div>

          <div className="md:col-span-2 flex items-center gap-3">
            <label className="label"># Concepts</label>
            <input type="number" className="input w-24" value={n} onChange={e=>setN(Number(e.target.value))}/>
            <button className="btn" type="submit" disabled={loading || !canSubmit}>{loading?'Generating…':'Generate'}</button>
          </div>
        </form>
        {error && <div className="text-red-400 text-sm">{error}</div>}
      </section>

      {/* RESULTS */}
      {concepts.length > 0 && (
        <section className="space-y-3">
          <div className="flex items-center justify-between">
            <h3 className="text-lg font-medium">Concepts</h3>
            <div className="flex items-center gap-2">
              <span className="label">Selected: {selected.length}</span>
              <button className="btn" onClick={rewriteBatch} disabled={rewriting || selected.length===0}>
                {rewriting ? 'Rewriting…' : 'Send Selected to Rewrite'}
              </button>
            </div>
          </div>
          <div className="grid md:grid-cols-2 gap-4">
            {concepts.map((c, i) => (
              <div key={i} className="card space-y-3">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <input type="checkbox" checked={selected.includes(i)} onChange={()=>{
                      setSelected(s => s.includes(i) ? s.filter(x=>x!==i) : [...s, i]);
                    }}/>
                    <div className="text-sm text-neutral-400">Select</div>
                  </div>
                  <button className="btn" onClick={()=>rewriteOne(i)}>Send to Rewrite</button>
                </div>

                <div><div className="font-semibold">Hook</div><div className="text-sm">{c.hook}</div></div>
                <div><div className="font-semibold">Premise</div><div className="text-sm">{c.premise}</div></div>
                <div>
                  <div className="font-semibold">Escalations</div>
                  <ul className="list-disc list-inside text-sm space-y-1">
                    {c.escalations.map((e: string, idx: number) => (<li key={idx}>{e}</li>))}
                  </ul>
                </div>
                <div><div className="font-semibold">CTA</div><div className="text-sm">{c.cta}</div></div>
              </div>
            ))}
          </div>
        </section>
      )}

      {/* REWRITE RESULTS */}
      {rewriteResults.length > 0 && (
        <section className="space-y-3">
          <h3 className="text-lg font-medium">Rewrites</h3>
          <div className="grid md:grid-cols-2 gap-4">
            {rewriteResults.map((r, i) => (
              <div key={i} className="card whitespace-pre-wrap text-sm">{r}</div>
            ))}
          </div>
        </section>
      )}
    </main>
  );
}
