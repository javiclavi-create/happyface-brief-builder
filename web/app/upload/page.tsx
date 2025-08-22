'use client';
import { useState } from 'react';
const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8000';

const STYLE_OPTIONS = ['deadpan','absurd','mockumentary','surreal','infomercial-parody'];

function num(v:string){
  const cleaned = (v ?? '').toString().replace(/[^0-9.\-]/g,'');
  return cleaned ? Number(cleaned) : undefined;
}

export default function Upload(){
  const [title,setTitle] = useState('');
  const [sourceType,setSourceType] = useState<'internal'|'inspo'>('internal');
  const [text,setText] = useState('');
  const [url,setUrl] = useState('');
  const [status,setStatus] = useState('');
  const [styleTags,setStyleTags] = useState<string[]>([]);
  const [addPerf,setAddPerf] = useState(false);
  const [metrics,setMetrics] = useState({
    hook_rate:'', cpm:'', hold_rate:'', cpl:'', link_clicks:'', spend:'' // <<< spend added
  });

  const toggleStyle = (t:string)=> setStyleTags(s => s.includes(t)? s.filter(x=>x!==t): [...s,t]);

  async function onSubmit(e:any){
    e.preventDefault();
    setStatus('Uploading…');
    const body:any = { title, text, source_type: sourceType, url, style_tags: styleTags };
    if(addPerf && sourceType==='internal'){
      body.hook_rate   = num(metrics.hook_rate);
      body.cpm         = num(metrics.cpm);
      body.hold_rate   = num(metrics.hold_rate);
      body.cpl         = num(metrics.cpl);
      body.link_clicks = num(metrics.link_clicks || '');
      body.spend       = num(metrics.spend);  // <<< send spend
    }
    const res = await fetch(`${API_BASE}/ingest/text`,{
      method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)
    });
    if(!res.ok){ setStatus('❌ Failed'); return; }
    const json = await res.json();
    setStatus(`✅ Done. Chunks: ${json.chunks} | Saved: ${JSON.stringify(json.metadata_saved.metrics)}`);
    setTitle(''); setText(''); setUrl('');
    setMetrics({ hook_rate:'', cpm:'', hold_rate:'', cpl:'', link_clicks:'', spend:'' });
    setStyleTags([]); setAddPerf(false);
  }

  return (
    <main className="space-y-6">
      <section className="card space-y-4">
        <h2 className="text-xl font-medium">Upload to Library</h2>
        <form className="space-y-4" onSubmit={onSubmit}>
          <div>
            <label className="label">Title</label>
            <input className="input" value={title} onChange={e=>setTitle(e.target.value)} />
          </div>

          <div className="flex gap-3 items-center">
            <span className="label">Source type</span>
            <button type="button" className={`btn ${sourceType==='internal'?'btn-selected':''}`} onClick={()=>setSourceType('internal')}>internal</button>
            <button type="button" className={`btn ${sourceType==='inspo'?'btn-selected':''}`} onClick={()=>setSourceType('inspo')}>inspo</button>
          </div>

          <div>
            <label className="label">Optional URL (link to the ad)</label>
            <input className="input" value={url} onChange={e=>setUrl(e.target.value)} placeholder="https://…" />
          </div>

          <div>
            <label className="label">Style tags</label>
            <div className="flex flex-wrap gap-2 mt-2">
              {STYLE_OPTIONS.map(t=> (
                <button key={t} type="button" className={`btn ${styleTags.includes(t)?'btn-selected':''}`} onClick={()=>toggleStyle(t)}>{t}</button>
              ))}
            </div>
          </div>

          <div>
            <label className="label">Text (script / transcript)</label>
            <textarea className="input h-60" value={text} onChange={e=>setText(e.target.value)} placeholder="Paste the script or transcript…" />
          </div>

          {sourceType==='internal' && (
            <div className="space-y-3">
              <div className="flex items-center gap-3">
                <input id="perf" type="checkbox" checked={addPerf} onChange={e=>setAddPerf(e.target.checked)} />
                <label htmlFor="perf" className="label">Add performance data?</label>
              </div>
              {addPerf && (
                <div className="grid md:grid-cols-3 gap-3">
                  <div><label className="label">Hook Rate %</label><input className="input" value={metrics.hook_rate} onChange={e=>setMetrics({...metrics, hook_rate:e.target.value})} placeholder="e.g. 28" /></div>
                  <div><label className="label">Hold Rate %</label><input className="input" value={metrics.hold_rate} onChange={e=>setMetrics({...metrics, hold_rate:e.target.value})} placeholder="e.g. 12" /></div>
                  <div><label className="label">CPM</label><input className="input" value={metrics.cpm} onChange={e=>setMetrics({...metrics, cpm:e.target.value})} placeholder="e.g. $5.30" /></div>
                  <div><label className="label">CPL</label><input className="input" value={metrics.cpl} onChange={e=>setMetrics({...metrics, cpl:e.target.value})} placeholder="e.g. $7.90" /></div>
                  <div><label className="label">Link Clicks</label><input className="input" value={metrics.link_clicks} onChange={e=>setMetrics({...metrics, link_clicks:e.target.value})} placeholder="e.g. 124" /></div>
                  <div><label className="label">Spend</label><input className="input" value={metrics.spend} onChange={e=>setMetrics({...metrics, spend:e.target.value})} placeholder="e.g. $2,450.00" /></div>
                </div>
              )}
            </div>
          )}

          <div className="flex items-center gap-3">
            <button className="btn btn-selected" type="submit">Ingest</button>
            <div className="text-sm text-neutral-400">{status}</div>
          </div>
        </form>
      </section>
    </main>
  );
}
