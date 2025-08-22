'use client';
import { useEffect, useMemo, useState } from 'react';
const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8000';

type Doc = {
  id: string;
  source_type: 'internal'|'inspo';
  title: string | null;
  url: string | null;
  created_at: string;
  metadata?: { style_tags?: string[], metrics?: Record<string, any> };
};

type StyleCount = { style: string; count: number };

const number = (v:any)=> (v===0 || v) ? Number(v) : undefined;

export default function LibraryPage(){
  const [loading, setLoading] = useState(true);
  const [docs, setDocs] = useState<Doc[]>([]);
  const [styles, setStyles] = useState<StyleCount[]>([]);
  const [totals, setTotals] = useState<Record<string, number>>({});
  const [filterType, setFilterType] = useState<'all'|'internal'|'inspo'>('all');
  const [filterTag, setFilterTag] = useState<string | null>(null);
  const [q, setQ] = useState('');

  const [editing, setEditing] = useState<string|null>(null);
  const [editData, setEditData] = useState<any>({});

  async function load(){
    setLoading(true);
    const [sumRes, listRes] = await Promise.all([
      fetch(`${API_BASE}/library/summary`),
      fetch(`${API_BASE}/documents`),
    ]);
    const sum = await sumRes.json();
    const list = await listRes.json();
    setStyles(sum.styles || []);
    setTotals(sum.totals || {});
    setDocs(list.documents || []);
    setLoading(false);
  }

  useEffect(()=>{ load(); },[]);

  const filtered = useMemo(()=>{
    return docs.filter(d=>{
      if(filterType!=='all' && d.source_type!==filterType) return false;
      if(filterTag){
        const tags = d.metadata?.style_tags || [];
        if(!tags.includes(filterTag)) return false;
      }
      if(q.trim()){
        const t = (d.title || '').toLowerCase();
        if(!t.includes(q.toLowerCase())) return false;
      }
      return true;
    });
  }, [docs, filterType, filterTag, q]);

  async function startEdit(d: Doc){
    setEditing(d.id);
    setEditData({
      title: d.title || '',
      url: d.url || '',
      source_type: d.source_type,
      style_tags: [...(d.metadata?.style_tags || [])],
      hook_rate: d.metadata?.metrics?.hook_rate ?? '',
      hold_rate: d.metadata?.metrics?.hold_rate ?? '',
      cpm: d.metadata?.metrics?.cpm ?? '',
      cpl: d.metadata?.metrics?.cpl ?? '',
      link_clicks: d.metadata?.metrics?.link_clicks ?? '',
      spend: d.metadata?.metrics?.spend ?? '',
    });
  }

  async function saveEdit(id: string){
    const body:any = {
      title: editData.title,
      url: editData.url,
      source_type: editData.source_type,
      style_tags: editData.style_tags,
      hook_rate: number(editData.hook_rate),
      hold_rate: number(editData.hold_rate),
      cpm: number(editData.cpm),
      cpl: number(editData.cpl),
      link_clicks: editData.link_clicks===''? undefined : Number(editData.link_clicks),
      spend: number(editData.spend),
    };
    const res = await fetch(`${API_BASE}/documents/${id}`,{
      method:'PATCH',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify(body)
    });
    if(!res.ok){ alert('Save failed'); return; }
    setEditing(null);
    await load();
  }

  const allTags = styles.map(s=>s.style).filter(Boolean);

  return (
    <main className="space-y-6">
      <section className="card space-y-4">
        <h2 className="text-xl font-medium">Library Overview</h2>
        <div className="flex flex-wrap items-center gap-3">
          <span className="badge">internal: {totals.internal || 0}</span>
          <span className="badge">inspo: {totals.inspo || 0}</span>
        </div>

        <div className="space-y-2">
          <div className="label">Style counter</div>
          <div className="flex flex-wrap gap-2">
            {styles.length===0 && <div className="text-sm text-neutral-400">No styles yet</div>}
            {styles.map(s=>(
              <button
                key={s.style}
                className={`chip ${filterTag===s.style ? 'chip-selected':''}`}
                onClick={()=> setFilterTag(filterTag===s.style? null : s.style)}
                title="Filter by this style"
              >
                {s.style} • {s.count}
              </button>
            ))}
            {filterTag && (
              <button className="chip" onClick={()=>setFilterTag(null)}>Clear style filter</button>
            )}
          </div>
        </div>

        <div className="grid md:grid-cols-3 gap-3">
          <div className="flex gap-2">
            <button className={`btn ${filterType==='all'?'btn-selected':''}`} onClick={()=>setFilterType('all')}>All</button>
            <button className={`btn ${filterType==='internal'?'btn-selected':''}`} onClick={()=>setFilterType('internal')}>Internal</button>
            <button className={`btn ${filterType==='inspo'?'btn-selected':''}`} onClick={()=>setFilterType('inspo')}>Inspo</button>
          </div>
          <div className="md:col-span-2">
            <input className="input" placeholder="Search title…" value={q} onChange={e=>setQ(e.target.value)} />
          </div>
        </div>
      </section>

      <section className="space-y-4">
        <h3 className="text-lg font-medium">Items</h3>
        {loading && <div className="card">Loading…</div>}
        {!loading && filtered.length===0 && <div className="card">No items match your filters.</div>}
        <div className="grid md:grid-cols-2 gap-4">
          {filtered.map(d=>{
            const tags = d.metadata?.style_tags || [];
            const m = d.metadata?.metrics || {};
            const isEditing = editing===d.id;
            return (
              <div key={d.id} className="card space-y-3">
                <div className="flex items-center justify-between gap-2">
                  <div className="flex items-center gap-2">
                    <span className="badge">{d.source_type}</span>
                    <span className="text-sm text-neutral-400">{new Date(d.created_at).toLocaleString()}</span>
                  </div>
                  <div className="flex gap-2">
                    {!isEditing && <button className="btn" onClick={()=>startEdit(d)}>Edit</button>}
                    {isEditing && (
                      <>
                        <button className="btn btn-selected" onClick={()=>saveEdit(d.id)}>Save</button>
                        <button className="btn" onClick={()=>setEditing(null)}>Cancel</button>
                      </>
                    )}
                  </div>
                </div>

                {!isEditing ? (
                  <>
                    <h4 className="text-lg">{d.title || 'Untitled'}</h4>
                    {d.url && <a className="text-sm text-blue-400 underline" href={d.url} target="_blank">Open link</a>}
                    <div className="flex flex-wrap gap-2">
                      {tags.map(t=>(
                        <span key={t} className="chip">{t}</span>
                      ))}
                      {tags.length===0 && <span className="label">No style tags</span>}
                    </div>

                    {Object.keys(m).length>0 && (
                      <div className="grid-metrics">
                        {"hook_rate" in m && <div className="metric">Hook Rate: <b>{m.hook_rate}%</b></div>}
                        {"hold_rate" in m && <div className="metric">Hold Rate: <b>{m.hold_rate}%</b></div>}
                        {"cpm" in m && <div className="metric">CPM: <b>${m.cpm}</b></div>}
                        {"cpl" in m && <div className="metric">CPL: <b>${m.cpl}</b></div>}
                        {"link_clicks" in m && <div className="metric">Link Clicks: <b>{m.link_clicks}</b></div>}
                        {"spend" in m && <div className="metric">Spend: <b>${m.spend}</b></div>}
                      </div>
                    )}
                  </>
                ) : (
                  <>
                    <div>
                      <div className="label">Title</div>
                      <input className="input" value={editData.title} onChange={e=>setEditData({...editData, title:e.target.value})} />
                    </div>
                    <div>
                      <div className="label">URL</div>
                      <input className="input" value={editData.url} onChange={e=>setEditData({...editData, url:e.target.value})} placeholder="https://…" />
                    </div>

                    <div className="flex gap-2">
                      <button type="button" className={`btn ${editData.source_type==='internal'?'btn-selected':''}`} onClick={()=>setEditData({...editData, source_type:'internal'})}>internal</button>
                      <button type="button" className={`btn ${editData.source_type==='inspo'?'btn-selected':''}`} onClick={()=>setEditData({...editData, source_type:'inspo'})}>inspo</button>
                    </div>

                    <div>
                      <div className="label mb-2">Style tags (click to toggle)</div>
                      <div className="flex flex-wrap gap-2">
                        {allTags.map(t=>(
                          <button
                            key={t}
                            type="button"
                            className={`chip ${editData.style_tags?.includes(t)?'chip-selected':''}`}
                            onClick={()=>{
                              const exists = editData.style_tags?.includes(t);
                              const next = exists
                                ? editData.style_tags.filter((x:string)=>x!==t)
                                : [...(editData.style_tags||[]), t];
                              setEditData({...editData, style_tags: next});
                            }}
                          >{t}</button>
                        ))}
                      </div>
                    </div>

                    <div className="grid md:grid-cols-3 gap-3">
                      <div><div className="label">Hook Rate %</div><input className="input" value={editData.hook_rate} onChange={e=>setEditData({...editData, hook_rate:e.target.value})} /></div>
                      <div><div className="label">Hold Rate %</div><input className="input" value={editData.hold_rate} onChange={e=>setEditData({...editData, hold_rate:e.target.value})} /></div>
                      <div><div className="label">CPM</div><input className="input" value={editData.cpm} onChange={e=>setEditData({...editData, cpm:e.target.value})} /></div>
                      <div><div className="label">CPL</div><input className="input" value={editData.cpl} onChange={e=>setEditData({...editData, cpl:e.target.value})} /></div>
                      <div><div className="label">Link Clicks</div><input className="input" value={editData.link_clicks} onChange={e=>setEditData({...editData, link_clicks:e.target.value})} /></div>
                      <div><div className="label">Spend</div><input className="input" value={editData.spend} onChange={e=>setEditData({...editData, spend:e.target.value})} /></div>
                    </div>
                  </>
                )}
              </div>
            )
          })}
        </div>
      </section>
    </main>
  );
}
