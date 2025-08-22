'use client';
import { useState } from 'react';
const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8000';

export default function Upload(){
  const [title,setTitle] = useState('');
  const [sourceType,setSourceType] = useState<'internal'|'inspo'>('internal');
  const [mediaKind,setMediaKind] = useState<'static'|'video'>('static');
  const [contentStyle,setContentStyle] = useState('');
  const [link,setLink] = useState('');
  const [file,setFile] = useState<File|null>(null);

  // Meta metrics (all optional)
  const [hookRate,setHookRate] = useState<string>('');    // %
  const [holdRate,setHoldRate] = useState<string>('');    // %
  const [cpm,setCpm] = useState<string>('');
  const [cpl,setCpl] = useState<string>('');
  const [linkClicks,setLinkClicks] = useState<string>('');
  const [spend,setSpend] = useState<string>('');

  const [status,setStatus] = useState('');

  async function onSubmit(e: React.FormEvent){
    e.preventDefault();
    setStatus('Uploading…');

    const fd = new FormData();
    fd.append('title', title);
    fd.append('source_type', sourceType);
    fd.append('media_kind', mediaKind);
    if (contentStyle) fd.append('content_style', contentStyle);
    if (link) fd.append('url', link);

    if (hookRate) fd.append('hook_rate', hookRate);
    if (holdRate) fd.append('hold_rate', holdRate);
    if (cpm) fd.append('cpm', cpm);
    if (cpl) fd.append('cpl', cpl);
    if (linkClicks) fd.append('link_clicks', linkClicks);
    if (spend) fd.append('spend', spend);

    if (file) fd.append('file', file);

    try{
      const res = await fetch(`${API_BASE}/ingest/media`, { method:'POST', body: fd });
      const json = await res.json();
      if (!res.ok) throw new Error(json.detail || 'Upload failed');
      setStatus(`✅ Saved. Chunks: ${json.chunks}${json.url ? ' | URL: '+json.url : ''}`);
      setTitle(''); setLink(''); setFile(null);
      setHookRate(''); setHoldRate(''); setCpm(''); setCpl(''); setLinkClicks(''); setSpend('');
    } catch(err:any){
      setStatus(`❌ ${err.message || 'Error'}`);
    }
  }

  return (
    <main className="space-y-6">
      <section className="card space-y-4">
        <h2 className="text-xl font-medium">Upload to Library</h2>

        <form className="grid grid-cols-1 md:grid-cols-2 gap-4" onSubmit={onSubmit}>
          <div><label className="label">Title</label><input className="input" value={title} onChange={e=>setTitle(e.target.value)} /></div>

          <div>
            <label className="label">Source</label>
            <div className="flex gap-2 mt-2">
              <button type="button" className={`chip ${sourceType==='internal'?'chip-selected':''}`} onClick={()=>setSourceType('internal')}>Internal</button>
              <button type="button" className={`chip ${sourceType==='inspo'?'chip-selected':''}`} onClick={()=>setSourceType('inspo')}>Inspo</button>
            </div>
          </div>

          <div>
            <label className="label">Asset</label>
            <div className="flex gap-2 mt-2">
              <button type="button" className={`chip ${mediaKind==='static'?'chip-selected':''}`} onClick={()=>setMediaKind('static')}>Static</button>
              <button type="button" className={`chip ${mediaKind==='video'?'chip-selected':''}`} onClick={()=>setMediaKind('video')}>Video</button>
            </div>
          </div>

          <div>
            <label className="label">Content Style (optional)</label>
            <input className="input" placeholder="UGC / Faceless / Skit / Parody / Talking Head…" value={contentStyle} onChange={e=>setContentStyle(e.target.value)} />
          </div>

          <div className="md:col-span-2">
            <label className="label">Link (optional)</label>
            <input className="input" placeholder="https://…" value={link} onChange={e=>setLink(e.target.value)} />
          </div>

          <div className="md:col-span-2">
            <label className="label">Upload File (optional)</label>
            <input type="file" className="input" onChange={(e)=>setFile(e.target.files?.[0] || null)} />
          </div>

          {/* Metrics */}
          <div className="md:col-span-2"><div className="label">Meta Metrics (optional)</div></div>
          <div><label className="label">Hook Rate %</label><input className="input" value={hookRate} onChange={e=>setHookRate(e.target.value)} /></div>
          <div><label className="label">Hold Rate %</label><input className="input" value={holdRate} onChange={e=>setHoldRate(e.target.value)} /></div>
          <div><label className="label">CPM</label><input className="input" value={cpm} onChange={e=>setCpm(e.target.value)} /></div>
          <div><label className="label">CPL</label><input className="input" value={cpl} onChange={e=>setCpl(e.target.value)} /></div>
          <div><label className="label">Link Clicks</label><input className="input" value={linkClicks} onChange={e=>setLinkClicks(e.target.value)} /></div>
          <div><label className="label">Spend</label><input className="input" value={spend} onChange={e=>setSpend(e.target.value)} /></div>

          <div className="md:col-span-2">
            <button className="btn" type="submit">Save to Library</button>
          </div>
        </form>

        {status && <div className="text-sm">{status}</div>}
      </section>
    </main>
  );
}
