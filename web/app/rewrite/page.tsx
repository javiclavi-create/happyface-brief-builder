'use client';
import { useState } from 'react';
const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8000';


export default function Rewrite() {
const [concept, setConcept] = useState('');
const [duration, setDuration] = useState(30);
const [platform, setPlatform] = useState('TikTok');
const [style, setStyle] = useState('faceless');
const [script, setScript] = useState('');
const [loading, setLoading] = useState(false);


async function onSubmit(e:any){
e.preventDefault();
setLoading(true);
setScript('');
const res = await fetch(`${API_BASE}/generate/rewrite`,{
method:'POST', headers:{'Content-Type':'application/json'},
body: JSON.stringify({ concept_text: concept, duration_s: duration, platform, style })
});
const json = await res.json();
setScript(json.script || '');
setLoading(false);
}


return (
<main className="space-y-6">
<section className="card space-y-4">
<h2 className="text-xl font-medium">Rewrite Studio</h2>
<form onSubmit={onSubmit} className="space-y-3">
<textarea className="input h-40" value={concept} onChange={e=>setConcept(e.target.value)} placeholder="Paste a concept from Pitch Room…"></textarea>
<div className="flex gap-3 items-center flex-wrap">
<label className="label">Duration</label>
<select className="input" value={duration} onChange={e=>setDuration(Number(e.target.value))}>
{[15,30,60].map(d=>(<option key={d} value={d}>{d}s</option>))}
</select>
<label className="label">Platform</label>
<select className="input" value={platform} onChange={e=>setPlatform(e.target.value)}>
{['TikTok','Reels','YT Shorts'].map(p=>(<option key={p}>{p}</option>))}
</select>
<label className="label">Style</label>
<select className="input" value={style} onChange={e=>setStyle(e.target.value)}>
{['faceless','UGC','actor'].map(s=>(<option key={s}>{s}</option>))}
</select>
<button className="btn" type="submit" disabled={loading}>{loading? 'Generating…' : 'Generate Script'}</button>
</div>
</form>
</section>


{script && (
<section className="card">
<h3 className="text-lg font-medium mb-3">Generated Script</h3>
<pre className="whitespace-pre-wrap text-sm">{script}</pre>
</section>
)}
</main>
)
}
