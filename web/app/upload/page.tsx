'use client';
import React, { useState } from 'react';
const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? 'http://localhost:8000';

export default function Upload() {
  const [title, setTitle] = useState('');
  const [sourceType, setSourceType] = useState<'internal' | 'inspo'>('internal');
  const [text, setText] = useState('');
  const [url, setUrl] = useState('');
  const [status, setStatus] = useState('');

  async function onSubmit(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    setStatus('Uploading…');
    try {
      const res = await fetch(`${API_BASE}/ingest/text`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ title, text, source_type: sourceType, url }),
      });
      if (!res.ok) {
        const t = await res.text();
        throw new Error(`API ${res.status}: ${t}`);
      }
      const json = await res.json();
      setStatus(`✅ Done. Chunks: ${json.chunks}`);
    } catch (e: any) {
      setStatus(`⚠️ ${e?.message || String(e)}`);
    }
  }

  return (
    <main className="space-y-6">
      <section className="card space-y-4">
        <h2 className="text-xl font-medium">Upload to Library</h2>
        <form onSubmit={onSubmit} className="space-y-3">
          <div>
            <label className="label">Title</label>
            <input className="input" value={title} onChange={(e) => setTitle(e.target.value)} />
          </div>
          <div>
            <label className="label">Source type</label>
            <div className="flex gap-2 mt-2">
              {(['internal', 'inspo'] as const).map((t) => (
                <button
                  type="button"
                  key={t}
                  onClick={() => setSourceType(t)}
                  className={`btn ${sourceType === t ? 'bg-neutral-800' : ''}`}
                >
                  {t}
                </button>
              ))}
            </div>
          </div>
          <div>
            <label className="label">Optional URL</label>
            <input
              className="input"
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              placeholder="https://…"
            />
          </div>
          <div>
            <label className="label">Text</label>
            <textarea
              className="input h-60"
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="Paste script/brief/article…"
            ></textarea>
          </div>
          <button className="btn" type="submit">Ingest</button>
          {status && <p className="text-sm text-neutral-400">{status}</p>}
          <p className="text-xs text-neutral-500">API base: {API_BASE}</p>
        </form>
      </section>
    </main>
  );
}
