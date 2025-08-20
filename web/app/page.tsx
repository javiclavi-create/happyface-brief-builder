'use client';
import React, { useState } from 'react';

const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? 'http://localhost:8000';

export default function Home() {
  const [form, setForm] = useState({
    industry: '',
    service: '',
    audience: '',
    platforms: 'TikTok, Reels',
    levers: ['juxtaposition', 'absurd escalation', 'deadpan'] as string[],
    n: 12,
  });
  const [loading, setLoading] = useState(false);
  const [markdown, setMarkdown] = useState<string>('');

  const toggleLever = (l: string) => {
    setForm((f) => ({
      ...f,
      levers: f.levers.includes(l)
        ? f.levers.filter((x) => x !== l)
        : [...f.levers, l],
    }));
  };

  async function onSubmit(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    setLoading(true);
    setMarkdown('');
    const res = await fetch(`${API_BASE}/generate/concepts`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        industry: form.industry,
        service: form.service,
        audience: form.audience,
        platforms: form.platforms.split(',').map((s) => s.trim()),
        levers: form.levers,
        n: form.n,
      }),
    });
    const json = await res.json();
    setMarkdown(json.markdown || '');
    setLoading(false);
  }

  return (
    <main className="space-y-6">
      <section className="card space-y-4">
        <h2 className="text-xl font-medium">New Brief</h2>

        <form
          className="grid grid-cols-1 md:grid-cols-2 gap-4"
          onSubmit={onSubmit}
        >
          <div>
            <label className="label">Industry</label>
            <input
              className="input"
              value={form.industry}
              onChange={(e) => setForm({ ...form, industry: e.target.value })}
              placeholder="e.g., Medspa"
            />
          </div>

          <div>
            <label className="label">Service</label>
            <input
              className="input"
              value={form.service}
              onChange={(e) => setForm({ ...form, service: e.target.value })}
              placeholder="e.g., Laser hair removal"
            />
          </div>

          <div>
            <label className="label">Audience</label>
            <input
              className="input"
              value={form.audience}
              onChange={(e) => setForm({ ...form, audience: e.target.value })}
              placeholder="e.g., Women 25–45, busy, value speed"
            />
          </div>

          <div>
            <label className="label">Platforms (comma-sep)</label>
            <input
              className="input"
              value={form.platforms}
              onChange={(e) => setForm({ ...form, platforms: e.target.value })}
            />
          </div>

          <div className="md:col-span-2">
            <label className="label">Funny levers</label>
            <div className="flex flex-wrap gap-2 mt-2">
              {['juxtaposition', 'absurd escalation', 'deadpan', 'pattern break']
                .map((l) => (
                  <button
                    key={l}
                    type="button"
                    className={`btn ${
                      form.levers.includes(l) ? 'bg-neutral-800' : ''
                    }`}
                    onClick={() => toggleLever(l)}
                  >
                    {l}
                  </button>
                ))}
            </div>
          </div>

          <div className="md:col-span-2 flex items-center gap-3">
            <label className="label"># Concepts</label>
            <input
              type="number"
              className="input w-24"
              value={form.n}
              min={6}
              max={30}
              onChange={(e) => setForm({ ...form, n: Number(e.target.value) })}
            />
            <button className="btn" type="submit" disabled={loading}>
              {loading ? 'Generating…' : 'Generate'}
            </button>
          </div>
        </form>
      </section>

      {markdown && (
        <section className="card">
          <h3 className="text-lg font-medium mb-3">Pitch Room Output</h3>
          <pre className="whitespace-pre-wrap text-sm">{markdown}</pre>
        </section>
      )}
    </main>
  );
}
