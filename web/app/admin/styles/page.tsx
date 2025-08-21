'use client';
import { useEffect, useState } from 'react';
const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8000';

type Row = { style: string; doc_count: number };

export default function StylesAdmin(){
  const [rows, setRows] = useState<Row[]>([]);
  useEffect(()=>{
    fetch(`${API_BASE}/stats/styles`).then(r=>r.json()).then(j=>setRows(j.styles||[]));
  },[]);
  return (
    <main className="space-y-6">
      <section className="card-pro">
        <h2 className="text-xl font-medium">Style Coverage</h2>
        <p className="text-sm text-neutral-400 mt-1">How much training material we’ve fed in each bucket.</p>
      </section>
      <section className="card-pro">
        <table className="w-full text-sm">
          <thead><tr><th className="text-left p-2">Style</th><th className="text-left p-2">Docs</th></tr></thead>
          <tbody>
            {rows.map((r,i)=>(
              <tr key={i} className="border-t border-neutral-800">
                <td className="p-2">{r.style}</td>
                <td className="p-2">{r.doc_count}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>
    </main>
  );
}
