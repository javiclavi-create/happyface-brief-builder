'use client';
import { useEffect, useState } from 'react';

export default function RewritePage(){
  const [items,setItems] = useState<{title:string, script:string}[]>([]);

  useEffect(()=>{
    try{
      const raw = localStorage.getItem('rewrites');
      if(raw){ setItems(JSON.parse(raw)); }
    }catch{}
  },[]);

  return (
    <main className="space-y-6">
      <section className="flex items-center justify-between">
        <h2 className="text-xl font-medium">Rewrite Studio</h2>
        <button
          className="btn"
          onClick={()=>{ navigator.clipboard.writeText(items.map(i=>`## ${i.title}\n\n${i.script}`).join('\n\n---\n\n')); }}
        >
          Copy All
        </button>
      </section>
      <div className="grid md:grid-cols-2 gap-4">
        {items.map((i,idx)=> (
          <div key={idx} className="card space-y-2">
            <h3 className="font-semibold">{i.title}</h3>
            <pre className="whitespace-pre-wrap text-sm">{i.script}</pre>
            <div className="pt-2">
              <button className="btn" onClick={()=>navigator.clipboard.writeText(i.script)}>Copy</button>
            </div>
          </div>
        ))}
      </div>
    </main>
  );
}
