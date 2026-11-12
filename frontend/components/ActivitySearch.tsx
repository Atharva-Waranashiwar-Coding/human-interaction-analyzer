'use client';
import { FormEvent, useState } from 'react';
import { Annotation } from '../lib/types';
type Result = { video_id: string; video_name: string; annotation: Annotation };
const fields = [
  ['behavior', 'Behavior or activity'], ['object_category', 'Object category'], ['person_count', 'Exact person count'],
  ['participant_id', 'Participant ID'], ['object_id', 'Object ID'], ['start_time', 'From seconds'], ['end_time', 'To seconds'],
  ['sequence', 'Ordered actions separated by >'],
];
export default function ActivitySearch({ api, videos, onOpen }: { api: string; videos: {id: string; original_name: string}[]; onOpen: (id: string, time: number, annotationId?: string) => void }) {
  const [filters, setFilters] = useState<Record<string,string>>({});
  const [result, setResult] = useState<{total: number; offset: number; results: Result[]} | null>(null);
  const [error, setError] = useState(''); const [loading, setLoading] = useState(false);
  const update = (key: string, value: string) => setFilters(p => ({...p,[key]:value}));
  const search = async (offset = 0) => {
    setError(''); setLoading(true);
    try {
      const params = new URLSearchParams(Object.entries(filters).filter(([,v]) => v !== ''));
      params.set('offset', String(offset)); params.set('limit','25');
      const response = await fetch(`${api}/api/activities/search?${params}`);
      if (!response.ok) throw new Error(await response.text());
      setResult(await response.json());
    } catch (e) { setError(e instanceof Error ? e.message : 'Search failed'); }
    finally { setLoading(false); }
  };
  const submit = (e: FormEvent) => { e.preventDefault(); void search(); };
  return <details className="mb-6 rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
    <summary className="cursor-pointer font-semibold">Search activities across videos</summary>
    <p className="my-3 text-sm text-slate-500">Search saved annotations, including unreviewed suggestions. IDs are local to each video. Example sequence: picking up / lifting &gt; carrying &gt; placing / putting down.</p>
    <form onSubmit={submit} className="grid gap-3 md:grid-cols-3">
      {fields.map(([key,label]) => <label key={key} className="text-sm">{label}<input className="mt-1 w-full rounded-lg border p-2" value={filters[key] || ''} onChange={e=>update(key,e.target.value)} type={['person_count','start_time','end_time'].includes(key)?'number':'text'} min="0" step={key==='person_count'?'1':'any'} /></label>)}
      <label className="text-sm">Review status<select className="mt-1 w-full rounded-lg border p-2" value={filters.review_status || ''} onChange={e=>update('review_status',e.target.value)}><option value="">Any</option>{['needs_review','reviewed','rejected'].map(s=><option key={s}>{s}</option>)}</select></label>
      <label className="text-sm">Annotation type<select className="mt-1 w-full rounded-lg border p-2" value={filters.annotation_kind || ''} onChange={e=>update('annotation_kind',e.target.value)}><option value="">Any</option>{['behavior','activity','spatial','audio'].map(s=><option key={s}>{s}</option>)}</select></label>
      <label className="text-sm">Video<select className="mt-1 w-full rounded-lg border p-2" value={filters.video_id || ''} onChange={e=>update('video_id',e.target.value)}><option value="">All videos</option>{videos.map(v=><option value={v.id} key={v.id}>{v.original_name}</option>)}</select></label>
      <button disabled={loading} className="self-end rounded-xl bg-slate-900 p-2 text-sm text-white disabled:opacity-50">{loading?'Searching…':'Search saved activities'}</button>
    </form>
    {error && <p role="alert" className="mt-3 text-sm text-red-700">{error}</p>}
    {result && <div className="mt-4 overflow-x-auto"><p className="mb-2 text-sm">{result.total} matches</p><table className="w-full text-left text-sm"><thead><tr>{['Video and time','Behavior or activity','Entities','Review','Open'].map(h=><th key={h} className="border-b p-2">{h}</th>)}</tr></thead><tbody>{result.results.map(r=><tr key={`${r.video_id}-${r.annotation.id}`}><td className="border-b p-2">{r.video_name}<br />{r.annotation.start_time.toFixed(2)}–{r.annotation.end_time.toFixed(2)}s</td><td className="border-b p-2">{r.annotation.behavior}</td><td className="border-b p-2">{r.annotation.participants.join(', ')}<br />{r.annotation.objects?.map(o=>`${o.id} (${o.class_name})`).join(', ')}</td><td className="border-b p-2">{r.annotation.review_status}</td><td className="border-b p-2"><button className="text-sky-700 underline" onClick={()=>onOpen(r.video_id,r.annotation.start_time,r.annotation.id)}>Open in editor</button></td></tr>)}</tbody></table>
      <div className="mt-3 flex gap-3"><button disabled={result.offset===0 || loading} onClick={()=>search(Math.max(0,result.offset-25))}>Previous</button><button disabled={result.offset+25>=result.total || loading} onClick={()=>search(result.offset+25)}>Next</button></div></div>}
  </details>;
}
