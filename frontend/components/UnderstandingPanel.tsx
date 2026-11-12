'use client';
import { useEffect, useState } from 'react';
import { Annotation } from '../lib/types';

export default function UnderstandingPanel({ api, video, annotations, onRefresh, onEdit, onAddAudio }: {
  api: string; video: any; annotations: Annotation[]; onRefresh: () => Promise<void>;
  onEdit: (annotation: Annotation) => void; onAddAudio: () => void;
}) {
  const [regions,setRegions] = useState('[]'); const [message,setMessage] = useState(''); const [busy,setBusy] = useState(false);
  const [covered,setCovered] = useState('carrying'); const [start,setStart] = useState(0); const [end,setEnd] = useState(video.duration);
  const [scenario,setScenario] = useState('unspecified'); const [complete,setComplete] = useState(false);
  const [evaluation,setEvaluation] = useState<any>(null);
  useEffect(()=>setRegions(JSON.stringify(video.scene_regions || [],null,2)),[video.scene_regions]);
  useEffect(()=>setEvaluation(null),[annotations]);
  const request = async (path: string, method='POST', body?: unknown) => {
    if(path!=='evaluation' && JSON.stringify(annotations)!==JSON.stringify(video.annotations || [])){setMessage('Save annotation edits before updating regions, suggestions or audio metadata.');return null;}
    setBusy(true); setMessage('');
    try {
      const response=await fetch(`${api}/api/videos/${video.id}/${path}`,{method,headers:{'Content-Type':'application/json'},body:body===undefined?undefined:JSON.stringify(body)});
      if(!response.ok) throw new Error(await response.text());
      const data=await response.json(); if(path!=='evaluation') await onRefresh(); setMessage('Saved.');return data;
    } catch(e){setMessage(e instanceof Error?e.message:'Request failed');return null;}
    finally {setBusy(false);}
  };
  const saveRegions=()=>{try {void request('regions','PUT',{regions:JSON.parse(regions)});}catch {setMessage('Regions must be valid JSON.');}};
  const advanced=annotations.filter(a=>['spatial','activity','audio'].includes(a.annotation_kind || ''));
  return <div className="space-y-5 text-sm">
    <p className="text-slate-600">Spatial relations and activities use saved detections and actions. All generated results require review. Save annotation edits before updating cached suggestions or regions.</p>
    <button disabled={busy || video.analysis_status==='processing'} onClick={()=>request('understanding')} className="rounded-lg border px-3 py-2 disabled:opacity-50">Update suggestions from saved results</button>
    {message && <p role="status" className="text-slate-700">{message}</p>}
    <div className="overflow-x-auto"><table className="w-full text-left"><thead><tr>{['Time','Type and label','Evidence and sequence','Review','Edit'].map(h=><th className="border-b p-2" key={h}>{h}</th>)}</tr></thead><tbody>{advanced.map(a=><tr key={a.id}><td className="border-b p-2">{a.start_time.toFixed(2)}–{a.end_time.toFixed(2)}s</td><td className="border-b p-2">{a.annotation_kind}<br />{a.behavior}</td><td className="border-b p-2">{a.description}<br />{a.details?.sequence_status && `Sequence: ${a.details.sequence_status}`}<br />{a.details?.action_ids?.map((id: string)=>annotations.find(x=>x.id===id)?.behavior || `Missing action ${id}`).join(' → ')}{a.details?.subject_id && <span>{a.details.subject_id} → {a.details.target_id}</span>}<br />{a.details?.source_region && `From ${a.details.source_region} `}{a.details?.destination_region && `to ${a.details.destination_region}`}</td><td className="border-b p-2">{a.review_status}</td><td className="border-b p-2"><button onClick={()=>onEdit(a)} className="text-sky-700 underline">Review in editor</button></td></tr>)}</tbody></table>{!advanced.length && <p className="py-4 text-slate-500">No spatial, activity or audio annotations yet.</p>}</div>
    <details><summary className="cursor-pointer font-semibold">Configure scene regions</summary><p className="my-2 text-slate-600">Regions use normalized coordinates [left, top, right, bottom] in the square sampled image, including letterbox padding. They are fixed image areas and require a static-camera estimate for automatic transitions. Boxes do not establish physical containment or depth.</p>
      <p className="mb-2">Example: {JSON.stringify([{id:'table_area',label:'Table',kind:'table',box:[0.4,0.4,0.9,0.8]}])}</p>
      <textarea aria-label="Scene regions JSON" rows={8} className="w-full rounded-lg border p-2 font-mono" value={regions} onChange={e=>setRegions(e.target.value)} />
      <button disabled={busy || video.analysis_status==='processing'} onClick={saveRegions} className="mt-2 rounded-lg border px-3 py-2">Save regions and update suggestions</button></details>
    <details><summary className="cursor-pointer font-semibold">Evaluation against independent ground truth</summary><p className="my-2 text-slate-600">Create separate manual ground-truth annotations in the editor and mark them Reviewed. Accepting a suggestion does not make it ground truth. Metrics compare original reviewed predictions, including rejected ones, against complete ground truth using exact entity IDs/classes and temporal IoU ≥ 0.5. Results apply only to this selected video scope.</p>
      <div className="grid gap-3 md:grid-cols-2"><label>Covered behaviors separated by commas<input className="mt-1 w-full rounded border p-2" value={covered} onChange={e=>setCovered(e.target.value)} /></label><label>Scenario<input className="mt-1 w-full rounded border p-2" value={scenario} onChange={e=>setScenario(e.target.value)} /></label><label>Start seconds<input type="number" min="0" step="any" className="mt-1 w-full rounded border p-2" value={start} onChange={e=>setStart(Number(e.target.value))} /></label><label>End seconds<input type="number" min="0" step="any" className="mt-1 w-full rounded border p-2" value={end} onChange={e=>setEnd(Number(e.target.value))} /></label></div>
      <label className="my-3 flex items-center gap-2"><input type="checkbox" checked={complete} onChange={e=>setComplete(e.target.checked)} />I have exhaustively labeled the selected behaviors in this interval, including missed actions.</label>
      <button disabled={busy} className="rounded-lg border px-3 py-2" onClick={async()=>{const r=await request('evaluation','POST',{covered_behaviors:covered.split(',').map(s=>s.trim()).filter(Boolean),start_time:start,end_time:end,complete_ground_truth:complete,scenario});setEvaluation(r);}}>Calculate saved evaluation</button>
      {evaluation && (evaluation.available?<><p className="mt-3">Scenario: {evaluation.scenario}. Reviewed predictions: {evaluation.reviewed_predictions}; ground truth: {evaluation.ground_truth_annotations}; unreviewed predictions excluded: {evaluation.unreviewed_predictions_excluded}.</p><div className="overflow-x-auto"><table className="mt-3 w-full text-left"><thead><tr>{['Behavior','TP','FP','FN','Precision','Recall','F1'].map(h=><th key={h} className="p-2">{h}</th>)}</tr></thead><tbody>{Object.entries({Overall:evaluation.summary,...evaluation.per_behavior}).map(([label,m]:[string,any])=><tr key={label}><td className="p-2">{label}</td>{['true_positives','false_positives','false_negatives','precision','recall','f1'].map(k=><td className="p-2" key={k}>{m[k]===null?'Not defined':Number.isInteger(m[k])?m[k]:m[k].toFixed(3)}</td>)}</tr>)}</tbody></table></div></>:<p className="mt-3">{evaluation.reason}</p>)}
    </details>
    <details><summary className="cursor-pointer font-semibold">Optional audio metadata and manual events</summary><p className="my-2 text-slate-600">Disabled until requested. This probes track metadata only, without audio extraction, speech recognition, speaker identification, or event classification.</p><button disabled={busy || video.analysis_status==='processing'} onClick={()=>request('audio')} className="rounded-lg border px-3 py-2">Inspect audio metadata</button>
      {video.audio_metadata?.enabled && <><p className="mt-2">Audio track: {video.audio_metadata.has_audio?'present':'absent'}</p>{video.audio_metadata.streams.map((s:any)=><p key={s.stream_index}>Stream {s.stream_index}: {s.codec}, {s.sample_rate || 'unknown'} Hz, {s.channels || 'unknown'} channels, {s.start_time.toFixed(2)}–{s.end_time.toFixed(2)}s ({s.timing_source})</p>)}</>}
      <button disabled={video.analysis_status==='processing'} onClick={onAddAudio} className="ml-2 rounded-lg border px-3 py-2">Add manual audio event</button></details>
  </div>;
}
