'use client';

import { ChangeEvent, useEffect, useMemo, useRef, useState } from 'react';
import DetectionOverlay from '../components/DetectionOverlay';

type ObjectReference = { id: string; class_name: string };

type Annotation = {
  id?: string;
  start_time: number;
  end_time: number;
  behavior: string;
  participants: string[];
  objects?: ObjectReference[];
  parent_activity_id?: string | null;
  description: string;
  source: string;
  confidence?: number | null;
  review_status: string;
};

type VideoItem = {
  id: string;
  original_name: string;
  duration: number;
  width: number;
  height: number;
  status: string;
  analysis_status: string;
  message?: string | null;
};

const OBJECT_AND_MOVEMENT_LABELS = [
  'reaching toward an object', 'picking up / lifting', 'holding', 'carrying', 'walking while carrying',
  'placing / putting down', 'releasing', 'dropping', 'pushing / pulling', 'dragging / sliding',
  'stacking / unstacking', 'opening / closing', 'inserting / removing', 'throwing / catching',
  'walking', 'running', 'standing', 'sitting', 'standing up', 'sitting down', 'bending', 'crouching',
  'kneeling', 'reaching', 'turning', 'jumping', 'walking backward', 'handing over / receiving objects',
  'passing objects', 'joint carrying / lifting', 'assisting with object movement', 'packing / unpacking',
  'sorting / organizing', 'loading / unloading', 'moving objects between locations', 'repeated manipulation sequences',
];

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:8000';

const emptyAnnotation = (): Annotation => ({
  id: crypto.randomUUID(),
  objects: [],
  parent_activity_id: null,
  start_time: 0,
  end_time: 1,
  behavior: 'greeting',
  participants: ['person_1', 'person_2'],
  description: '',
  source: 'manual',
  confidence: null,
  review_status: 'needs_review',
});

async function apiFetch<T>(path: string, options: RequestInit = {}): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
    ...options,
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || 'Request failed');
  }
  if (response.status === 204) {
    return undefined as T;
  }
  return response.json() as Promise<T>;
}

export default function HomePage() {
  const [videos, setVideos] = useState<VideoItem[]>([]);
  const [selectedId, setSelectedId] = useState<string>('');
  const selectedVideoIdRef = useRef(selectedId);
  selectedVideoIdRef.current = selectedId;
  const [selectedVideo, setSelectedVideo] = useState<any | null>(null);
  const [annotations, setAnnotations] = useState<Annotation[]>([]);
  const [uploading, setUploading] = useState(false);
  const [uploadResults, setUploadResults] = useState<{ name: string; error?: string }[]>([]);
  const [statusMessage, setStatusMessage] = useState('');
  const [videoTab, setVideoTab] = useState<'review' | 'table'>('review');
  const [annotationFilter, setAnnotationFilter] = useState('');
  const [detectionPage, setDetectionPage] = useState(0);
  const [exportFormat, setExportFormat] = useState<'json' | 'jsonl'>('json');

  const selectedAnnotation = useMemo(
    () => annotations.find((item) => item.id === selectedId) || null,
    [annotations, selectedId],
  );

  const refreshVideos = async (selectFirst = true) => {
    try {
      const data = await apiFetch<VideoItem[]>('/api/videos');
      setVideos(data);
      if (selectFirst && !selectedId && data[0]) {
        setSelectedId(data[0].id);
      }
    } catch (error) {
      setStatusMessage(error instanceof Error ? error.message : 'Unable to load videos');
    }
  };

  const refreshVideoDetail = async (videoId: string) => {
    if (!videoId) return;
    try {
      const detail = await apiFetch<any>(`/api/videos/${videoId}`);
      if (selectedVideoIdRef.current !== videoId) return;
      setSelectedVideo(detail);
      setAnnotations(detail.annotations || []);
      setStatusMessage(detail.message || 'Video loaded');
    } catch (error) {
      setStatusMessage(error instanceof Error ? error.message : 'Unable to load video');
    }
  };

  useEffect(() => {
    void refreshVideos();
  }, []);

  useEffect(() => {
    setVideoTab('review');
    setAnnotationFilter('');
    setDetectionPage(0);
    if (selectedId) {
      void refreshVideoDetail(selectedId);
    }
  }, [selectedId]);

  useEffect(() => {
    if (!selectedId || selectedVideo?.analysis_status !== 'processing') return;
    const timer = setInterval(() => { void refreshVideoDetail(selectedId); void refreshVideos(); }, 1500);
    return () => clearInterval(timer);
  }, [selectedId, selectedVideo?.analysis_status]);

  const handleUpload = async (event: ChangeEvent<HTMLInputElement>) => {
    const input = event.currentTarget;
    const files = Array.from(input.files || []);
    if (!files.length || uploading) return;
    input.value = '';
    setUploading(true);
    setUploadResults([]);
    let firstUploadedId = '';
    let uploaded = 0;
    try {
      // Keep memory and backend load bounded; a failed file does not stop the batch.
      for (let index = 0; index < files.length; index += 1) {
        const file = files[index];
        setStatusMessage(`Uploading ${index + 1} of ${files.length}: ${file.name}`);
        const formData = new FormData();
        formData.append('file', file);
        try {
          const response = await fetch(`${API_BASE}/api/videos`, { method: 'POST', body: formData });
          if (!response.ok) {
            const text = await response.text();
            let message = text || 'Upload failed';
            try {
              const payload = JSON.parse(text);
              message = typeof payload.detail === 'string' ? payload.detail : message;
            } catch { /* Keep the original message for non-JSON errors. */ }
            throw new Error(message);
          }
          const payload = await response.json();
          if (!firstUploadedId) firstUploadedId = payload.id;
          uploaded += 1;
          setUploadResults((previous) => [...previous, { name: file.name }]);
        } catch (error) {
          setUploadResults((previous) => [...previous, {
            name: file.name,
            error: error instanceof Error ? error.message : 'Upload failed',
          }]);
        }
      }
      await refreshVideos(false);
      if (firstUploadedId) setSelectedId(firstUploadedId);
      setStatusMessage(`Uploaded ${uploaded} of ${files.length} videos. Select a video to analyze it.`);
    } finally {
      setUploading(false);
    }
  };

  const handleAnalyze = async (videoId: string, force = false) => {
    try {
      const result = await apiFetch<{ id: string; status: string; analysis_status?: string; message?: string }>(`/api/videos/${videoId}/analyze?force=${force}`, {
        method: 'POST',
      });
      setStatusMessage(result.message || 'Analysis started');
      await refreshVideoDetail(videoId);
      await refreshVideos();
    } catch (error) {
      setStatusMessage(error instanceof Error ? error.message : 'Analyze request failed');
    }
  };

  const saveAnnotations = async () => {
    if (!selectedId) return;
    try {
      await apiFetch(`/api/videos/${selectedId}/annotations`, {
        method: 'PUT',
        body: JSON.stringify({ annotations }),
      });
      setStatusMessage('Annotations saved.');
      await refreshVideoDetail(selectedId);
      await refreshVideos();
    } catch (error) {
      setStatusMessage(error instanceof Error ? error.message : 'Unable to save annotations');
    }
  };

  const addAnnotation = () => {
    const next = emptyAnnotation();
    setAnnotations((previous) => [...previous, next]);
  };

  const removeAnnotation = (index: number) => {
    setAnnotations((previous) => {
      const removed = previous[index]?.id;
      return previous.filter((_, itemIndex) => itemIndex !== index).map((item) => item.parent_activity_id === removed ? { ...item, parent_activity_id: null, source: 'manual' } : item);
    });
  };

  const updateAnnotation = (index: number, field: keyof Annotation, value: string | number | string[] | ObjectReference[] | null) => {
    setAnnotations((previous) =>
      previous.map((annotation, itemIndex) => {
        if (itemIndex !== index) return annotation;
        return { ...annotation, [field]: value, source: 'manual' };
      }),
    );
  };

  const exportDataset = async () => {
    if (!selectedId) return;
    const response = await fetch(`${API_BASE}/api/videos/${selectedId}/export?format=${exportFormat}`);
    if (!response.ok) {
      setStatusMessage('Export failed');
      return;
    }
    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = `annotation-export.${exportFormat}`;
    anchor.click();
    URL.revokeObjectURL(url);
    setStatusMessage('Dataset exported');
  };

  const deleteVideo = async (videoId: string) => {
    try {
      await apiFetch(`/api/videos/${videoId}`, { method: 'DELETE' });
      setStatusMessage('Video deleted');
      setSelectedId('');
      setSelectedVideo(null);
      setAnnotations([]);
      await refreshVideos();
    } catch (error) {
      setStatusMessage(error instanceof Error ? error.message : 'Delete failed');
    }
  };

  const detectionRows = (selectedVideo?.detections || []).flatMap((frame: { time: number; items: { id: string; class_name: string; box: number[] }[] }) =>
    frame.items.map((item) => ({ ...item, time: frame.time })))
    .filter((item: { id: string; class_name: string; time: number }) => `${item.id} ${item.class_name} ${item.time}`.toLowerCase().includes(annotationFilter.toLowerCase()));
  const pageCount = Math.max(1, Math.ceil(detectionRows.length / 50));
  const visibleDetectionPage = Math.min(detectionPage, pageCount - 1);

  const selectedVideoUrl = selectedId ? `${API_BASE}/api/videos/${selectedId}/file` : '';

  return (
    <main className="min-h-screen bg-slate-100 p-6 text-slate-900">
      <div className="mx-auto max-w-7xl space-y-6">
        <datalist id="behavior-labels">{OBJECT_AND_MOVEMENT_LABELS.map((label) => <option key={label} value={label} />)}</datalist>
        <header className="rounded-2xl bg-slate-950 p-6 text-white shadow-lg">
          <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
            <div>
              <p className="text-sm uppercase tracking-[0.2em] text-slate-300">Behavioral Analysis</p>
              <h1 className="mt-2 text-3xl font-bold">Human Interaction Analyzer</h1>
            </div>
            <div className="flex items-center gap-3">
            <label className="inline-flex cursor-pointer items-center justify-center rounded-xl bg-white px-4 py-2 text-sm font-semibold text-slate-900 shadow-sm">
              <input type="file" multiple disabled={uploading} accept="video/mp4,video/quicktime,video/webm" onChange={handleUpload} className="hidden" />
              {uploading ? 'Uploading...' : 'Upload videos'}
            </label>
            </div>
          </div>
        </header>
        {statusMessage && <p role="status" className="rounded-xl border border-slate-200 bg-white px-4 py-3 text-sm text-slate-700">{statusMessage}</p>}

        {uploadResults.length > 0 && (
          <ul aria-label="Upload results" aria-live="polite" className="space-y-1 rounded-xl border border-slate-200 bg-white px-4 py-3 text-sm">
            {uploadResults.map((result, index) => (
              <li key={index} className={result.error ? 'text-red-700' : 'text-emerald-700'}>
                {result.name}: {result.error || 'Uploaded'}
              </li>
            ))}
          </ul>
        )}

        <div className="grid gap-6 lg:grid-cols-[320px,1fr]">
          <aside className="space-y-4 rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
            <div>
              <h2 className="text-lg font-semibold">Videos</h2>
            </div>
            <div className="space-y-3">
              {videos.length === 0 && <p className="text-sm text-slate-500">No videos uploaded yet.</p>}
              {videos.map((video) => (
                <button
                  key={video.id}
                  type="button"
                  onClick={() => setSelectedId(video.id)}
                  className={`w-full rounded-xl border p-3 text-left transition ${
                    selectedId === video.id
                      ? 'border-sky-500 bg-sky-50'
                      : 'border-slate-200 bg-white hover:border-slate-300'
                  }`}
                >
                  <div className="flex items-center justify-between gap-2">
                    <span className="truncate font-medium">{video.original_name}</span>
                    <span className="rounded-full bg-slate-200 px-2 py-0.5 text-[10px] uppercase tracking-wide text-slate-700">
                      {video.analysis_status}
                    </span>
                  </div>
                  <p className="mt-1 text-xs text-slate-500">{video.duration.toFixed(1)}s • {video.width}x{video.height}</p>
                  {video.message && <p className="mt-2 text-xs text-slate-500">{video.message}</p>}
                </button>
              ))}
            </div>
          </aside>

          <section className="space-y-6">
            {selectedVideo && selectedVideo.id === selectedId ? (
              <>
                <div className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
                  <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
                    <div>
                      <p className="text-sm uppercase tracking-[0.2em] text-slate-500">Selected video</p>
                      <h2 className="text-xl font-semibold">{selectedVideo.original_name}</h2>
                    </div>
                    <div className="flex flex-wrap gap-2">
                      <button type="button" disabled={uploading || selectedVideo.analysis_status === 'processing'} onClick={() => handleAnalyze(selectedId)}
                        className="rounded-xl bg-sky-600 px-4 py-2 text-sm font-semibold text-white hover:bg-sky-500 disabled:cursor-not-allowed disabled:opacity-50">
                        {selectedVideo.analysis_status === 'processing' ? 'Analyzing...' : 'Analyze'}
                      </button>
                      {['completed', 'manual_review'].includes(selectedVideo.analysis_status) && (
                        <button type="button" className="rounded-xl border border-sky-300 px-4 py-2 text-sm font-medium text-sky-700" onClick={() => handleAnalyze(selectedId, true)}>
                          Reanalyze
                        </button>
                      )}
                      <button
                        type="button"
                        className="rounded-xl border border-slate-300 px-4 py-2 text-sm font-medium text-slate-700 hover:bg-slate-100"
                        disabled={selectedVideo.analysis_status === 'processing'}
                        onClick={saveAnnotations}
                      >
                        Save
                      </button>
                      <button
                        type="button"
                        className="rounded-xl border border-red-200 px-4 py-2 text-sm font-medium text-red-700 hover:bg-red-50"
                        onClick={() => deleteVideo(selectedId)}
                      >
                        Delete
                      </button>
                    </div>
                  </div>

                  <div role="tablist" aria-label="Selected video views" className="mb-4 flex gap-2 border-b border-slate-200">
                    <button id="review-tab" role="tab" aria-selected={videoTab === 'review'} aria-controls="review-panel" onClick={() => setVideoTab('review')} className={`rounded-t-lg px-4 py-2 text-sm font-semibold ${videoTab === 'review' ? 'bg-sky-100 text-sky-800' : 'text-slate-500'}`}>Video & editor</button>
                    <button id="annotations-tab" role="tab" aria-selected={videoTab === 'table'} aria-controls="annotations-panel" onClick={() => setVideoTab('table')} className={`rounded-t-lg px-4 py-2 text-sm font-semibold ${videoTab === 'table' ? 'bg-sky-100 text-sky-800' : 'text-slate-500'}`}>Annotations table</button>
                  </div>
                  <div id="review-panel" role="tabpanel" aria-labelledby="review-tab" hidden={videoTab !== 'review'}>
                  {selectedVideoUrl ? (
                    <DetectionOverlay key={selectedId} src={selectedVideoUrl} frames={selectedVideo.detections || []} />
                  ) : null}

                  </div>
                  <div id="annotations-panel" role="tabpanel" aria-labelledby="annotations-tab" hidden={videoTab !== 'table'}>
                    <label className="mb-3 block text-sm text-slate-600">Filter annotations
                      <input type="search" value={annotationFilter} onChange={(e) => { setAnnotationFilter(e.target.value); setDetectionPage(0); }} placeholder="Behavior, person, object, evidence or status" className="mt-1 w-full rounded-lg border border-slate-300 px-3 py-2" />
                    </label>
                    <p className="mb-3 text-xs text-slate-500">Includes current edits. Click Save to persist them. Automatic rows remain unverified until reviewed.</p>
                    <h3 className="mb-2 font-semibold">Behavior annotations ({annotations.length})</h3>
                    <div className="overflow-x-auto">
                      <table className="w-full border-collapse text-left text-sm">
                        <thead className="bg-slate-100"><tr>{['Time (s)', 'Behavior', 'People', 'Objects', 'Review', 'Evidence', 'Parent activity'].map((heading) => <th key={heading} className="border-b border-slate-200 p-3">{heading}</th>)}</tr></thead>
                        <tbody>
                          {[...annotations].filter((a) => JSON.stringify(a).toLowerCase().includes(annotationFilter.toLowerCase())).sort((a, b) => a.start_time - b.start_time).map((a, index) => <tr key={a.id || index} className="border-b border-slate-200 align-top">
                            <td className="whitespace-nowrap p-3">{a.start_time.toFixed(2)}–{a.end_time.toFixed(2)}</td>
                            <td className="p-3 font-medium">{a.behavior}<span className="mt-1 block text-xs font-normal text-slate-500">{a.source === 'local_model' ? 'Automatic suggestion' : 'Manual / saved'}</span></td>
                            <td className="p-3">{a.participants.join(', ') || 'Unknown'}</td>
                            <td className="p-3">{a.objects?.map((o) => `${o.id} (${o.class_name})`).join(', ') || '—'}</td>
                            <td className="p-3">{a.review_status.replaceAll('_', ' ')}</td>
                            <td className="min-w-[260px] p-3">{a.description || '—'}</td>
                            <td className="p-3">{a.parent_activity_id ? annotations.find((p) => p.id === a.parent_activity_id)?.behavior || a.parent_activity_id : '—'}</td>
                          </tr>)}
                        </tbody>
                      </table>
                      {!annotations.length && <p className="py-6 text-sm text-slate-500">No behavior annotations qualify yet. Detection samples are listed below; add or edit behaviors in Video & editor.</p>}
                    </div>
                    <h3 className="mb-2 mt-6 font-semibold">Detection samples ({detectionRows.length})</h3>
                    <p className="mb-3 text-xs text-slate-500">Raw model observations drawn on the video. These are unverified detections, separate from behavior suggestions.</p>
                    <div className="overflow-x-auto">
                      <table className="w-full border-collapse text-left text-sm">
                        <thead className="bg-slate-100"><tr>{['Sample time (s)', 'Track ID', 'Detected class', 'Type'].map((label) => <th key={label} className="border-b p-3">{label}</th>)}</tr></thead>
                        <tbody>{detectionRows.slice(visibleDetectionPage * 50, (visibleDetectionPage + 1) * 50).map((d: { id: string; time: number; class_name: string }, index: number) => <tr key={`${d.time}-${d.id}-${index}`} className="border-b border-slate-200"><td className="p-3">{d.time.toFixed(2)}</td><td className="p-3">{d.id}</td><td className="p-3">{d.class_name}</td><td className="p-3">{d.class_name === 'person' ? 'Person' : 'Object'}</td></tr>)}</tbody>
                      </table>
                      {!detectionRows.length && <p className="py-4 text-sm text-slate-500">No matching detection samples. Reanalyze to generate current detections.</p>}
                    </div>
                    {pageCount > 1 && <div className="mt-3 flex items-center gap-3 text-sm"><button type="button" disabled={visibleDetectionPage === 0} onClick={() => setDetectionPage(visibleDetectionPage - 1)} className="rounded border px-3 py-1 disabled:opacity-40">Previous</button><span>Page {visibleDetectionPage + 1} of {pageCount}</span><button type="button" disabled={visibleDetectionPage >= pageCount - 1} onClick={() => setDetectionPage(visibleDetectionPage + 1)} className="rounded border px-3 py-1 disabled:opacity-40">Next</button></div>}
                    <button type="button" onClick={() => setVideoTab('review')} className="mt-3 rounded-lg border border-slate-300 px-3 py-2 text-sm">Edit annotations in Video & editor</button>
                  </div>
                  {(selectedVideo.analysis_warnings || []).map((warning: string, index: number) => <p key={index} role="status" className="mt-3 text-sm text-amber-700">{warning}</p>)}
                  <div className="mt-4 flex flex-wrap gap-3 text-sm text-slate-600">
                    <span>Duration: {selectedVideo.duration.toFixed(1)}s</span>
                    <span>Resolution: {selectedVideo.width}×{selectedVideo.height}</span>
                    <span>Status: {selectedVideo.analysis_status}</span>
                  </div>
                </div>

                <div hidden={videoTab !== 'review'} className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
<fieldset disabled={selectedVideo.analysis_status === 'processing'}>
                  <div className="mb-4 flex items-center justify-between gap-2">
                    <h3 className="text-lg font-semibold">Annotations</h3>
                    <button type="button" className="rounded-xl bg-slate-900 px-3 py-2 text-sm font-medium text-white" onClick={addAnnotation}>
                      Add annotation
                    </button>
                  </div>
                  <p className="mb-4 text-sm text-slate-500">Automatic annotations are geometry-based suggestions, not confirmed actions. Review the evidence, correct the label, or mark Rejected, then Save.</p>
                  <div className="space-y-4">
                    {annotations.length === 0 ? (
                      <p className="text-sm text-slate-500">No annotations yet. Add one or run analysis.</p>
                    ) : null}
                    {annotations.map((annotation, index) => (
                      <div key={`${annotation.behavior}-${index}`} className="rounded-xl border border-slate-200 p-3">
                        <p className="mb-3 text-xs font-medium text-slate-500">{annotation.source === 'local_model' ? 'Automatic review suggestion' : 'Saved / manual annotation'} · {annotation.review_status.replaceAll('_', ' ')}</p>
                        <div className="grid gap-3 md:grid-cols-2">
                          <label className="text-sm text-slate-700">
                            Start time
                            <input
                              type="number"
                              step="0.1"
                              value={annotation.start_time}
                              onChange={(event) => updateAnnotation(index, 'start_time', Number(event.target.value))}
                              className="mt-1 w-full rounded-lg border border-slate-300 px-3 py-2"
                            />
                          </label>
                          <label className="text-sm text-slate-700">
                            End time
                            <input
                              type="number"
                              step="0.1"
                              value={annotation.end_time}
                              onChange={(event) => updateAnnotation(index, 'end_time', Number(event.target.value))}
                              className="mt-1 w-full rounded-lg border border-slate-300 px-3 py-2"
                            />
                          </label>
                          <label className="text-sm text-slate-700">
                            Behavior (choices include manual-only labels)
                            <input
                              type="text"
                              list="behavior-labels"
                              value={annotation.behavior}
                              onChange={(event) => updateAnnotation(index, 'behavior', event.target.value)}
                              className="mt-1 w-full rounded-lg border border-slate-300 px-3 py-2"
                            />
                          </label>
                          <label className="text-sm text-slate-700">
                            Review status
                            <select
                              value={annotation.review_status}
                              onChange={(event) => updateAnnotation(index, 'review_status', event.target.value)}
                              className="mt-1 w-full rounded-lg border border-slate-300 px-3 py-2"
                            >
                              <option value="needs_review">Needs review</option>
                              <option value="reviewed">Reviewed</option>
                              <option value="rejected">Rejected</option>
                            </select>
                          </label>
                        </div>
                        <label className="mt-3 block text-sm text-slate-700">
                          Participants
                          <input
                            type="text"
                            value={annotation.participants.join(', ')}
                            onChange={(event) =>
                              updateAnnotation(index, 'participants', event.target.value.split(',').map((value) => value.trim()).filter(Boolean))
                            }
                            className="mt-1 w-full rounded-lg border border-slate-300 px-3 py-2"
                          />
                        </label>
                        <div className="mt-3 space-y-2 text-sm text-slate-700">
                          <p>Objects (anonymous ID and detected/corrected class)</p>
                          {(annotation.objects || []).map((object, objectIndex) => (
                            <div key={objectIndex} className="flex gap-2">
                              <input aria-label="Object ID" value={object.id} className="w-1/3 rounded-lg border border-slate-300 px-3 py-2"
                                onChange={(e) => updateAnnotation(index, 'objects', (annotation.objects || []).map((o, j) => j === objectIndex ? { ...o, id: e.target.value } : o))} />
                              <input aria-label="Object class" value={object.class_name} className="w-1/3 rounded-lg border border-slate-300 px-3 py-2"
                                onChange={(e) => updateAnnotation(index, 'objects', (annotation.objects || []).map((o, j) => j === objectIndex ? { ...o, class_name: e.target.value } : o))} />
                              <button type="button" onClick={() => updateAnnotation(index, 'objects', (annotation.objects || []).filter((_, j) => j !== objectIndex))}>Remove object</button>
                            </div>
                          ))}
                          <button type="button" className="rounded-lg border border-slate-300 px-3 py-1" onClick={() => updateAnnotation(index, 'objects', [...(annotation.objects || []), { id: 'unknown_object', class_name: 'unknown' }])}>Add object association</button>
                          <label className="block">Parent activity
                            <select className="mt-1 w-full rounded-lg border border-slate-300 px-3 py-2" value={annotation.parent_activity_id || ''} onChange={(e) => updateAnnotation(index, 'parent_activity_id', e.target.value || null)}>
                              <option value="">No parent activity</option>
                              {annotations.filter((item) => item.id && item.id !== annotation.id).map((item) => <option key={item.id} value={item.id}>{item.behavior} · {item.start_time.toFixed(1)}–{item.end_time.toFixed(1)}s · {item.id?.slice(0, 8)}</option>)}
                            </select>
                          </label>
                          {annotations.some((item) => item.parent_activity_id === annotation.id) && <p className="text-xs text-sky-700">Sequence steps: {annotations.filter((item) => item.parent_activity_id === annotation.id).map((item) => item.behavior).join(' → ')}</p>}
                        </div>
                        <label className="mt-3 block text-sm text-slate-700">
                          Description / observable evidence
                          <textarea
                            value={annotation.description}
                            onChange={(event) => updateAnnotation(index, 'description', event.target.value)}
                            className="mt-1 h-24 w-full rounded-lg border border-slate-300 px-3 py-2"
                          />
                        </label>
                        <div className="mt-3 flex justify-end">
                          <button type="button" className="rounded-lg border border-red-200 px-3 py-2 text-sm text-red-700 hover:bg-red-50" onClick={() => removeAnnotation(index)}>
                            Remove
                          </button>
                        </div>
                      </div>
                    ))}
                  </div>
                  </fieldset>
                </div>

                <div className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
                  <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
                    <h3 className="text-lg font-semibold">Export</h3>
                    <div className="flex items-center gap-2">
                      <select value={exportFormat} onChange={(event) => setExportFormat(event.target.value as 'json' | 'jsonl')} className="rounded-lg border border-slate-300 px-3 py-2">
                        <option value="json">JSON</option>
                        <option value="jsonl">JSONL</option>
                      </select>
                      <button type="button" className="rounded-xl bg-emerald-600 px-4 py-2 text-sm font-medium text-white hover:bg-emerald-500" onClick={exportDataset}>
                        Download export
                      </button>
                    </div>
                  </div>
                </div>
              </>
            ) : (
              <div className="rounded-2xl border border-dashed border-slate-300 bg-white p-10 text-center text-slate-500 shadow-sm">
                Choose one or more videos using Upload videos. Once the upload finishes, click Analyze in the Selected video section.
              </div>
            )}
          </section>
        </div>


      </div>
    </main>
  );
}
