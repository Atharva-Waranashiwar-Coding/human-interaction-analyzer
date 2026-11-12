'use client';

import { useState } from 'react';

type Detection = { id: string; class_name: string; box: number[] };
type Frame = { time: number; items: Detection[] };

export default function DetectionOverlay({ src, frames }: { src: string; frames: Frame[] }) {
  const [time, setTime] = useState(0);
  const [ratio, setRatio] = useState(1);
  const [visible, setVisible] = useState(true);
  let current: Frame | undefined;
  for (const frame of frames) {
    if (frame.time <= time + .001) current = frame;
    else break;
  }
  const interval = frames.length > 1 ? Math.max(...frames.slice(1).map((f, i) => f.time - frames[i].time)) : .5;
  if (current && time - current.time > interval + .1) current = undefined;
  // Inference coordinates describe the square letterboxed input. Use the actual
  // browser video dimensions so display rotation/aspect ratio also match.
  const w = Math.min(1, ratio), h = Math.min(1, 1 / ratio);
  const boxes = (current?.items || []).map((item) => {
    const [x1, y1, x2, y2] = item.box;
    const clamp = (v: number) => Math.max(0, Math.min(1000, v));
    return { ...item, x1: clamp((x1 - (1 - w) / 2) / w * 1000), x2: clamp((x2 - (1 - w) / 2) / w * 1000),
      y1: clamp((y1 - (1 - h) / 2) / h * 1000), y2: clamp((y2 - (1 - h) / 2) / h * 1000) };
  });
  return (
    <div>
      <div className="relative overflow-hidden rounded-xl bg-black">
        <video src={src} controls className="block w-full" onTimeUpdate={(e) => setTime(e.currentTarget.currentTime)}
          onSeeked={(e) => setTime(e.currentTarget.currentTime)}
          onLoadedMetadata={(e) => { setRatio(e.currentTarget.videoWidth / e.currentTarget.videoHeight || 1); setTime(0); }} />
        {visible && <svg aria-label="Sampled detections" viewBox="0 0 1000 1000" preserveAspectRatio="none" className="pointer-events-none absolute inset-0 h-full w-full">
          {boxes.map((b) => <g key={b.id} fill="none" stroke={b.class_name === 'person' ? '#38bdf8' : '#fbbf24'}>
            <rect x={b.x1} y={b.y1} width={b.x2-b.x1} height={b.y2-b.y1} strokeWidth="3" vectorEffect="non-scaling-stroke" />
            <text x={b.x1 + 4} y={Math.max(18,b.y1 + 20)} fill={b.class_name === 'person' ? '#38bdf8' : '#fbbf24'} stroke="black" strokeWidth=".5" fontSize="20">{b.id} · {b.class_name}</text>
          </g>)}
        </svg>}
      </div>
      <label className="mt-2 flex items-center gap-2 text-sm text-slate-600"><input type="checkbox" checked={visible} onChange={(e) => setVisible(e.target.checked)} />Show sampled person/object detections</label>
      <p className="mt-1 text-xs text-slate-500">{current ? `Detection sample ${current.time.toFixed(1)}s: ${current.items.map((d) => `${d.id} (${d.class_name})`).join(', ') || 'no reliable tracks'}` : 'No detection sample at this time. Analyze the video to generate tracks.'}</p>
    </div>
  );
}
