/** Fixed-template patch tracking. Correlation is a heuristic, not a probability. */
export type GrayFrame = { width: number; height: number; data: Uint8Array };
export type Box = { x: number; y: number; width: number; height: number };
export type Track = {
  box: Box | null;
  score: number;
  status: 'tracking' | 'lost' | 'unselected';
  dx: number | null;
  dy: number | null;
};
const N = 16;
function sample(frame: GrayFrame, box: Box) {
  if (
    box.x < 0 ||
    box.y < 0 ||
    box.x + box.width > frame.width ||
    box.y + box.height > frame.height
  )
    return null;
  const out = new Float32Array(N * N);
  let mean = 0;
  for (let y = 0; y < N; y++)
    for (let x = 0; x < N; x++) {
      const xx = Math.min(
        frame.width - 1,
        Math.floor(box.x + ((x + 0.5) * box.width) / N),
      );
      const yy = Math.min(
        frame.height - 1,
        Math.floor(box.y + ((y + 0.5) * box.height) / N),
      );
      const i = y * N + x;
      out[i] = frame.data[yy * frame.width + xx];
      mean += out[i];
    }
  mean /= out.length;
  let energy = 0;
  for (let i = 0; i < out.length; i++) {
    out[i] -= mean;
    energy += out[i] * out[i];
  }
  if (energy / out.length < 36) return null;
  const norm = Math.sqrt(energy);
  for (let i = 0; i < out.length; i++) out[i] /= norm;
  return out;
}
export class PatchTracker {
  template: Float32Array | null = null;
  box: Box | null = null;
  origin: Box | null = null;
  lost = false;
  select(frame: GrayFrame, box: Box) {
    this.clear();
    if (
      box.width < 10 ||
      box.height < 10 ||
      box.width > 120 ||
      box.height > 120
    )
      return false;
    const template = sample(frame, box);
    if (!template) return false;
    this.template = template;
    this.box = { ...box };
    this.origin = { ...box };
    return true;
  }
  clear() {
    this.template = null;
    this.box = null;
    this.origin = null;
    this.lost = false;
  }
  update(frame: GrayFrame): Track {
    if (!this.template || !this.box || !this.origin)
      return { box: null, score: 0, status: 'unselected', dx: null, dy: null };
    if (this.lost)
      return { box: null, score: 0, status: 'lost', dx: null, dy: null };
    const prev = this.box,
      origin = this.origin,
      template = this.template;
    const candidates: { box: Box; score: number }[] = [];
    for (const scale of [1, 0.9, 1.1]) {
      const w = prev.width * scale,
        h = prev.height * scale;
      if (w < origin.width * 0.5 || w > origin.width * 2 || h < 10) continue;
      for (let dy = -24; dy <= 24; dy += 2)
        for (let dx = -24; dx <= 24; dx += 2) {
          const box = {
            x: prev.x + prev.width / 2 + dx - w / 2,
            y: prev.y + prev.height / 2 + dy - h / 2,
            width: w,
            height: h,
          };
          const values = sample(frame, box);
          if (!values) continue;
          let score = 0;
          for (let i = 0; i < values.length; i++)
            score += values[i] * template[i];
          candidates.push({ box, score });
        }
    }
    candidates.sort((a, b) => b.score - a.score);
    const best = candidates[0];
    const rival =
      best &&
      candidates.find(
        (c) =>
          Math.hypot(
            c.box.x + c.box.width / 2 - best.box.x - best.box.width / 2,
            c.box.y + c.box.height / 2 - best.box.y - best.box.height / 2,
          ) > Math.max(8, Math.min(best.box.width, best.box.height) * 0.75),
      );
    if (
      !best ||
      best.score < 0.65 ||
      (rival && best.score - rival.score < 0.06)
    ) {
      this.lost = true;
      return {
        box: null,
        score: best?.score || 0,
        status: 'lost',
        dx: null,
        dy: null,
      };
    }
    this.box = best.box;
    return {
      box: { ...this.box },
      score: best.score,
      status: 'tracking',
      dx: this.box.x + this.box.width / 2 - origin.x - origin.width / 2,
      dy: this.box.y + this.box.height / 2 - origin.y - origin.height / 2,
    };
  }
}
