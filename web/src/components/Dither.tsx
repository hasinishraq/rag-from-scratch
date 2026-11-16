import { useEffect, useRef } from "react";

import { cn } from "@/lib/utils";

/**
 * 1-bit ordered dithering (Bayer 8x8) on a canvas. A field function gives a tone from 0 (paper)
 * to 1 (ink) for each cell; the Bayer threshold turns tone into on/off pixels. Colours come from
 * the theme tokens, so dark mode inverts it. Animation pauses off-screen and is disabled for
 * reduced motion (one still frame is drawn instead).
 */

// Classic 8x8 Bayer matrix, normalised to thresholds in (0, 1).
const BAYER = [
  0, 32, 8, 40, 2, 34, 10, 42, 48, 16, 56, 24, 50, 18, 58, 26, 12, 44, 4, 36, 14, 46, 6, 38, 60, 28,
  52, 20, 62, 30, 54, 22, 3, 35, 11, 43, 1, 33, 9, 41, 51, 19, 59, 27, 49, 17, 57, 25, 15, 47, 7, 39,
  13, 45, 5, 37, 63, 31, 55, 23, 61, 29, 53, 21,
].map((v) => (v + 0.5) / 64);

export type Field = (x: number, y: number, t: number, w: number, h: number) => number;

/** A sphere lit by a light that orbits slowly; the shadow side falls off to paper. */
export const orbField: Field = (x, y, t, w, h) => {
  const r = Math.min(w, h) * 0.42;
  const dx = (x - w / 2) / r;
  const dy = (y - h / 2) / r;
  const d2 = dx * dx + dy * dy;
  if (d2 > 1) {
    // A faint dithered halo just outside the rim.
    const rim = Math.sqrt(d2) - 1;
    return rim < 0.12 ? 0.18 * (1 - rim / 0.12) : 0;
  }
  const dz = Math.sqrt(1 - d2);
  const a = t * 0.35;
  const lx = Math.cos(a) * 0.75;
  const ly = -0.45 + Math.sin(a * 0.7) * 0.2;
  const lz = Math.sin(a) * 0.55 + 0.45;
  const len = Math.hypot(lx, ly, lz);
  const lambert = Math.max(0, (dx * lx + dy * ly + dz * lz) / len);
  const spec = lambert ** 18;
  // Capped below 1 so even the brightest side keeps a visible dot texture, like a halftone.
  return Math.min(0.9, 0.05 + lambert * 0.78 + spec * 0.3);
};

/** Soft interference waves, used as a quiet texture. */
export const wavesField: Field = (x, y, t, w, h) => {
  const u = x / w;
  const v = y / h;
  const s =
    Math.sin(u * 9 + t * 0.9) + Math.sin(v * 7 - t * 0.6) + Math.sin((u + v) * 6 + t * 0.4);
  return Math.max(0, Math.min(1, 0.5 + s / 6 - v * 0.25));
};

/** Density rises with `progress` from left to right; used while an answer is being prepared. */
export function revealField(progress: () => number): Field {
  return (x, y, t, w) => {
    const edge = progress() * 1.2 - 0.1;
    const u = x / w;
    const ripple = Math.sin(y * 0.35 + t * 4) * 0.03;
    const d = edge - u + ripple;
    return d > 0 ? Math.min(1, 0.35 + d * 2.2) : Math.max(0, 0.12 + d * 3);
  };
}

function readColor(el: Element, name: string, fallback: string): [number, number, number] {
  const raw = getComputedStyle(el).getPropertyValue(name).trim() || fallback;
  const hex = raw.replace("#", "");
  const full = hex.length === 3 ? hex.split("").map((c) => c + c).join("") : hex;
  const n = Number.parseInt(full, 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

export function Dither({
  field,
  cell = 3,
  fps = 24,
  className,
  label,
  animate = true,
}: {
  field: Field;
  cell?: number;
  fps?: number;
  className?: string;
  label?: string;
  animate?: boolean;
}) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const fieldRef = useRef(field);
  fieldRef.current = field;

  useEffect(() => {
    const el = canvas.current;
    if (!el) return;
    const ctx = el.getContext("2d");
    if (!ctx) return;
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    let visible = true;
    let raf = 0;
    let last = 0;
    let image: ImageData | null = null;
    let ink: [number, number, number] = [0, 0, 0];
    let paper: [number, number, number] = [255, 255, 255];
    const start = performance.now();

    const resize = () => {
      const rect = el.getBoundingClientRect();
      const w = Math.max(1, Math.round(rect.width / cell));
      const h = Math.max(1, Math.round(rect.height / cell));
      if (el.width !== w || el.height !== h) {
        el.width = w;
        el.height = h;
        image = ctx.createImageData(w, h);
      }
      ink = readColor(document.documentElement, "--ink", "#000000");
      paper = readColor(document.documentElement, "--paper", "#ffffff");
    };

    const draw = (now: number) => {
      if (!image) resize();
      if (!image) return;
      const t = (now - start) / 1000;
      const { width: w, height: h, data } = image;
      const f = fieldRef.current;
      for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
          const on = f(x, y, t, w, h) > BAYER[(y & 7) * 8 + (x & 7)];
          const c = on ? ink : paper;
          const i = (y * w + x) * 4;
          data[i] = c[0];
          data[i + 1] = c[1];
          data[i + 2] = c[2];
          data[i + 3] = 255;
        }
      }
      ctx.putImageData(image, 0, 0);
    };

    const loop = (now: number) => {
      raf = requestAnimationFrame(loop);
      if (!visible || now - last < 1000 / fps) return;
      last = now;
      draw(now);
    };

    resize();
    draw(start);
    const ro = new ResizeObserver(() => {
      resize();
      draw(performance.now());
    });
    ro.observe(el);
    const io = new IntersectionObserver(([entry]) => {
      visible = entry?.isIntersecting ?? true;
    });
    io.observe(el);
    // Re-read the theme colours when the theme switches.
    const mo = new MutationObserver(() => {
      resize();
      draw(performance.now());
    });
    mo.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
    if (animate && !reduce) raf = requestAnimationFrame(loop);

    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
      io.disconnect();
      mo.disconnect();
    };
  }, [cell, fps, animate]);

  return (
    <canvas
      ref={canvas}
      role={label ? "img" : undefined}
      aria-label={label}
      aria-hidden={label ? undefined : true}
      className={cn("block h-full w-full [image-rendering:pixelated]", className)}
    />
  );
}

/** A static Bayer dot texture for CSS backgrounds (used for uncited relevance bars). */
export function bayerTexture(level: number): string {
  const size = 8;
  const rects: string[] = [];
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      if (level > BAYER[y * 8 + x]) rects.push(`<rect x="${x}" y="${y}" width="1" height="1"/>`);
    }
  }
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${size}" height="${size}" shape-rendering="crispEdges">${rects.join("")}</svg>`;
  return `url("data:image/svg+xml,${encodeURIComponent(svg)}")`;
}
