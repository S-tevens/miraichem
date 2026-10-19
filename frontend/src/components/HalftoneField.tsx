import { useEffect, useRef } from "react";
import { usePrefersReducedMotion } from "../lib/hooks";

/**
 * A halftone field in the reference's teal: flowing diagonal bands rendered as dots whose size
 * follows a slowly moving wave. The pointer acts like a measurement: dots under it sharpen and
 * brighten, then relax. Paused when off-screen or when the tab is hidden; drawn once when the
 * visitor prefers reduced motion.
 */
export function HalftoneField() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const reduced = usePrefersReducedMotion();

  useEffect(() => {
    const canvas = canvasRef.current;
    const host = canvas?.parentElement;
    if (!canvas || !host) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const CELL = 7;
    let w = 0;
    let h = 0;
    let dpr = 1;
    let raf = 0;
    let visible = true;
    const pointer = { x: -999, y: -999, gain: 0 };

    const resize = () => {
      const r = host.getBoundingClientRect();
      dpr = Math.min(window.devicePixelRatio || 1, 2);
      w = r.width;
      h = r.height;
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };

    const mix = (a: number, b: number, t: number) => a + (b - a) * t;

    const draw = (t: number) => {
      ctx.fillStyle = "#08161b";
      ctx.fillRect(0, 0, w, h);
      const cols = Math.ceil(w / CELL) + 1;
      const rows = Math.ceil(h / (CELL * 0.866)) + 1;
      for (let j = 0; j < rows; j++) {
        const y = j * CELL * 0.866;
        const v = y / h;
        for (let i = 0; i < cols; i++) {
          const x = i * CELL + (j % 2 ? CELL / 2 : 0);
          const u = x / w;
          const band1 = Math.sin((u * 0.9 + v * 1.7) * 5.2 + t * 0.35 + 1.6 * Math.sin(u * 3.1 + t * 0.2));
          const band2 = Math.sin((u * 1.8 - v * 0.9) * 7.4 - t * 0.22);
          let val = 0.5 + 0.3 * band1 + 0.2 * band2;
          val *= 0.55 + 0.6 * Math.pow(1 - Math.abs(v - 0.45), 1.4);
          if (pointer.gain > 0.01) {
            const dx = x - pointer.x;
            const dy = y - pointer.y;
            val += pointer.gain * 0.55 * Math.exp(-(dx * dx + dy * dy) / (2 * 110 * 110));
          }
          val = Math.min(Math.max(val, 0), 1);
          const r = (CELL / 2) * 0.96 * Math.pow(val, 1.25);
          if (r < 0.35) continue;
          const red = mix(24, 191, val * val);
          const green = mix(66, 232, val);
          const blue = mix(74, 226, val);
          ctx.fillStyle = `rgb(${red | 0} ${green | 0} ${blue | 0})`;
          ctx.beginPath();
          ctx.arc(x, y, r, 0, Math.PI * 2);
          ctx.fill();
        }
      }
    };

    const start = performance.now();
    const frame = (now: number) => {
      pointer.gain *= 0.94;
      draw((now - start) / 1000);
      if (visible && !document.hidden) raf = requestAnimationFrame(frame);
      else raf = 0;
    };
    const kick = () => {
      if (!raf && !reduced) raf = requestAnimationFrame(frame);
    };

    const onMove = (e: PointerEvent) => {
      const r = host.getBoundingClientRect();
      pointer.x = e.clientX - r.left;
      pointer.y = e.clientY - r.top;
      pointer.gain = Math.min(1, pointer.gain + 0.25);
      kick();
    };

    resize();
    draw(0);
    kick();
    const ro = new ResizeObserver(() => {
      resize();
      draw((performance.now() - start) / 1000);
    });
    ro.observe(host);
    const io = new IntersectionObserver(([entry]) => {
      visible = entry.isIntersecting;
      if (visible) kick();
    });
    io.observe(host);
    const onVis = () => !document.hidden && kick();
    document.addEventListener("visibilitychange", onVis);
    host.addEventListener("pointermove", onMove);
    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
      io.disconnect();
      document.removeEventListener("visibilitychange", onVis);
      host.removeEventListener("pointermove", onMove);
    };
  }, [reduced]);

  return <canvas ref={canvasRef} className="halftone" aria-hidden="true" />;
}
