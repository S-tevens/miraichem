import { useEffect, useState } from "react";

/**
 * Width of an element, kept in sync with its size (for responsive SVG charts). Returns a callback
 * ref, so it also works when the element only appears later (for example after data has loaded).
 */
export function useWidth<T extends HTMLElement>(fallback = 640): [(el: T | null) => void, number] {
  const [el, setEl] = useState<T | null>(null);
  const [width, setWidth] = useState(fallback);
  useEffect(() => {
    if (!el) return;
    // The content width: an element with padding (the enlarged chart sheet) must not draw under it.
    const update = () => {
      const cs = getComputedStyle(el);
      const inner = el.clientWidth - parseFloat(cs.paddingLeft || "0") - parseFloat(cs.paddingRight || "0");
      setWidth(Math.max(240, Math.round(inner > 0 ? inner : el.getBoundingClientRect().width)));
    };
    update();
    const ro = new ResizeObserver(update);
    ro.observe(el);
    return () => ro.disconnect();
  }, [el]);
  return [setEl, width];
}

export function usePrefersReducedMotion(): boolean {
  const [reduced, setReduced] = useState(false);
  useEffect(() => {
    const q = window.matchMedia("(prefers-reduced-motion: reduce)");
    setReduced(q.matches);
    const on = () => setReduced(q.matches);
    q.addEventListener("change", on);
    return () => q.removeEventListener("change", on);
  }, []);
  return reduced;
}
