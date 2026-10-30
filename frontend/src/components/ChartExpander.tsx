import { useEffect } from "react";
import "./ChartExpander.css";

/**
 * Gives every chart an expand button in its top-right corner.
 *
 * Charts are plain elements with a few known classes. Rather than wrapping each chart (and
 * remounting it), this watches the page and decorates any chart container that has an SVG. Expanding
 * turns the same element into a full-window sheet; the charts measure their own width, so they
 * redraw larger by themselves. Escape, the button or a click outside closes it.
 */
const SELECTOR = ".chart, .strip, .lih__chart";
const OPEN = "is-expanded";

const EXPAND_ICON =
  '<svg viewBox="0 0 24 24" width="16" height="16" aria-hidden="true"><path d="M14 4h6v6M10 20H4v-6M20 4l-7 7M4 20l7-7" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg>';
const CLOSE_ICON =
  '<svg viewBox="0 0 24 24" width="16" height="16" aria-hidden="true"><path d="M6 6l12 12M18 6L6 18" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/></svg>';

function setOpen(el: HTMLElement, open: boolean) {
  const button = el.querySelector<HTMLButtonElement>(":scope > .expand-btn");
  const parent = el.parentElement;
  if (open) {
    // The sheet leaves the page flow, so keep the space it used to take.
    if (parent) parent.style.minHeight = `${el.offsetHeight}px`;
    el.classList.add(OPEN);
    document.body.classList.add("has-expanded");
  } else {
    el.classList.remove(OPEN);
    document.body.classList.remove("has-expanded");
    if (parent) parent.style.minHeight = "";
  }
  if (button) {
    button.innerHTML = open ? CLOSE_ICON : EXPAND_ICON;
    button.setAttribute("aria-label", open ? "Close the enlarged chart" : "Enlarge this chart");
    button.title = open ? "Close (Esc)" : "Enlarge";
  }
}

function decorate(root: ParentNode) {
  root.querySelectorAll<HTMLElement>(SELECTOR).forEach((el) => {
    if (el.dataset.expandable || !el.querySelector("svg")) return;
    el.dataset.expandable = "1";
    const button = document.createElement("button");
    button.type = "button";
    button.className = "expand-btn";
    button.innerHTML = EXPAND_ICON;
    button.setAttribute("aria-label", "Enlarge this chart");
    button.title = "Enlarge";
    button.addEventListener("click", (e) => {
      // Swapping the icon detaches the element that was clicked, so the page-level "click outside"
      // check below would read this click as outside the sheet and close it again at once.
      e.stopPropagation();
      setOpen(el, !el.classList.contains(OPEN));
    });
    el.appendChild(button);
  });
}

export function ChartExpander() {
  useEffect(() => {
    decorate(document.body);
    let queued = false;
    const watch = new MutationObserver(() => {
      if (queued) return;
      queued = true;
      window.setTimeout(() => {
        queued = false;
        decorate(document.body);
      }, 60);
    });
    watch.observe(document.body, { childList: true, subtree: true });

    const close = () => document.querySelectorAll<HTMLElement>(`.${OPEN}`).forEach((el) => setOpen(el, false));
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && close();
    const onClick = (e: MouseEvent) => {
      // A click on the dimmed area (the backdrop is the body pseudo-layer) is outside any sheet.
      const inside = e.composedPath().some((n) => n instanceof HTMLElement && n.classList.contains(OPEN));
      if (document.body.classList.contains("has-expanded") && !inside) close();
    };
    document.addEventListener("keydown", onKey);
    document.addEventListener("click", onClick);
    return () => {
      watch.disconnect();
      document.removeEventListener("keydown", onKey);
      document.removeEventListener("click", onClick);
      close();
    };
  }, []);
  return null;
}
