"use client";
import { useEffect } from "react";

// Observes [data-reveal] elements and reveals them as they enter the viewport.
// Mounted from app/template.tsx so it re-runs on every route change.
export default function RevealObserver() {
  useEffect(() => {
    const els = Array.from(document.querySelectorAll<HTMLElement>("[data-reveal]:not(.revealed)"));
    if (!("IntersectionObserver" in window) || matchMedia("(prefers-reduced-motion: reduce)").matches) {
      els.forEach((el) => el.classList.add("revealed"));
      return;
    }
    const io = new IntersectionObserver(
      (entries) => {
        for (const en of entries) {
          if (en.isIntersecting) {
            const el = en.target as HTMLElement;
            // Stagger siblings slightly for a cascade effect
            const idx = Number(el.dataset.revealIdx || 0);
            el.style.transitionDelay = `${Math.min(idx * 60, 360)}ms`;
            el.classList.add("revealed");
            io.unobserve(el);
          }
        }
      },
      { rootMargin: "0px 0px -8% 0px", threshold: 0.08 }
    );
    els.forEach((el, i) => {
      if (el.dataset.revealIdx === undefined) {
        // index within its parent for stagger
        const sibs = el.parentElement ? Array.from(el.parentElement.children).filter((c) => (c as HTMLElement).dataset?.reveal !== undefined) : [];
        el.dataset.revealIdx = String(Math.max(0, sibs.indexOf(el)));
      }
      io.observe(el);
    });
    return () => io.disconnect();
  }, []);
  return null;
}
