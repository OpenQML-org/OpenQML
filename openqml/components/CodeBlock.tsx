"use client";
import { useState } from "react";

export default function CodeBlock({ code, label = "PennyLane" }: { code: string; label?: string }) {
  const [copied, setCopied] = useState(false);
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(code);
      setCopied(true);
      setTimeout(() => setCopied(false), 1600);
    } catch {}
  };
  return (
    <div className="code">
      <div className="code-head">
        <span>{label}</span>
        <button onClick={copy} className="code-copy">{copied ? "Copied" : "Copy"}</button>
      </div>
      <pre><code>{code}</code></pre>
    </div>
  );
}
