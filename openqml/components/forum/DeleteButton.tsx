"use client";
import { useRouter } from "next/navigation";
import { useState } from "react";

export default function DeleteButton({ threadId, postId }: { threadId: string; postId?: string }) {
  const router = useRouter();
  const [busy, setBusy] = useState(false);
  const del = async () => {
    if (!confirm("Delete permanently?")) return;
    setBusy(true);
    const url = postId ? `/api/forum/${threadId}?post=${postId}` : `/api/forum/${threadId}`;
    const r = await fetch(url, { method: "DELETE" });
    setBusy(false);
    if (r.ok) {
      if (postId) router.refresh();
      else router.push("/forum");
    }
  };
  return (
    <button className="del-btn" onClick={del} disabled={busy} title="Delete">
      delete
    </button>
  );
}
