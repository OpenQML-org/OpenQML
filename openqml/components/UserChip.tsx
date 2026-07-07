"use client";
import { useEffect, useState } from "react";

interface Me { name: string; avatar: string | null }

export default function UserChip() {
  const [me, setMe] = useState<Me | null>(null);
  useEffect(() => {
    fetch("/api/auth/me").then((r) => (r.ok ? r.json() : null)).then(setMe).catch(() => {});
  }, []);
  if (!me) return null;
  return (
    <span className="user-chip">
      {me.avatar ? (
        // eslint-disable-next-line @next/next/no-img-element
        <img src={me.avatar} alt="" width={20} height={20} style={{ borderRadius: "50%" }} />
      ) : (
        <span className="user-dot">{me.name.slice(0, 1).toUpperCase()}</span>
      )}
      <span className="user-name">{me.name}</span>
      <a href="/api/auth/logout" title="Sign out" className="user-out">↦</a>
    </span>
  );
}
