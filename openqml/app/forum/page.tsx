import Link from "next/link";
import Nav from "@/components/Nav";
import Footer from "@/components/Footer";
import Composer from "@/components/forum/Composer";
import VoteButton from "@/components/forum/VoteButton";
import { listThreads, votesFor } from "@/lib/forum";
import { getSession } from "@/lib/auth";
import { timeAgo } from "@/lib/time";

export const dynamic = "force-dynamic";
export const metadata = { title: "Forum — OpenQML" };

export default async function ForumPage() {
  const [threads, session] = await Promise.all([listThreads(), getSession()]);
  const votes = await votesFor(threads.map((t) => t.id), session?.uid || "");
  return (
    <>
      <Nav />
      <header className="page-head">
        <div className="wrap">
          <h1>Forum</h1>
          <p>Threads on methods, benchmarks, and registry entries.</p>
        </div>
      </header>
      <section className="section" style={{ borderTop: "none", paddingTop: 32 }}>
        <div className="wrap" style={{ maxWidth: 860 }}>
          <div style={{ marginBottom: 22 }}>
            <Composer mode="thread" />
          </div>
          <div className="rlist">
            {threads.map((t) => (
              <Link key={t.id} href={`/forum/${t.id}`} className="rrow">
                <div className="rrow-main" style={{ flexDirection: "column", gap: 3, alignItems: "flex-start" }}>
                  <span className="rrow-name" style={{ whiteSpace: "normal" }}>{t.title}</span>
                  <span className="thread-meta">{t.author} · {timeAgo(t.createdAt)}</span>
                </div>
                <div className="rrow-meta">
                  <VoteButton itemId={t.id} count={votes[t.id]?.count || 0} voted={votes[t.id]?.voted || false} />
                  <span className="chip">{t.replies} {t.replies === 1 ? "reply" : "replies"}</span>
                  <span className="rrow-year" style={{ minWidth: 60 }}>{timeAgo(t.lastActivity)}</span>
                </div>
              </Link>
            ))}
            {threads.length === 0 && (
              <div style={{ padding: 24, color: "var(--ink-faint)", fontSize: 14 }}>No threads yet.</div>
            )}
          </div>
        </div>
      </section>
      <Footer />
    </>
  );
}
