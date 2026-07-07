import Link from "next/link";
import { notFound } from "next/navigation";
import Nav from "@/components/Nav";
import Footer from "@/components/Footer";
import Composer from "@/components/forum/Composer";
import VoteButton from "@/components/forum/VoteButton";
import DeleteButton from "@/components/forum/DeleteButton";
import { getThread, votesFor } from "@/lib/forum";
import { getSession } from "@/lib/auth";
import { timeAgo } from "@/lib/time";

export const dynamic = "force-dynamic";

export default async function ThreadPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const [data, session] = await Promise.all([getThread(id), getSession()]);
  if (!data) notFound();
  const { thread, posts } = data;
  const votes = await votesFor([thread.id, ...posts.map((p) => p.id)], session?.uid || "");
  return (
    <>
      <Nav />
      <section className="detail" style={{ paddingTop: 56 }}>
        <div className="wrap" style={{ maxWidth: 860 }}>
          <Link href="/forum" className="back">← Forum</Link>
          <h1 style={{ fontSize: "clamp(26px,4vw,38px)" }}>{thread.title}</h1>
          <p className="thread-meta" style={{ marginTop: 10, display: "flex", gap: 12, alignItems: "center" }}>
            {thread.author} · {timeAgo(thread.createdAt)} · {posts.length} {posts.length === 1 ? "reply" : "replies"}
            <VoteButton itemId={thread.id} count={votes[thread.id]?.count || 0} voted={votes[thread.id]?.voted || false} />
            {session?.uid === thread.userId && <DeleteButton threadId={thread.id} />}
          </p>

          <div className="post op">
            <p className="post-body">{thread.body}</p>
          </div>

          {posts.map((p) => (
            <div key={p.id} className="post">
              <div className="post-head">
                <span className="post-author">{p.author}</span>
                <span style={{ display: "flex", gap: 10, alignItems: "center" }}>
                  <VoteButton itemId={p.id} count={votes[p.id]?.count || 0} voted={votes[p.id]?.voted || false} />
                  {session?.uid === p.userId && <DeleteButton threadId={thread.id} postId={p.id} />}
                  <span className="thread-meta">{timeAgo(p.createdAt)}</span>
                </span>
              </div>
              <p className="post-body">{p.body}</p>
            </div>
          ))}

          <div style={{ marginTop: 28 }}>
            <Composer mode="reply" threadId={thread.id} />
          </div>
        </div>
      </section>
      <Footer />
    </>
  );
}
