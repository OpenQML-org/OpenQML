import Link from "next/link";
import Nav from "@/components/Nav";
import Footer from "@/components/Footer";
import Interference from "@/components/Interference";
import { models, algorithms, stats, Entry } from "@/lib/catalog";

function Row({ e, base }: { e: Entry; base: string }) {
  return (
    <Link href={`${base}/${e.slug}`} className="rrow">
      <div className="rrow-main">
        <span className="rrow-name">{e.name}</span>
        <span className="rrow-tag">{e.tagline}</span>
      </div>
      <div className="rrow-meta">
        <span className="chip">{e.category}</span>
        <span className="rrow-year">{e.year}</span>
        {e.code && <span className="chip mat">code</span>}
      </div>
    </Link>
  );
}

export default function Home() {
  return (
    <>
      <Nav />

      <header className="hero" style={{ padding: "72px 0 48px" }}>
        <Interference className="interference" />
        <div className="wrap">
          <h1 style={{ fontSize: "clamp(32px, 5vw, 52px)" }}>
            A registry for quantum machine learning.
          </h1>
          <p className="lede" style={{ fontSize: 18, marginTop: 20 }}>
            {stats.total} models and algorithms, each documented with mechanism, scope of
            validity, known failure modes, and references. {stats.withCode} include a
            runnable implementation. Machine-readable at{" "}
            <code style={{ fontFamily: "var(--f-mono)", fontSize: 15, background: "var(--chip-bg)", padding: "2px 7px", borderRadius: 5 }}>
              /api/catalog
            </code>.
          </p>
          <div className="statline">
            <span>{stats.models} models</span>
            <span>{stats.algorithms} algorithms</span>
            <span>{stats.withRefs} with references</span>
            <span><Link href="/playground">circuit simulator</Link></span>
            <span><Link href="/benchmarks">benchmarks</Link></span>
            <span><Link href="/docs">description standard</Link></span>
          </div>
        </div>
      </header>

      <section className="section" style={{ paddingTop: 56 }}>
        <div className="wrap">
          <div className="section-head" style={{ marginBottom: 20 }}>
            <h2 style={{ fontSize: 24 }}>Models</h2>
            <Link href="/models" className="num" style={{ color: "var(--indigo-bright)" }}>filter →</Link>
          </div>
          <div className="rlist">
            {models.map((e) => <Row key={e.slug} e={e} base="/models" />)}
          </div>

          <div className="section-head" style={{ margin: "48px 0 20px" }}>
            <h2 style={{ fontSize: 24 }}>Algorithms</h2>
            <Link href="/algorithms" className="num" style={{ color: "var(--indigo-bright)" }}>filter →</Link>
          </div>
          <div className="rlist">
            {algorithms.map((e) => <Row key={e.slug} e={e} base="/algorithms" />)}
          </div>
        </div>
      </section>

      <section className="section" style={{ paddingTop: 48, paddingBottom: 48 }}>
        <div className="wrap">
          <p style={{ maxWidth: "68ch", color: "var(--ink-soft)", fontSize: 15 }}>
            Every entry states where the method fails, with a citation where one exists.
            Simulation results here are classical and cannot show quantum advantage.
            Submissions without failure modes are rejected
            (<Link href="/submit" style={{ color: "var(--indigo-bright)" }}>submit</Link>).
          </p>
        </div>
      </section>

      <Footer />
    </>
  );
}
