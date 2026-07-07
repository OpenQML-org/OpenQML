import Nav from "@/components/Nav";

export default function ForumLoading() {
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
          <div className="rlist">
            {Array.from({ length: 5 }, (_, i) => (
              <div key={i} className="rrow" style={{ pointerEvents: "none" }}>
                <div className="rrow-main" style={{ flexDirection: "column", gap: 8, alignItems: "flex-start", width: "100%" }}>
                  <span className="skel" style={{ width: `${52 - i * 6}%`, height: 15 }} />
                  <span className="skel" style={{ width: 120, height: 11 }} />
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>
    </>
  );
}
