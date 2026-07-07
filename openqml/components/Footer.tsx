import Link from "next/link";

export default function Footer() {
  return (
    <footer className="footer">
      <div className="wrap footer-grid">
        <div style={{ maxWidth: 260 }}>
          <div className="brand" style={{ marginBottom: 14 }}>
            <svg className="dot" viewBox="0 0 18 18" fill="none">
              <circle cx="9" cy="9" r="8" stroke="#2b2d6e" strokeWidth="1.2" />
              <circle cx="9" cy="9" r="3.4" fill="#4a4de8" />
            </svg>
            <span>OpenQML</span>
          </div>
          <p style={{ fontSize: 14, color: "var(--ink-soft)" }}>
            An open registry for quantum machine learning. Maintained by QcatX.
          </p>
        </div>
        <div>
          <h5>Catalog</h5>
          <Link href="/models">Models</Link>
          <Link href="/algorithms">Algorithms</Link>
          <Link href="/benchmarks">Benchmarks</Link>
          <Link href="/docs">The QML Standard</Link>
        </div>
        <div>
          <h5>Project</h5>
          <Link href="/forum">Forum</Link>
          <Link href="/playground">Playground</Link>
          <Link href="/submit">Submit an entry</Link>
          <Link href="/about">About</Link>
          <Link href="/about#roadmap">Roadmap</Link>
        </div>
        <div>
          <h5>Ecosystem</h5>
          <a href="https://pennylane.ai" target="_blank" rel="noreferrer">PennyLane</a>
          <a href="https://qiskit.org" target="_blank" rel="noreferrer">Qiskit</a>
        </div>
      </div>
      <div className="wrap">
        <p className="fine">
          Simulation results on this site are classical. © {new Date().getFullYear()} QcatX · v0.7
        </p>
      </div>
    </footer>
  );
}
