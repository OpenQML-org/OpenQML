import Nav from "@/components/Nav";
import Footer from "@/components/Footer";

export const metadata = { title: "Benchmarks — OpenQML" };

// Parameter-matched quantum vs classical results. The point of this page is the
// third row as much as the first two: publishing where quantum does NOT win.
const ROWS = [
  { task: "Fit sin(6x)", metric: "MSE", quantum: 0.0004, classical: 0.16, verdict: "Quantum", win: true },
  { task: "Fit sin(9x)", metric: "MSE", quantum: 0.0009, classical: 0.24, verdict: "Quantum", win: true },
  { task: "Language modeling (text)", metric: "Loss delta", quantum: 0, classical: 0, verdict: "No difference", win: false },
];

function Bar({ v, max, color }: { v: number; max: number; color: string }) {
  const w = Math.max((v / max) * 100, v > 0 ? 1.2 : 0);
  return (
    <div className="bench-track">
      <div className="bench-fill" style={{ width: `${w}%`, background: color }} />
      <span className="bench-val">{v === 0 ? "±0" : v}</span>
    </div>
  );
}

export default function BenchmarksPage() {
  return (
    <>
      <Nav />
      <header className="page-head">
        <div className="wrap">
          <h1>Benchmarks</h1>
          <p>
            Setup: data re-uploading circuit vs MLP at equal parameter count, same
            optimizer and budget, simulated (default.qubit). Values are final test MSE.
          </p>
        </div>
      </header>

      <section className="section" style={{ borderTop: "none", paddingTop: 40 }}>
        <div className="wrap">
          <div className="bench">
            <div className="bench-head">
              <span>Task</span>
              <span>Quantum circuit</span>
              <span>Classical MLP</span>
              <span>Verdict</span>
            </div>
            {ROWS.map((r) => (
              <div key={r.task} className="bench-row">
                <span className="bench-task">{r.task}<em>{r.metric}</em></span>
                <Bar v={r.quantum} max={0.25} color="var(--indigo-bright)" />
                <Bar v={r.classical} max={0.25} color="var(--ink-faint)" />
                <span className={`chip ${r.win ? "mat" : ""}`}>{r.verdict}</span>
              </div>
            ))}
          </div>

          <div className="honesty" style={{ marginTop: 48 }}>
            <div className="tag">Notes</div>
            <p>
              A data re-uploading circuit is a truncated Fourier series (Schuld et al. 2021),
              which explains both rows: the function class matches sin(kx) exactly and
              contributes nothing on text, where training bypasses the quantum layer.
              Negative results are in scope. All rows are classical simulations; they
              establish nothing about hardware advantage.
            </p>
          </div>
        </div>
      </section>

      <Footer />
    </>
  );
}
