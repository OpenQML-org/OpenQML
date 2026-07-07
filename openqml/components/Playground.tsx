"use client";
import { useMemo, useState } from "react";
import { Op, simulate, PRESETS } from "@/lib/simulator";

const SINGLE = ["H", "X", "Y", "Z", "S", "T"];
const ROT = ["RX", "RY", "RZ"];
const SHOT_OPTS = [0, 100, 1000, 10000];

/* ---------- code export ---------- */
function toPennyLane(n: number, ops: Op[]): string {
  const g = (o: Op) => {
    if (o.g === "CNOT") return `    qml.CNOT(wires=[${o.c}, ${o.t}])`;
    if (o.theta !== undefined) return `    qml.${o.g}(${o.theta.toFixed(4)}, wires=${o.t})`;
    const m: Record<string, string> = { H: "Hadamard", X: "PauliX", Y: "PauliY", Z: "PauliZ", S: "S", T: "T" };
    return `    qml.${m[o.g]}(wires=${o.t})`;
  };
  return `import pennylane as qml

dev = qml.device("default.qubit", wires=${n})

@qml.qnode(dev)
def circuit():
${ops.length ? ops.map(g).join("\n") : "    pass"}
    return qml.probs(wires=range(${n}))

print(circuit())`;
}

function toQiskit(n: number, ops: Op[]): string {
  const g = (o: Op) => {
    if (o.g === "CNOT") return `qc.cx(${o.c}, ${o.t})`;
    if (o.theta !== undefined) return `qc.${o.g.toLowerCase()}(${o.theta.toFixed(4)}, ${o.t})`;
    return `qc.${o.g.toLowerCase()}(${o.t})`;
  };
  return `from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

qc = QuantumCircuit(${n})
${ops.map(g).join("\n")}

print(Statevector(qc).probabilities())`;
}

/* ---------- circuit diagram ---------- */
function Diagram({ n, ops }: { n: number; ops: Op[] }) {
  const colW = 40, rowH = 42, left = 44, top = 16;
  const width = left + Math.max(ops.length, 1) * colW + 16;
  const height = top + (n - 1) * rowH + 32;
  const wireY = (q: number) => top + q * rowH + 8;
  return (
    <svg className="diagram" viewBox={`0 0 ${width} ${height}`} style={{ width: "100%", maxWidth: width }} aria-label="circuit diagram">
      {Array.from({ length: n }, (_, q) => (
        <g key={q}>
          <text x={6} y={wireY(q) + 4} className="dg-label">{`q${q}`}</text>
          <line x1={left - 10} y1={wireY(q)} x2={width - 8} y2={wireY(q)} className="dg-wire" />
        </g>
      ))}
      {ops.map((o, i) => {
        const x = left + i * colW + colW / 2;
        if (o.g === "CNOT" && o.c !== undefined) {
          const y1 = wireY(o.c), y2 = wireY(o.t);
          return (
            <g key={i}>
              <line x1={x} y1={Math.min(y1, y2)} x2={x} y2={Math.max(y1, y2)} className="dg-conn" />
              <circle cx={x} cy={y1} r={4} className="dg-dot" />
              <circle cx={x} cy={y2} r={9} className="dg-oplus" />
              <line x1={x - 6} y1={y2} x2={x + 6} y2={y2} className="dg-conn" />
              <line x1={x} y1={y2 - 6} x2={x} y2={y2 + 6} className="dg-conn" />
            </g>
          );
        }
        const y = wireY(o.t);
        const rot = o.theta !== undefined;
        return (
          <g key={i}>
            <rect x={x - 14} y={y - 13} width={28} height={26} rx={6} className={rot ? "dg-box rot" : "dg-box"} />
            <text x={x} y={y + 4} className="dg-gtext">{o.g}</text>
          </g>
        );
      })}
    </svg>
  );
}

/* ---------- sampling ---------- */
function sample(probs: number[], shots: number): number[] {
  const counts = new Array(probs.length).fill(0);
  const cum: number[] = [];
  let acc = 0;
  for (const p of probs) { acc += p; cum.push(acc); }
  for (let s = 0; s < shots; s++) {
    const r = Math.random() * acc;
    let i = cum.findIndex((c) => r <= c);
    if (i < 0) i = probs.length - 1;
    counts[i]++;
  }
  return counts;
}

export default function Playground() {
  const [n, setN] = useState(2);
  const [ops, setOps] = useState<Op[]>(PRESETS["Bell pair"].ops);
  const [gate, setGate] = useState("H");
  const [target, setTarget] = useState(0);
  const [control, setControl] = useState(1);
  const [theta, setTheta] = useState(Math.PI / 2);
  const [blurb, setBlurb] = useState(PRESETS["Bell pair"].blurb);
  const [shots, setShots] = useState(0);
  const [seed, setSeed] = useState(0);
  const [exportTab, setExportTab] = useState<"none" | "pennylane" | "qiskit">("none");
  const [copied, setCopied] = useState(false);

  const result = useMemo(() => simulate(n, ops), [n, ops]);
  const N = 1 << n;
  // eslint-disable-next-line react-hooks/exhaustive-deps
  const counts = useMemo(() => (shots > 0 ? sample(result.probs, shots) : null), [result, shots, seed]);

  const addGate = () => {
    if (gate === "CNOT") {
      if (control === target) return;
      setOps([...ops, { g: "CNOT", t: target, c: control }]);
    } else if (ROT.includes(gate)) {
      setOps([...ops, { g: gate, t: target, theta }]);
    } else {
      setOps([...ops, { g: gate, t: target }]);
    }
    setBlurb("");
  };

  const setQubits = (k: number) => {
    setN(k); setOps([]); setBlurb("");
    if (target >= k) setTarget(0);
    if (control >= k) setControl(Math.min(1, k - 1));
  };

  const loadPreset = (name: string) => {
    const p = PRESETS[name];
    setN(p.n); setOps(p.ops); setBlurb(p.blurb);
  };

  const opLabel = (o: Op) =>
    o.g === "CNOT" ? `CNOT q${o.c}→q${o.t}`
    : o.theta !== undefined ? `${o.g}(${o.theta.toFixed(2)}) q${o.t}`
    : `${o.g} q${o.t}`;

  const exportCode = exportTab === "pennylane" ? toPennyLane(n, ops) : exportTab === "qiskit" ? toQiskit(n, ops) : "";
  const copyExport = async () => {
    try { await navigator.clipboard.writeText(exportCode); setCopied(true); setTimeout(() => setCopied(false), 1500); } catch {}
  };

  return (
    <div className="pg">
      <div className="pg-panel">
        <div className="pg-row">
          <span className="pg-label">Qubits</span>
          {[1, 2, 3, 4, 5].map((k) => (
            <button key={k} className={`filter ${n === k ? "active" : ""}`} onClick={() => setQubits(k)}>{k}</button>
          ))}
        </div>
        <div className="pg-row">
          <span className="pg-label">Presets</span>
          {Object.keys(PRESETS).map((p) => (
            <button key={p} className="filter" onClick={() => loadPreset(p)}>{p}</button>
          ))}
        </div>
        <div className="pg-row">
          <span className="pg-label">Gate</span>
          {[...SINGLE, ...ROT, "CNOT"].map((g) => (
            <button key={g} className={`filter ${gate === g ? "active" : ""}`} onClick={() => setGate(g)}>{g}</button>
          ))}
        </div>
        <div className="pg-row">
          <span className="pg-label">{gate === "CNOT" ? "Target" : "Qubit"}</span>
          {Array.from({ length: n }, (_, i) => (
            <button key={i} className={`filter ${target === i ? "active" : ""}`} onClick={() => setTarget(i)}>q{i}</button>
          ))}
          {gate === "CNOT" && (
            <>
              <span className="pg-label" style={{ marginLeft: 12 }}>Control</span>
              {Array.from({ length: n }, (_, i) => (
                <button key={i} className={`filter ${control === i ? "active" : ""}`} onClick={() => setControl(i)}>q{i}</button>
              ))}
            </>
          )}
          {ROT.includes(gate) && (
            <span className="pg-theta">
              θ = {theta.toFixed(2)}
              <input type="range" min={0} max={2 * Math.PI} step={0.01} value={theta}
                onChange={(e) => setTheta(parseFloat(e.target.value))} />
            </span>
          )}
        </div>
        <div className="pg-row">
          <button className="btn btn-primary" onClick={addGate} style={{ padding: "9px 20px", fontSize: 14 }}>Add gate</button>
          <button className="filter" onClick={() => { setOps(ops.slice(0, -1)); setBlurb(""); }} disabled={!ops.length}>Undo</button>
          <button className="filter" onClick={() => { setOps([]); setBlurb(""); }} disabled={!ops.length}>Clear</button>
        </div>

        <Diagram n={n} ops={ops} />

        <div className="pg-circuit">
          {ops.length === 0 ? (
            <span className="pg-empty">Circuit is empty — state is |{"0".repeat(n)}⟩.</span>
          ) : (
            ops.map((o, i) => (
              <span key={i} className="chip pg-op" title="click to remove"
                onClick={() => setOps(ops.filter((_, j) => j !== i))}>
                {opLabel(o)} ×
              </span>
            ))
          )}
        </div>
        {blurb && <p className="pg-blurb">{blurb}</p>}

        <div className="pg-row" style={{ marginTop: 18, marginBottom: 0 }}>
          <span className="pg-label">Export</span>
          <button className={`filter ${exportTab === "pennylane" ? "active" : ""}`}
            onClick={() => setExportTab(exportTab === "pennylane" ? "none" : "pennylane")}>PennyLane</button>
          <button className={`filter ${exportTab === "qiskit" ? "active" : ""}`}
            onClick={() => setExportTab(exportTab === "qiskit" ? "none" : "qiskit")}>Qiskit</button>
          {exportTab !== "none" && (
            <button className="filter" onClick={copyExport}>{copied ? "Copied" : "Copy"}</button>
          )}
        </div>
        {exportTab !== "none" && (
          <div className="code" style={{ marginTop: 12 }}>
            <pre><code>{exportCode}</code></pre>
          </div>
        )}
      </div>

      <div className="pg-panel">
        <div className="pg-row" style={{ marginBottom: 14 }}>
          <span className="pg-label">Shots</span>
          {SHOT_OPTS.map((s) => (
            <button key={s} className={`filter ${shots === s ? "active" : ""}`} onClick={() => setShots(s)}>
              {s === 0 ? "exact" : s.toLocaleString()}
            </button>
          ))}
          {shots > 0 && <button className="filter" onClick={() => setSeed(seed + 1)}>Resample</button>}
        </div>

        <div className="pg-bars">
          {Array.from({ length: N }, (_, i) => {
            const p = result.probs[i];
            const phase = result.phases[i];
            const hue = ((phase + Math.PI) / (2 * Math.PI)) * 260;
            const freq = counts ? counts[i] / shots : null;
            return (
              <div key={i} className="pg-bar-row">
                <span className="pg-basis">|{i.toString(2).padStart(n, "0")}⟩</span>
                <div className="pg-track">
                  <div className="pg-fill" style={{
                    width: `${(p * 100).toFixed(2)}%`,
                    background: p > 1e-9 ? `hsl(${hue} 62% 52%)` : "transparent",
                  }} />
                  {freq !== null && (
                    <div className="pg-sample" style={{ width: `${(freq * 100).toFixed(2)}%` }} />
                  )}
                </div>
                <span className="pg-pct">
                  {freq !== null ? `${(freq * 100).toFixed(1)}%` : `${(p * 100).toFixed(1)}%`}
                </span>
              </div>
            );
          })}
        </div>

        {shots > 0 && (
          <p className="pg-note" style={{ marginTop: 14 }}>
            Colored: exact probability. Gray: frequency over {shots.toLocaleString()} shots.
            Error scales as 1/√shots.
          </p>
        )}
        {shots === 0 && (
          <p className="pg-note">
            Exact state-vector simulation up to 5 qubits. Bar color encodes amplitude
            phase. Memory doubles per qubit.
          </p>
        )}
      </div>
    </div>
  );
}
