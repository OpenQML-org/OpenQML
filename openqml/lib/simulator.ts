// Minimal state-vector simulator, up to 6 qubits, zero dependencies.
// Convention: qubit 0 is the least-significant bit of the basis index.

export type Op = { g: string; t: number; c?: number; theta?: number };

type C = [number, number]; // [re, im]
type Mat2 = [C, C, C, C];  // row-major [m00, m01, m10, m11]

const SQ = 1 / Math.SQRT2;

function gateMatrix(g: string, theta = 0): Mat2 | null {
  const c = Math.cos(theta / 2), s = Math.sin(theta / 2);
  switch (g) {
    case "H": return [[SQ, 0], [SQ, 0], [SQ, 0], [-SQ, 0]];
    case "X": return [[0, 0], [1, 0], [1, 0], [0, 0]];
    case "Y": return [[0, 0], [0, -1], [0, 1], [0, 0]];
    case "Z": return [[1, 0], [0, 0], [0, 0], [-1, 0]];
    case "S": return [[1, 0], [0, 0], [0, 0], [0, 1]];
    case "T": return [[1, 0], [0, 0], [0, 0], [SQ, SQ]];
    case "RX": return [[c, 0], [0, -s], [0, -s], [c, 0]];
    case "RY": return [[c, 0], [-s, 0], [s, 0], [c, 0]];
    case "RZ": return [[c, -s], [0, 0], [0, 0], [c, s]];
    default: return null;
  }
}

export function simulate(n: number, ops: Op[]) {
  const N = 1 << n;
  const re = new Float64Array(N);
  const im = new Float64Array(N);
  re[0] = 1;

  for (const op of ops) {
    if (op.g === "CNOT" && op.c !== undefined) {
      const cb = 1 << op.c, tb = 1 << op.t;
      for (let i = 0; i < N; i++) {
        if ((i & cb) !== 0 && (i & tb) === 0) {
          const j = i | tb;
          const tr = re[i], ti = im[i];
          re[i] = re[j]; im[i] = im[j];
          re[j] = tr; im[j] = ti;
        }
      }
      continue;
    }
    const m = gateMatrix(op.g, op.theta);
    if (!m) continue;
    const [[a_r, a_i], [b_r, b_i], [c_r, c_i], [d_r, d_i]] = m;
    const tb = 1 << op.t;
    for (let i = 0; i < N; i++) {
      if ((i & tb) === 0) {
        const j = i | tb;
        const xr = re[i], xi = im[i], yr = re[j], yi = im[j];
        re[i] = a_r * xr - a_i * xi + b_r * yr - b_i * yi;
        im[i] = a_r * xi + a_i * xr + b_r * yi + b_i * yr;
        re[j] = c_r * xr - c_i * xi + d_r * yr - d_i * yi;
        im[j] = c_r * xi + c_i * xr + d_r * yi + d_i * yr;
      }
    }
  }

  const probs = new Array<number>(N);
  const phases = new Array<number>(N);
  for (let i = 0; i < N; i++) {
    probs[i] = re[i] * re[i] + im[i] * im[i];
    phases[i] = Math.atan2(im[i], re[i]);
  }
  return { probs, phases, re: Array.from(re), im: Array.from(im) };
}

export const PRESETS: Record<string, { n: number; ops: Op[]; blurb: string }> = {
  "Bell pair": {
    n: 2,
    ops: [{ g: "H", t: 0 }, { g: "CNOT", t: 1, c: 0 }],
    blurb: "Maximally entangled: |00⟩ and |11⟩ at 50% each.",
  },
  "GHZ state": {
    n: 3,
    ops: [{ g: "H", t: 0 }, { g: "CNOT", t: 1, c: 0 }, { g: "CNOT", t: 2, c: 1 }],
    blurb: "|000⟩ + |111⟩.",
  },
  "Interference": {
    n: 1,
    ops: [{ g: "H", t: 0 }, { g: "Z", t: 0 }, { g: "H", t: 0 }],
    blurb: "H·Z·H = X: the middle phase flip redirects interference, so |0⟩ ends at |1⟩.",
  },
  "Grover |11⟩": {
    n: 2,
    ops: [
      { g: "H", t: 0 }, { g: "H", t: 1 },
      // oracle: CZ marking |11⟩, built as H(1)·CNOT(0→1)·H(1)
      { g: "H", t: 1 }, { g: "CNOT", t: 1, c: 0 }, { g: "H", t: 1 },
      // diffusion
      { g: "H", t: 0 }, { g: "H", t: 1 },
      { g: "X", t: 0 }, { g: "X", t: 1 },
      { g: "H", t: 1 }, { g: "CNOT", t: 1, c: 0 }, { g: "H", t: 1 },
      { g: "X", t: 0 }, { g: "X", t: 1 },
      { g: "H", t: 0 }, { g: "H", t: 1 },
    ],
    blurb: "One Grover iteration: the oracle marks |11⟩, diffusion amplifies it from 25% to 100%.",
  },
  "Fourier basis": {
    n: 3,
    ops: [{ g: "H", t: 0 }, { g: "H", t: 1 }, { g: "H", t: 2 }],
    blurb: "Uniform superposition over all 8 basis states.",
  },
};
