// OpenQML registry data. Entries follow the description standard:
// mechanism, scope of validity, known failure modes, references.
// Style: declarative, no adjectives that don't carry information.

export type Category =
  | "Variational"
  | "Kernel"
  | "Encoding"
  | "Error Mitigation"
  | "Algorithm"
  | "Gradient"
  | "QEC"
  | "Tensor Network"
  | "Generative"
  | "Temporal"
  | "Diagnostics";

export interface Ref {
  label: string;
  url?: string;
}

export interface Entry {
  slug: string;
  name: string;
  category: Category;
  tagline: string;
  description: string;
  wins: string;
  limits: string;
  tags: string[];
  maturity: "Research" | "Emerging" | "Established";
  framework: string;
  year: number;        // year the method was introduced
  qubits?: string;     // typical scale in current use
  refs?: Ref[];
  code?: string;
}

const arxiv = (id: string) => `https://arxiv.org/abs/${id}`;

export const models: Entry[] = [
  {
    slug: "vqc-classifier",
    name: "Variational Quantum Classifier",
    category: "Variational",
    tagline: "Parameterized circuit trained as a classifier.",
    description:
      "Input features are encoded into single-qubit rotations, followed by entangling layers with trainable parameters. Prediction is read from Pauli-Z expectation values. Trained with parameter-shift gradients and a classical optimizer.",
    wins: "Input dimensions up to ~10. Standard first benchmark for a new encoding or ansatz.",
    limits: "No demonstrated advantage over classical baselines on tabular data. Deep random ansätze show gradients vanishing exponentially in qubit count (McClean et al. 2018).",
    tags: ["classification", "parameter-shift", "hybrid"],
    maturity: "Established",
    framework: "PennyLane / Qiskit",
    year: 2018,
    qubits: "2–20",
    refs: [
      { label: "Farhi, Neven (2018), arXiv:1802.06002", url: arxiv("1802.06002") },
      { label: "Havlíček et al., Nature 567 (2019)", url: arxiv("1804.11326") },
    ],
    code: `import pennylane as qml
from pennylane import numpy as np

dev = qml.device("default.qubit", wires=2)

@qml.qnode(dev)
def circuit(x, weights):
    qml.AngleEmbedding(x, wires=[0, 1])
    qml.BasicEntanglerLayers(weights, wires=[0, 1])
    return qml.expval(qml.PauliZ(0))

weights = np.random.uniform(0, np.pi, (3, 2), requires_grad=True)
opt = qml.GradientDescentOptimizer(0.2)
# loss = mean((circuit(x, w) - y)**2) over the dataset`,
  },
  {
    slug: "quantum-kernel-svm",
    name: "Quantum Kernel SVM",
    category: "Kernel",
    tagline: "Kernel matrix estimated on a quantum device, SVM trained classically.",
    description:
      "The device estimates |⟨φ(x₂)|φ(x₁)⟩|² between data-encoded states. The Gram matrix goes to a classical SVM. Training is convex; the quantum part is the feature map only.",
    wins: "Feature maps that are hard to simulate classically and match the data. A constructed separation exists for discrete-log-structured data (Liu et al. 2021).",
    limits: "O(N²) circuit runs to fill the Gram matrix for N samples. On natural datasets no quantum feature map has beaten tuned classical kernels.",
    tags: ["kernel", "svm", "feature-map"],
    maturity: "Established",
    framework: "PennyLane",
    year: 2019,
    qubits: "2–20",
    refs: [
      { label: "Havlíček et al., Nature 567 (2019)", url: arxiv("1804.11326") },
      { label: "Schuld, Killoran, PRL 122 (2019)", url: arxiv("1803.07128") },
    ],
    code: `import pennylane as qml
from sklearn.svm import SVC

dev = qml.device("default.qubit", wires=2)

@qml.qnode(dev)
def kernel(x1, x2):
    qml.AngleEmbedding(x1, wires=[0, 1])
    qml.adjoint(qml.AngleEmbedding)(x2, wires=[0, 1])
    return qml.probs(wires=[0, 1])

def k(x1, x2):
    return kernel(x1, x2)[0]  # |<phi(x2)|phi(x1)>|^2

# gram matrix -> SVC(kernel="precomputed")`,
  },
  {
    slug: "data-reuploading",
    name: "Data Re-uploading Circuit",
    category: "Encoding",
    tagline: "Repeated data encoding; the model is a truncated Fourier series.",
    description:
      "Data-encoding rotations are interleaved with trainable gates over L layers. The resulting function class is a Fourier series in the input with L accessible frequencies (Schuld et al. 2021). A single qubit suffices for universal single-variable approximation.",
    wins: "Periodic and high-frequency targets. Parameter-matched: MSE 4e-4 on sin(6x) vs 0.16 for an equal-parameter MLP (see /benchmarks).",
    limits: "The function class is exactly a Fourier series. On data without periodic structure the quantum layer is bypassed by training; measured effect on text: none.",
    tags: ["encoding", "fourier", "expressivity"],
    maturity: "Emerging",
    framework: "PennyLane",
    year: 2020,
    qubits: "1–8",
    refs: [
      { label: "Pérez-Salinas et al., Quantum 4, 226 (2020)", url: arxiv("1907.02085") },
      { label: "Schuld, Sweke, Meyer, PRA 103 (2021)", url: arxiv("2008.08605") },
    ],
    code: `import pennylane as qml
from pennylane import numpy as np

dev = qml.device("default.qubit", wires=1)

@qml.qnode(dev)
def f(x, w):                      # truncated Fourier series in x
    for layer in range(len(w)):
        qml.RY(w[layer, 0], wires=0)
        qml.RZ(x, wires=0)        # each re-upload adds a frequency
        qml.RY(w[layer, 1], wires=0)
    return qml.expval(qml.PauliZ(0))`,
  },
  {
    slug: "quantum-gpt-positional",
    name: "Quantum Positional Encoding",
    category: "Encoding",
    tagline: "Transformer with quantum circuit replacing positional encoding.",
    description:
      "Attention and MLP layers stay classical; token positions are encoded by a data re-uploading circuit. Placement follows the Fourier argument: position is the one periodic quantity in a language model.",
    wins: "Long-range periodic sequence tasks. In ablations the quantum encoding outperforms sinusoidal encoding on synthetic periodic signals at matched parameter count.",
    limits: "On natural text the advantage measured is zero. A quantum feed-forward variant was tested and rejected: gradient descent routes around it.",
    tags: ["transformer", "positional-encoding", "hybrid"],
    maturity: "Research",
    framework: "CQAI",
    year: 2026,
    qubits: "4–8 (simulated)",
    refs: [{ label: "Internal ablations, CQAI (2026); see /benchmarks" }],
  },
  {
    slug: "qcnn",
    name: "Quantum Convolutional Network",
    category: "Variational",
    tagline: "Log-depth circuit with convolution and pooling structure.",
    description:
      "Alternates translationally-invariant parameterized layers with measurement-based pooling that halves the qubit count. Depth O(log n). The architecture provably avoids barren plateaus (Pesah et al. 2021).",
    wins: "Classification of quantum inputs: phase recognition, entanglement witness. Trainability is guaranteed by construction.",
    limits: "For classical images the encoding cost dominates and CNNs remain far ahead. The result is about trainability, not accuracy on classical tasks.",
    tags: ["qcnn", "phase-recognition", "trainability"],
    maturity: "Emerging",
    framework: "PennyLane / TFQ",
    year: 2019,
    qubits: "4–16",
    refs: [
      { label: "Cong, Choi, Lukin, Nat. Phys. 15 (2019)", url: arxiv("1810.03787") },
    ],
  },
  {
    slug: "quantum-gan",
    name: "Quantum GAN",
    category: "Generative",
    tagline: "Adversarial training with a circuit generator.",
    description:
      "A parameterized circuit generates samples or state amplitudes; a discriminator (classical or quantum) provides the adversarial signal.",
    wins: "Approximate loading of probability distributions into amplitudes with shallow circuits — a state-preparation subroutine for amplitude estimation pipelines.",
    limits: "GAN instability plus shot noise plus hardware noise. Sample quality is not competitive with classical generative models on any published benchmark.",
    tags: ["gan", "distribution-loading", "adversarial"],
    maturity: "Research",
    framework: "PennyLane / Qiskit",
    year: 2018,
    qubits: "2–8",
    refs: [
      { label: "Lloyd, Weedbrook, PRL 121 (2018)", url: arxiv("1804.09139") },
    ],
  },
  {
    slug: "quantum-reservoir",
    name: "Quantum Reservoir Computing",
    category: "Temporal",
    tagline: "Fixed quantum dynamics as a feature map; only the readout is trained.",
    description:
      "Input signals drive a fixed many-body quantum system. A linear readout on measured observables is trained by ridge regression. No gradients through the quantum part.",
    wins: "Temporal signals where fixed dynamics substitute for training. Sidesteps barren plateaus because nothing quantum is optimized.",
    limits: "Results depend on the arbitrary reservoir choice. Controlled comparisons against classical echo-state networks at matched readout size are scarce.",
    tags: ["reservoir", "time-series", "no-gradient"],
    maturity: "Research",
    framework: "Custom / PennyLane",
    year: 2017,
    qubits: "5–10",
    refs: [
      { label: "Fujii, Nakajima, PR Applied 8 (2017)", url: arxiv("1602.08159") },
    ],
  },
  {
    slug: "mps-model",
    name: "MPS Sequence Model",
    category: "Tensor Network",
    tagline: "Matrix product state model; runs classically.",
    description:
      "A matrix product state with bond dimension χ parameterizes the model. Entanglement is bounded by log χ. Training and inference are classical, cost linear in sequence length.",
    wins: "Sequences with bounded correlations. Also the standard classical control: if an MPS at moderate χ matches a quantum model, the quantum claim fails.",
    limits: "Cannot represent volume-law entangled states — by construction it excludes the regime where quantum advantage would live.",
    tags: ["mps", "tensor-network", "baseline"],
    maturity: "Established",
    framework: "NumPy / ITensor",
    year: 2016,
    refs: [
      { label: "Stoudenmire, Schwab, NeurIPS (2016)", url: arxiv("1605.05775") },
    ],
  },
  {
    slug: "quantum-attention",
    name: "Quantum Attention",
    category: "Kernel",
    tagline: "Attention scores from circuit state overlaps.",
    description:
      "Dot-product attention is replaced by overlaps between circuit-encoded token states, i.e. attention as a quantum kernel.",
    wins: "A controlled probe for whether quantum similarity measures change sequence modeling. Useful for negative results.",
    limits: "Parameter-matched runs on text show no improvement; trained circuits converge toward what dot-product attention computes.",
    tags: ["attention", "kernel", "ablation"],
    maturity: "Research",
    framework: "CQAI",
    year: 2026,
    qubits: "4–8 (simulated)",
    refs: [{ label: "Internal ablations, CQAI (2026)" }],
  },
  {
    slug: "vq-regressor",
    name: "Variational Quantum Regressor",
    category: "Variational",
    tagline: "Expectation value trained against a continuous target.",
    description:
      "A circuit's measured expectation is fit to a scalar target with squared loss and parameter-shift gradients.",
    wins: "Smooth low-dimensional signals, in particular periodic ones matching the circuit's Fourier spectrum.",
    limits: "Estimating an expectation to precision ε costs O(1/ε²) shots: each additional digit of precision costs 100× more measurements.",
    tags: ["regression", "expectation", "shots"],
    maturity: "Emerging",
    framework: "PennyLane",
    year: 2018,
    qubits: "1–8",
    refs: [
      { label: "Mitarai et al., PRA 98 (2018)", url: arxiv("1803.00745") },
    ],
  },
];

export const algorithms: Entry[] = [
  {
    slug: "vqe",
    name: "Variational Quantum Eigensolver",
    category: "Algorithm",
    tagline: "Ground-state energy by variational minimization.",
    description:
      "A parameterized trial state is prepared; the Hamiltonian expectation is measured term by term and minimized classically. Tolerates shallow circuits and moderate noise.",
    wins: "Small molecules (H₂, LiH, BeH₂) on current hardware, combined with error mitigation.",
    limits: "Classical methods (CCSD(T), DMRG) win at every currently reachable system size. Measurement count grows with Hamiltonian terms; deep ansätze hit barren plateaus.",
    tags: ["chemistry", "ground-state", "hybrid"],
    maturity: "Established",
    framework: "PennyLane / Qiskit",
    year: 2014,
    qubits: "2–12",
    refs: [
      { label: "Peruzzo et al., Nat. Comms 5 (2014)", url: arxiv("1304.3061") },
    ],
    code: `import pennylane as qml
from pennylane import numpy as np

symbols, coords = ["H", "H"], np.array([0., 0., -0.66, 0., 0., 0.66])
H, n_qubits = qml.qchem.molecular_hamiltonian(symbols, coords)
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev)
def energy(theta):
    qml.BasisState([1, 1, 0, 0], wires=range(n_qubits))
    qml.DoubleExcitation(theta, wires=[0, 1, 2, 3])
    return qml.expval(H)

opt = qml.GradientDescentOptimizer(0.4)
theta = np.array(0.0, requires_grad=True)
for _ in range(50):
    theta = opt.step(energy, theta)`,
  },
  {
    slug: "qaoa",
    name: "QAOA",
    category: "Algorithm",
    tagline: "Alternating-operator heuristic for combinatorial optimization.",
    description:
      "p rounds of problem and mixer Hamiltonian evolution with 2p classically tuned angles. Approximation ratio improves with p.",
    wins: "MaxCut-type problems as a hardware benchmark; analyzable structure.",
    limits: "At hardware-feasible depth, Goemans–Williamson and local search match or beat it. No practical advantage demonstrated as of 2026.",
    tags: ["optimization", "maxcut", "hybrid"],
    maturity: "Emerging",
    framework: "PennyLane",
    year: 2014,
    qubits: "4–27 demonstrated",
    refs: [
      { label: "Farhi, Goldstone, Gutmann (2014)", url: arxiv("1411.4028") },
    ],
  },
  {
    slug: "grover",
    name: "Grover Search",
    category: "Algorithm",
    tagline: "Unstructured search in O(√N) oracle calls.",
    description:
      "Amplitude amplification: each iteration applies the oracle phase flip and the diffusion operator, rotating the state toward marked items. Optimal for black-box search (Bennett et al. 1997).",
    wins: "Oracle problems without exploitable structure. Halves effective key length of symmetric ciphers: AES-128 → ~2⁶⁴ quantum operations.",
    limits: "Quadratic only: √(exponential) is still exponential, so it does not make exponential candidate spaces tractable. Requires an oracle that already decides the answer — if scoring is the hard part, Grover does not apply.",
    tags: ["search", "amplitude-amplification", "oracle"],
    maturity: "Established",
    framework: "Textbook / Qiskit",
    year: 1996,
    qubits: "2–5 demonstrated",
    refs: [
      { label: "Grover (1996), quant-ph/9605043", url: arxiv("quant-ph/9605043") },
    ],
    code: `import pennylane as qml

dev = qml.device("default.qubit", wires=2)

@qml.qnode(dev)
def grover_11():
    for w in [0, 1]: qml.Hadamard(w)     # uniform superposition
    qml.CZ(wires=[0, 1])                  # oracle: mark |11>
    for w in [0, 1]: qml.Hadamard(w)     # diffusion
    for w in [0, 1]: qml.PauliX(w)
    qml.CZ(wires=[0, 1])
    for w in [0, 1]: qml.PauliX(w)
    for w in [0, 1]: qml.Hadamard(w)
    return qml.probs(wires=[0, 1])

print(grover_11())  # [0, 0, 0, 1] -> |11> with certainty`,
  },
  {
    slug: "shor",
    name: "Shor's Algorithm",
    category: "Algorithm",
    tagline: "Integer factoring in polynomial time on fault-tolerant hardware.",
    description:
      "Order finding via phase estimation over modular exponentiation; the period yields a factor. The ECDLP variant breaks elliptic-curve keys the same way.",
    wins: "Superpolynomial speedup over the best known classical factoring (GNFS). The reason NIST post-quantum migration is underway.",
    limits: "RSA-2048 requires ~20M physical qubits and ~8 hours under surface-code assumptions (Gidney, Ekerå 2021). Current devices: <10³ physical qubits. Symmetric ciphers are weakened by Grover only, not broken.",
    tags: ["cryptography", "factoring", "fault-tolerant"],
    maturity: "Research",
    framework: "Textbook / Qiskit",
    year: 1994,
    qubits: "~4100 logical for RSA-2048",
    refs: [
      { label: "Shor (1997), quant-ph/9508027", url: arxiv("quant-ph/9508027") },
      { label: "Gidney, Ekerå, Quantum 5 (2021)", url: arxiv("1905.09749") },
    ],
  },
  {
    slug: "hhl",
    name: "HHL Linear Solver",
    category: "Algorithm",
    tagline: "Linear systems in polylog time, under strict caveats.",
    description:
      "Phase estimation extracts eigenvalues of A; controlled rotations implement A⁻¹ on the amplitudes. Runtime polylog(N) for s-sparse A with condition number κ, as O(κ²s²/ε).",
    wins: "Subroutine use where input and output remain quantum states throughout a larger algorithm.",
    limits: "Loading classical data and reading out the solution vector each generally cost Ω(N), erasing the exponential gain. Aaronson (2015) catalogs the fine print.",
    tags: ["linear-systems", "qpe", "io-bottleneck"],
    maturity: "Research",
    framework: "Textbook / Qiskit",
    year: 2009,
    refs: [
      { label: "Harrow, Hassidim, Lloyd, PRL 103 (2009)", url: arxiv("0811.3171") },
    ],
  },
  {
    slug: "qpe",
    name: "Quantum Phase Estimation",
    category: "Algorithm",
    tagline: "Eigenphase readout to n-bit precision.",
    description:
      "Controlled powers of U write the eigenphase onto an ancilla register; inverse QFT converts it to binary. Subroutine inside Shor and HHL.",
    wins: "Precise eigenvalues on fault-tolerant hardware; chemistry beyond variational accuracy.",
    limits: "Depth scales with precision, beyond NISQ coherence. Iterative variants trade depth for repetitions.",
    tags: ["eigenvalues", "qft", "fault-tolerant"],
    maturity: "Research",
    framework: "Textbook / Qiskit",
    year: 1995,
    refs: [
      { label: "Kitaev (1995), quant-ph/9511026", url: arxiv("quant-ph/9511026") },
    ],
  },
  {
    slug: "qft",
    name: "Quantum Fourier Transform",
    category: "Algorithm",
    tagline: "Fourier transform on 2ⁿ amplitudes in O(n²) gates.",
    description:
      "Hadamards and controlled phase rotations implement the DFT on the amplitude vector. Approximate versions drop small rotations for O(n log n).",
    wins: "Period finding and phase estimation over quantum data. The exponential gap versus FFT is real when amplitudes are already quantum.",
    limits: "Loading 2ⁿ classical values or reading 2ⁿ amplitudes costs Ω(2ⁿ). Not a replacement for FFT in signal processing.",
    tags: ["qft", "subroutine", "period-finding"],
    maturity: "Established",
    framework: "Textbook",
    year: 1994,
    refs: [{ label: "Nielsen, Chuang, ch. 5 (2000)" }],
  },
  {
    slug: "quantum-walk",
    name: "Quantum Walk",
    category: "Algorithm",
    tagline: "Coherent walk on graphs; quadratic spreading.",
    description:
      "Superposition over vertices evolves under a coined or continuous-time walk operator. Basis for element-distinctness and spatial-search speedups.",
    wins: "Glued-trees traversal: proven exponential separation over any classical algorithm (Childs et al. 2003). Spatial search: quadratic.",
    limits: "The exponential case is an oracle construction. Mapping practical problems onto walk structure with surviving speedup is generally unsolved.",
    tags: ["graphs", "walk", "search"],
    maturity: "Emerging",
    framework: "PennyLane / Custom",
    year: 2003,
    refs: [
      { label: "Childs et al., STOC (2003)", url: arxiv("quant-ph/0209131") },
    ],
  },
  {
    slug: "amplitude-estimation",
    name: "Amplitude Estimation",
    category: "Algorithm",
    tagline: "Monte Carlo estimation with quadratically fewer samples.",
    description:
      "Estimates amplitude a to error ε in O(1/ε) oracle calls versus O(1/ε²) classical samples, via phase estimation on the Grover operator.",
    wins: "Rigorous quadratic speedup for expectation estimation; basis of proposed risk and pricing applications.",
    limits: "Depth for realistic ε exceeds NISQ coherence by orders of magnitude. Today a laptop running plain Monte Carlo wins.",
    tags: ["monte-carlo", "finance", "estimation"],
    maturity: "Research",
    framework: "Qiskit",
    year: 2002,
    refs: [
      { label: "Brassard et al. (2002), quant-ph/0005055", url: arxiv("quant-ph/0005055") },
    ],
  },
  {
    slug: "zne",
    name: "Zero-Noise Extrapolation",
    category: "Error Mitigation",
    tagline: "Extrapolate observables to the zero-noise limit.",
    description:
      "The circuit is run at amplified noise levels (gate folding or pulse stretching); Richardson or exponential extrapolation estimates the noiseless observable. No extra qubits.",
    wins: "Recovers usable expectation values from noisy VQE/QAOA runs; standard practice on current hardware.",
    limits: "Sampling overhead grows exponentially with depth. Extrapolation amplifies statistical error and breaks if the noise model drifts between runs.",
    tags: ["error-mitigation", "nisq", "extrapolation"],
    maturity: "Established",
    framework: "PennyLane / Mitiq",
    year: 2017,
    refs: [
      { label: "Temme, Bravyi, Gambetta, PRL 119 (2017)", url: arxiv("1612.02058") },
    ],
    code: `import pennylane as qml
from pennylane.transforms import fold_global, richardson_extrapolate

noisy_dev = qml.device("default.mixed", wires=2)

@qml.transforms.mitigate_with_zne(
    scale_factors=[1.0, 2.0, 3.0],
    folding=fold_global,
    extrapolate=richardson_extrapolate,
)
@qml.qnode(noisy_dev)
def circuit(w):
    qml.RY(w, wires=0)
    qml.CNOT(wires=[0, 1])
    return qml.expval(qml.PauliZ(1))`,
  },
  {
    slug: "parameter-shift",
    name: "Parameter-Shift Rule",
    category: "Gradient",
    tagline: "Exact gradients from two shifted circuit evaluations.",
    description:
      "For gates generated by operators with eigenvalues ±1/2: ∂f/∂θ = [f(θ+π/2) − f(θ−π/2)]/2, exactly. Runs on hardware; no finite differences.",
    wins: "The standard method for hardware gradients. Unbiased under shot noise.",
    limits: "Two evaluations per parameter per step; linear in parameter count, versus one backward pass for backprop.",
    tags: ["gradients", "hardware", "training"],
    maturity: "Established",
    framework: "PennyLane",
    year: 2018,
    refs: [
      { label: "Mitarai et al., PRA 98 (2018)", url: arxiv("1803.00745") },
      { label: "Schuld et al., PRA 99 (2019)", url: arxiv("1811.11184") },
    ],
    code: `import pennylane as qml
from pennylane import numpy as np

dev = qml.device("default.qubit", wires=1)

@qml.qnode(dev, diff_method="parameter-shift")
def f(theta):
    qml.RX(theta, wires=0)
    return qml.expval(qml.PauliZ(0))

theta = np.array(0.7, requires_grad=True)
print(qml.grad(f)(theta))
# equals [f(theta + pi/2) - f(theta - pi/2)] / 2`,
  },
  {
    slug: "adjoint-diff",
    name: "Adjoint Differentiation",
    category: "Gradient",
    tagline: "All gradients in one backward sweep, simulator only.",
    description:
      "A reverse pass applies inverse gates while accumulating derivatives against the adjoint state — the state-vector analogue of backpropagation. O(P) memory, ~2× forward cost for P parameters.",
    wins: "Training simulated circuits with many parameters; orders of magnitude faster than parameter-shift there.",
    limits: "Requires the full state vector: simulator only, memory 2ⁿ complex amplitudes.",
    tags: ["gradients", "simulation", "autograd"],
    maturity: "Established",
    framework: "PennyLane Lightning / CQAI",
    year: 2020,
    refs: [
      { label: "Jones, Gacon (2020), arXiv:2009.02823", url: arxiv("2009.02823") },
    ],
  },
  {
    slug: "steane-qec",
    name: "Steane [[7,1,3]] Code",
    category: "QEC",
    tagline: "7 physical qubits, 1 logical qubit, distance 3.",
    description:
      "CSS code built from the [7,4] Hamming code. Corrects any single-qubit error; Clifford gates are transversal.",
    wins: "The pedagogical fault-tolerance construction; small enough for full logical-qubit demonstrations on current devices.",
    limits: "Distance 3 corrects one error. Practical fault tolerance uses surface codes at ~10³ physical qubits per logical qubit — the origin of the 10⁶–10⁷ qubit requirement for Shor-scale algorithms.",
    tags: ["qec", "css-code", "fault-tolerance"],
    maturity: "Emerging",
    framework: "Stim / shorAI",
    year: 1996,
    qubits: "7 physical / 1 logical",
    refs: [
      { label: "Steane (1996), quant-ph/9601029", url: arxiv("quant-ph/9601029") },
    ],
  },
  {
    slug: "barren-plateau-diag",
    name: "Barren Plateau Diagnostics",
    category: "Diagnostics",
    tagline: "Gradient-variance test for trainability.",
    description:
      "Sample the loss gradient over random initializations; variance decaying exponentially in qubit count indicates a barren plateau (McClean et al. 2018). A few hundred samples suffice for the diagnosis.",
    wins: "Cheap pre-training check; rejects untrainable ansätze before any training budget is spent.",
    limits: "Diagnosis only. Known escapes are architectural: locality, structure, layer-wise training, better initialization — and some problem classes may admit none.",
    tags: ["barren-plateau", "variance", "trainability"],
    maturity: "Emerging",
    framework: "PennyLane / CQAI",
    year: 2018,
    refs: [
      { label: "McClean et al., Nat. Comms 9 (2018)", url: arxiv("1803.11173") },
    ],
  },
];

export const allEntries = [...models, ...algorithms];
export const isModel = (e: Entry) => models.includes(e);

export const stats = {
  total: allEntries.length,
  models: models.length,
  algorithms: algorithms.length,
  withCode: allEntries.filter((e) => e.code).length,
  withRefs: allEntries.filter((e) => e.refs?.length).length,
};
