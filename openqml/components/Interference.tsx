// The signature element: two overlapping circular wave sources producing an
// interference pattern — a literal picture of superposition, rendered as hairline
// arcs rather than the usual glowing-particle cliché.
export default function Interference({ className = "" }: { className?: string }) {
  const arcs = [];
  const cx1 = 210, cy1 = 250, cx2 = 320, cy2 = 250;
  for (let r = 16; r < 300; r += 16) {
    arcs.push(
      <circle key={`a${r}`} cx={cx1} cy={cy1} r={r} fill="none"
        stroke="url(#gradA)" strokeWidth="1" opacity={1 - r / 340} />
    );
    arcs.push(
      <circle key={`b${r}`} cx={cx2} cy={cy2} r={r} fill="none"
        stroke="url(#gradB)" strokeWidth="1" opacity={1 - r / 340} />
    );
  }
  return (
    <svg className={className} viewBox="0 0 520 520" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
      <defs>
        <linearGradient id="gradA" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stopColor="#5b8def" />
          <stop offset="1" stopColor="#2b2d6e" />
        </linearGradient>
        <linearGradient id="gradB" x1="1" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor="#b06ad9" />
          <stop offset="1" stopColor="#2b2d6e" />
        </linearGradient>
      </defs>
      {arcs}
    </svg>
  );
}
