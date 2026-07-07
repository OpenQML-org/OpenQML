import RevealObserver from "@/components/RevealObserver";

export default function Template({ children }: { children: React.ReactNode }) {
  return (
    <div className="page-enter">
      <RevealObserver />
      {children}
    </div>
  );
}
