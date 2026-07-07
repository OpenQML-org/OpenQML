import Nav from "@/components/Nav";
import Footer from "@/components/Footer";
import Playground from "@/components/Playground";

export const metadata = { title: "Playground — OpenQML" };

export default function PlaygroundPage() {
  return (
    <>
      <Nav />
      <header className="page-head">
        <div className="wrap">
          <h1>Playground</h1>
          <p>
            Gate-level circuit builder on an exact state-vector simulator, in the browser.
            Presets: Bell, GHZ, single-qubit interference, one Grover iteration.
          </p>
        </div>
      </header>
      <section className="section" style={{ borderTop: "none", paddingTop: 40 }}>
        <div className="wrap">
          <Playground />
        </div>
      </section>
      <Footer />
    </>
  );
}
