import Nav from "@/components/Nav";
import Footer from "@/components/Footer";
import CatalogList from "@/components/CatalogList";
import CommunitySection from "@/components/CommunitySection";
import { algorithms } from "@/lib/catalog";

export const metadata = { title: "Algorithms — OpenQML" };

export default function AlgorithmsPage() {
  return (
    <>
      <Nav />
      <header className="page-head">
        <div className="wrap">
          <h1>Algorithms</h1>
          <p>
            Quantum algorithms, gradient rules, error mitigation, and error correction.
            Year is the year of introduction; maturity reflects current hardware status.
          </p>
        </div>
      </header>
      <section className="section" style={{ borderTop: "none", paddingTop: 40 }}>
        <div className="wrap">
          <CatalogList entries={algorithms} base="/algorithms" />
          <CommunitySection type="algorithm" />
        </div>
      </section>
      <Footer />
    </>
  );
}
