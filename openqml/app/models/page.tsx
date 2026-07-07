import Nav from "@/components/Nav";
import Footer from "@/components/Footer";
import CatalogList from "@/components/CatalogList";
import CommunitySection from "@/components/CommunitySection";
import { models } from "@/lib/catalog";

export const metadata = { title: "Models — OpenQML" };

export default function ModelsPage() {
  return (
    <>
      <Nav />
      <header className="page-head">
        <div className="wrap">
          <h1>Models</h1>
          <p>
            Trainable quantum and hybrid models. Fields per entry: mechanism, scope of
            validity, failure modes, references, typical qubit count.
          </p>
        </div>
      </header>
      <section className="section" style={{ borderTop: "none", paddingTop: 40 }}>
        <div className="wrap">
          <CatalogList entries={models} base="/models" />
          <CommunitySection type="model" />
        </div>
      </section>
      <Footer />
    </>
  );
}
