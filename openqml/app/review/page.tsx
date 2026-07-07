import Nav from "@/components/Nav";
import Footer from "@/components/Footer";
import ReviewPanel from "@/components/ReviewPanel";

export const metadata = { title: "Review queue — OpenQML", robots: { index: false } };

export default function ReviewPage() {
  return (
    <>
      <Nav />
      <header className="page-head">
        <div className="wrap">
          <h1>Review queue</h1>
          <p>
            Maintainer view. Approve entries that meet the standard — honest limits stated,
            claims reproducible. Approved entries appear in the community section of the catalog.
          </p>
        </div>
      </header>
      <section className="section" style={{ borderTop: "none", paddingTop: 40 }}>
        <div className="wrap" style={{ maxWidth: 760 }}>
          <ReviewPanel />
        </div>
      </section>
      <Footer />
    </>
  );
}
