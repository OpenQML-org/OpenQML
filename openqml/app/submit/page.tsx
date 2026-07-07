import Nav from "@/components/Nav";
import Footer from "@/components/Footer";
import SubmitForm from "@/components/SubmitForm";

export const metadata = { title: "Submit — OpenQML" };

export default function SubmitPage() {
  return (
    <>
      <Nav />
      <header className="page-head">
        <div className="wrap">
          <h1>Submit an entry</h1>
          <p>
            Submissions are validated against the description standard v0.1.
            The failure_modes field is required; omitting it returns a 422.
          </p>
        </div>
      </header>
      <section className="section" style={{ borderTop: "none", paddingTop: 40 }}>
        <div className="wrap" style={{ maxWidth: 720 }}>
          <SubmitForm />
        </div>
      </section>
      <Footer />
    </>
  );
}
