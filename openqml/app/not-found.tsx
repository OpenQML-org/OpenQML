import Link from "next/link";
import Nav from "@/components/Nav";
import Footer from "@/components/Footer";

export default function NotFound() {
  return (
    <>
      <Nav />
      <section className="detail" style={{ minHeight: "50vh" }}>
        <div className="wrap">
          <div style={{ fontFamily: "var(--f-mono)", fontSize: 12, color: "var(--ink-faint)", marginBottom: 14 }}>404</div>
          <h1>No such entry.</h1>
          <p style={{ marginTop: 16, color: "var(--ink-soft)", maxWidth: "48ch" }}>
            The page does not exist. The catalog index is on the{" "}
            <Link href="/" style={{ color: "var(--indigo-bright)" }}>front page</Link>;
            the API lists everything at <code style={{ fontFamily: "var(--f-mono)", fontSize: 14, background: "var(--chip-bg)", padding: "2px 7px", borderRadius: 5 }}>/api/catalog</code>.
          </p>
        </div>
      </section>
      <Footer />
    </>
  );
}
