import { notFound } from "next/navigation";
import Nav from "@/components/Nav";
import Footer from "@/components/Footer";
import EntryDetail from "@/components/EntryDetail";
import { algorithms } from "@/lib/catalog";

export function generateStaticParams() {
  return algorithms.map((m) => ({ slug: m.slug }));
}

export async function generateMetadata({ params }: { params: Promise<{ slug: string }> }) {
  const { slug } = await params;
  const e = algorithms.find((m) => m.slug === slug);
  return { title: e ? `${e.name} — OpenQML` : "Not found — OpenQML" };
}

export default async function AlgorithmDetail({ params }: { params: Promise<{ slug: string }> }) {
  const { slug } = await params;
  const e = algorithms.find((m) => m.slug === slug);
  if (!e) notFound();
  return (
    <>
      <Nav />
      <EntryDetail e={e} backHref="/algorithms" backLabel="All algorithms" />
      <Footer />
    </>
  );
}
