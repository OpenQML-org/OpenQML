import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  metadataBase: new URL("https://openqml.org"),
  title: {
    default: "OpenQML — The open home for quantum machine learning",
    template: "%s",
  },
  description:
    "An open catalog, standard, and playground for quantum machine learning. Every entry states where it wins and where it does not.",
  openGraph: {
    title: "OpenQML — The open home for quantum machine learning",
    description:
      "Catalog, standard, live circuit playground, and honest benchmarks for QML.",
    url: "https://openqml.org",
    siteName: "OpenQML",
    type: "website",
  },
  twitter: {
    card: "summary",
    title: "OpenQML",
    description: "The open home for quantum machine learning.",
  },
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <script
          dangerouslySetInnerHTML={{
            __html: `try{var t=localStorage.getItem("theme")||"light";document.documentElement.dataset.theme=t}catch(e){}`,
          }}
        />
      </head>
      <body>{children}</body>
    </html>
  );
}
