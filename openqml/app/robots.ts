import { MetadataRoute } from "next";

export default function robots(): MetadataRoute.Robots {
  return {
    rules: { userAgent: "*", allow: "/", disallow: ["/review", "/api/review"] },
    sitemap: "https://openqml.org/sitemap.xml",
  };
}
