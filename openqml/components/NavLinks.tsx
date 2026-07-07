"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";

const LINKS: Array<[string, string]> = [
  ["/models", "Models"],
  ["/algorithms", "Algorithms"],
  ["/playground", "Playground"],
  ["/benchmarks", "Benchmarks"],
  ["/forum", "Forum"],
  ["/docs", "Standard"],
  ["/about", "About"],
];

export default function NavLinks() {
  const path = usePathname();
  return (
    <div className="nav-links">
      {LINKS.map(([href, label]) => {
        const active = path === href || path.startsWith(href + "/");
        return (
          <Link key={href} href={href} className={active ? "active" : ""}>
            {label}
          </Link>
        );
      })}
    </div>
  );
}
