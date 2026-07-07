import Link from "next/link";
import SearchBox from "@/components/SearchBox";
import ThemeToggle from "@/components/ThemeToggle";
import UserChip from "@/components/UserChip";
import NavLinks from "@/components/NavLinks";

export default function Nav() {
  return (
    <nav className="nav">
      <div className="wrap nav-inner">
        <Link href="/" className="brand">
          <svg className="dot" viewBox="0 0 18 18" fill="none">
            <circle cx="9" cy="9" r="8" stroke="#2b2d6e" strokeWidth="1.2" />
            <circle cx="9" cy="9" r="3.4" fill="#4a4de8" />
            <circle cx="9" cy="9" r="8" stroke="#b06ad9" strokeWidth="0.6" strokeDasharray="2 3" />
          </svg>
          <span>OpenQML</span>
        </Link>
        <NavLinks />
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <SearchBox />
          <ThemeToggle />
          <UserChip />
          <Link href="/submit" className="nav-cta">Submit</Link>
        </div>
      </div>
    </nav>
  );
}
