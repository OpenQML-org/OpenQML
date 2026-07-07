import { providers, devLoginEnabled } from "@/lib/auth";
import DevLogin from "@/components/DevLogin";
import Interference from "@/components/Interference";

export const dynamic = "force-dynamic";
export const metadata = { title: "Sign in — OpenQML" };

const ICONS: Record<string, React.ReactNode> = {
  github: (
    <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor" aria-hidden>
      <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27s1.36.09 2 .27c1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8z"/>
    </svg>
  ),
  google: (
    <svg width="16" height="16" viewBox="0 0 24 24" aria-hidden>
      <path fill="#4285F4" d="M23.5 12.27c0-.85-.08-1.66-.22-2.45H12v4.64h6.45a5.52 5.52 0 0 1-2.39 3.62v3h3.87c2.26-2.09 3.57-5.17 3.57-8.81z"/>
      <path fill="#34A853" d="M12 24c3.24 0 5.96-1.07 7.93-2.91l-3.87-3a7.19 7.19 0 0 1-10.71-3.78H1.34v3.1A12 12 0 0 0 12 24z"/>
      <path fill="#FBBC05" d="M5.35 14.31a7.2 7.2 0 0 1 0-4.6V6.6H1.34a12 12 0 0 0 0 10.8l4.01-3.1z"/>
      <path fill="#EA4335" d="M12 4.75c1.77 0 3.35.61 4.6 1.8l3.44-3.44A11.97 11.97 0 0 0 1.34 6.6l4.01 3.1A7.19 7.19 0 0 1 12 4.75z"/>
    </svg>
  ),
  apple: (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor" aria-hidden>
      <path d="M16.36 12.94c.03 3.16 2.77 4.21 2.8 4.22-.02.07-.44 1.5-1.44 2.98-.87 1.28-1.77 2.55-3.19 2.58-1.4.03-1.85-.83-3.44-.83-1.6 0-2.1.8-3.42.86-1.37.05-2.42-1.38-3.3-2.65C2.58 17.5 1.2 12.72 3.05 9.6a5.13 5.13 0 0 1 4.34-2.63c1.35-.03 2.62.91 3.44.91.82 0 2.37-1.12 4-96.96a4.87 4.87 0 0 1 4 2.03c-.1.06-2.4 1.4-2.47 3.99zM13.7 4.66c.73-.88 1.22-2.1 1.08-3.32-1.05.04-2.32.7-3.07 1.58-.67.78-1.26 2.03-1.1 3.22 1.17.09 2.36-.6 3.09-1.48z"/>
    </svg>
  ),
};

export default async function LoginPage({ searchParams }: { searchParams: Promise<{ error?: string; next?: string }> }) {
  const sp = await searchParams;
  const provs = await providers();
  const configured = provs.filter((p) => p.configured);
  const dev = await devLoginEnabled();
  const next = sp.next && sp.next.startsWith("/") ? sp.next : "/";

  return (
    <main className="login-page">
      <Interference className="login-bg" />
      <div className="login-card">
        <div className="brand" style={{ marginBottom: 8 }}>
          <svg className="dot" viewBox="0 0 18 18" fill="none" style={{ width: 20, height: 20 }}>
            <circle cx="9" cy="9" r="8" stroke="#2b2d6e" strokeWidth="1.2" />
            <circle cx="9" cy="9" r="3.4" fill="#4a4de8" />
          </svg>
          <span style={{ fontSize: 18 }}>OpenQML</span>
        </div>
        <p className="login-sub">Sign in to continue.</p>

        {sp.error && (
          <p className="login-err">
            Sign-in failed ({sp.error}). Try again or use another provider.
          </p>
        )}

        <div className="login-btns">
          {configured.map((p) => (
            <a key={p.id} className="login-btn" href={`/api/auth/${p.id}?next=${encodeURIComponent(next)}`}>
              {ICONS[p.id]}
              <span>Continue with {p.label}</span>
            </a>
          ))}
          {configured.length === 0 && !dev && (
            <p style={{ fontSize: 14, color: "var(--ink-soft)" }}>
              No sign-in providers are configured. Set OAuth credentials as described in DEPLOY.md.
            </p>
          )}
        </div>

        {dev && <DevLogin next={next} showDivider={configured.length > 0} />}

        <p className="login-fine">
          Name and avatar are stored. Sessions last 30 days.
        </p>
      </div>
    </main>
  );
}
