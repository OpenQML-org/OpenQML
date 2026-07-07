import { cookies } from "next/headers";
import { SessionUser, verifySessionValue, SESSION_COOKIE } from "@/lib/session";

async function cfEnv(): Promise<Record<string, unknown> | null> {
  try {
    const { getCloudflareContext } = await import("@opennextjs/cloudflare");
    const ctx = await getCloudflareContext({ async: true });
    return ctx.env as unknown as Record<string, unknown>;
  } catch {
    return null;
  }
}

export async function env(name: string): Promise<string> {
  const e = await cfEnv();
  return ((e?.[name] as string) || process.env[name] || "").trim();
}

export async function sessionSecret(): Promise<string> {
  const s = await env("SESSION_SECRET");
  if (s) return s;
  return process.env.NODE_ENV !== "production" ? "dev-secret" : "";
}

export async function devLoginEnabled(): Promise<boolean> {
  return (await env("DEV_LOGIN")) === "1" || process.env.NODE_ENV !== "production";
}

/** Session from server components / route handlers. */
export async function getSession(): Promise<SessionUser | null> {
  const jar = await cookies();
  const v = jar.get(SESSION_COOKIE)?.value;
  if (!v) return null;
  const secret = await sessionSecret();
  if (!secret) return null;
  return verifySessionValue(v, secret);
}

export interface ProviderConf {
  id: "github" | "google" | "apple";
  label: string;
  configured: boolean;
}

export async function providers(): Promise<ProviderConf[]> {
  return [
    { id: "github", label: "GitHub", configured: !!(await env("GITHUB_CLIENT_ID")) && !!(await env("GITHUB_CLIENT_SECRET")) },
    { id: "google", label: "Google", configured: !!(await env("GOOGLE_CLIENT_ID")) && !!(await env("GOOGLE_CLIENT_SECRET")) },
    { id: "apple", label: "Apple", configured: !!(await env("APPLE_CLIENT_ID")) && !!(await env("APPLE_TEAM_ID")) && !!(await env("APPLE_KEY_ID")) && !!(await env("APPLE_PRIVATE_KEY")) },
  ];
}
