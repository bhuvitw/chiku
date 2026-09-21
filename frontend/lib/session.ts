/**
 * Minimal session handling (PRD §13, implementation-plan §2).
 *
 * This is deliberately the smallest thing that gives `users`/authorization
 * something real to check against — an email in a signed-cookie-shaped
 * session, or a dev-mode bypass. It is **not** authentication:
 *
 * - The backend still attributes every study to its dev user
 *   (`backend/api/deps.py`), so this gates the UI, not the data. Two people
 *   signing in with different emails see the same studies.
 * - The cookie is not signed or encrypted, so it proves nothing to the server.
 *
 * Both of those are fine for a research prototype and are exactly what
 * Phase 2 asks for, but they must be replaced before anything real is stored.
 * The shape here — `getSession()` returning a user or null — is what the
 * replacement fills in, so call sites do not move.
 */

import "server-only";

import { cookies } from "next/headers";

export const SESSION_COOKIE = "chiku_session";

/** Dev bypass: when set, every request is treated as this user. */
const DEV_BYPASS_EMAIL = process.env.DEV_AUTH_BYPASS_EMAIL;

export interface Session {
  email: string;
  /** True when the session came from the bypass rather than a sign-in. */
  bypassed: boolean;
}

export async function getSession(): Promise<Session | null> {
  if (DEV_BYPASS_EMAIL) {
    return { email: DEV_BYPASS_EMAIL, bypassed: true };
  }
  const value = (await cookies()).get(SESSION_COOKIE)?.value;
  if (!value) return null;
  return { email: value, bypassed: false };
}

export async function signIn(email: string): Promise<void> {
  const store = await cookies();
  store.set(SESSION_COOKIE, email, {
    httpOnly: true,
    sameSite: "lax",
    secure: process.env.NODE_ENV === "production",
    path: "/",
    maxAge: 60 * 60 * 8,
  });
}

export async function signOut(): Promise<void> {
  (await cookies()).delete(SESSION_COOKIE);
}
