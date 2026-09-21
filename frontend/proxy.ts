/**
 * Route protection (Next.js 16 renamed Middleware to Proxy).
 *
 * This is an *optimistic* check only — it reads the session cookie's presence
 * and redirects, nothing more. The Next.js authentication guide is explicit
 * that Proxy is not a session-management or authorization solution, so every
 * protected page re-checks `getSession()` itself rather than trusting that a
 * request reaching it was already vetted.
 */

import { NextResponse } from "next/server";
import type { NextRequest } from "next/server";

import { SESSION_COOKIE } from "@/lib/session";

export function proxy(request: NextRequest) {
  // The bypass is read here too, or a developer who set it would still be
  // bounced to /login by a check that never sees their (absent) cookie.
  const bypassed = Boolean(process.env.DEV_AUTH_BYPASS_EMAIL);
  const signedIn = bypassed || Boolean(request.cookies.get(SESSION_COOKIE)?.value);
  const { pathname } = request.nextUrl;

  if (!signedIn && pathname !== "/login") {
    const url = request.nextUrl.clone();
    url.pathname = "/login";
    // Preserved so a deep link survives the round trip through sign-in.
    url.searchParams.set("next", pathname);
    return NextResponse.redirect(url);
  }

  if (signedIn && pathname === "/login") {
    const url = request.nextUrl.clone();
    url.pathname = "/";
    url.search = "";
    return NextResponse.redirect(url);
  }

  return NextResponse.next();
}

export const config = {
  // Everything except Next's own assets and the route handlers. `/api/*` is
  // excluded deliberately: redirecting an API request to an HTML login page
  // hands a `fetch` a 200 full of markup, which the caller then tries to parse
  // as JSON. Those routes check the session themselves and answer with 401.
  matcher: ["/((?!api/|_next/static|_next/image|favicon.ico).*)"],
};
