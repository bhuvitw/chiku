/**
 * Status proxy for the processing view.
 *
 * The browser cannot reach the backend directly — `API_INTERNAL_URL` is a
 * server-side name and the session cookie is httpOnly — so the poll goes
 * through here, which re-checks the session on every request.
 */

import { NextResponse } from "next/server";

import { getStatus } from "@/lib/api";
import { getSession } from "@/lib/session";

export async function GET(_request: Request, ctx: RouteContext<"/api/studies/[id]/status">) {
  if ((await getSession()) === null) {
    return NextResponse.json(
      { error: { code: "UNAUTHORIZED", message: "Authentication is required." } },
      { status: 401 },
    );
  }

  const { id } = await ctx.params;
  try {
    return NextResponse.json(await getStatus(id));
  } catch {
    return NextResponse.json(
      { error: { code: "NOT_FOUND", message: "The requested resource was not found." } },
      { status: 404 },
    );
  }
}
