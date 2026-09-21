import Link from "next/link";
import { redirect } from "next/navigation";

import { signOutAction, uploadAndAnalyzeAction } from "./actions";
import { listStudies, type Study } from "@/lib/api";
import { getSession } from "@/lib/session";
import { Dropzone } from "@/components/dropzone";

// The study list must reflect database state at request time, and this keeps
// `next build` from trying to reach the backend while prerendering.
export const dynamic = "force-dynamic";

async function loadStudies(): Promise<{ studies: Study[]; error: string | null }> {
  try {
    return { studies: await listStudies(), error: null };
  } catch {
    return { studies: [], error: "Backend unreachable." };
  }
}

const STATUS_TONE: Record<string, string> = {
  COMPLETED: "border-emerald-300 text-emerald-700 dark:border-emerald-900/50 dark:text-emerald-300",
  FAILED: "border-red-300 text-red-700 dark:border-red-900/50 dark:text-red-300",
  UNSUPPORTED: "border-red-300 text-red-700 dark:border-red-900/50 dark:text-red-300",
};

export default async function Home() {
  const session = await getSession();
  if (session === null) redirect("/login");

  const { studies, error } = await loadStudies();

  return (
    <main className="mx-auto w-full max-w-3xl px-4 py-10">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h1 className="text-2xl font-semibold">Analyse a wrist X-ray</h1>
          <p className="mt-1 text-sm text-neutral-500">
            Paediatric wrist radiographs, PNG or JPEG. Each upload creates a study and starts
            an analysis.
          </p>
        </div>
        <form action={signOutAction}>
          <button
            type="submit"
            className="shrink-0 rounded-md border border-neutral-300 px-3 py-1.5 text-xs text-neutral-600 dark:border-neutral-700 dark:text-neutral-400"
          >
            {session.bypassed ? "Bypass active" : `Sign out ${session.email}`}
          </button>
        </form>
      </div>

      <div className="mt-6">
        <Dropzone action={uploadAndAnalyzeAction} />
      </div>

      <h2 className="mt-10 text-lg font-medium">Studies</h2>
      {error ? (
        <p className="mt-3 rounded-md border border-amber-300 bg-amber-50 px-4 py-3 text-sm text-amber-900 dark:border-amber-900/50 dark:bg-amber-950/30 dark:text-amber-200">
          {error} Start the API and refresh.
        </p>
      ) : studies.length === 0 ? (
        <p className="mt-3 text-sm text-neutral-500">No studies yet.</p>
      ) : (
        <ul className="mt-3 divide-y divide-neutral-200 dark:divide-neutral-800">
          {studies.map((study) => (
            <li key={study.id}>
              <Link
                href={`/studies/${study.id}`}
                className="flex items-center justify-between gap-4 py-3 hover:opacity-70"
              >
                <div>
                  <p className="font-medium">
                    {study.modality.toUpperCase()}
                    {study.body_part ? ` — ${study.body_part}` : ""}
                  </p>
                  <p className="font-mono text-xs text-neutral-500">{study.id}</p>
                </div>
                <span
                  className={`rounded-full border px-2.5 py-0.5 text-xs ${
                    STATUS_TONE[study.status] ??
                    "border-neutral-300 text-neutral-600 dark:border-neutral-700 dark:text-neutral-400"
                  }`}
                >
                  {study.status}
                </span>
              </Link>
            </li>
          ))}
        </ul>
      )}
    </main>
  );
}
