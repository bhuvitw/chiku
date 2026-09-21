import Link from "next/link";
import { notFound } from "next/navigation";

import { ApiError, getResults, getStatus, getStudy } from "@/lib/api";
import { getSession } from "@/lib/session";
import { ErrorView } from "@/components/error-view";
import { ProcessingView } from "@/components/processing-view";
import { ResultsView } from "@/components/results-view";
import { redirect } from "next/navigation";

// Per-study state changes between requests; nothing here is prerenderable.
export const dynamic = "force-dynamic";

/**
 * One page, four states.
 *
 * Which view renders is decided here from the study's status rather than by
 * the client, so the processing view's only job is to poll and refresh — it
 * never has to know what a completed study should look like.
 */
export default async function StudyPage({ params }: PageProps<"/studies/[id]">) {
  // The proxy redirect is optimistic; this is the check that counts.
  if ((await getSession()) === null) redirect("/login");

  const { id } = await params;

  let study;
  try {
    study = await getStudy(id);
  } catch (error) {
    if (error instanceof ApiError && error.code === "NOT_FOUND") notFound();
    throw error;
  }

  const status = await getStatus(id);

  return (
    <main className="mx-auto w-full max-w-3xl px-4 py-10">
      <Link href="/" className="text-sm text-neutral-500 hover:underline">
        ← All studies
      </Link>

      <h1 className="mt-3 text-2xl font-semibold">
        {study.modality.toUpperCase()}
        {study.body_part ? ` — ${study.body_part}` : ""}
      </h1>
      <p className="mt-1 font-mono text-xs text-neutral-500">{study.id}</p>

      <div className="mt-8">
        {status.status === "COMPLETED" ? (
          <CompletedStudy id={id} />
        ) : status.status === "FAILED" || status.status === "UNSUPPORTED" ? (
          <ErrorView
            code={status.error_code ?? "PROCESSING_FAILED"}
            message={status.message}
            studyId={id}
          />
        ) : (
          <ProcessingView initial={status} />
        )}
      </div>
    </main>
  );
}

/**
 * Fetching is separated from rendering so no JSX is constructed inside the
 * try/catch: React renders lazily, so a component built in a `try` would
 * throw outside it and never be caught.
 */
async function loadResults(id: string) {
  try {
    return { ok: true as const, results: await getResults(id) };
  } catch (error) {
    // A study can be COMPLETED with no readable result only if something is
    // inconsistent; surface it through the contract rather than a blank page.
    if (error instanceof ApiError) {
      return { ok: false as const, code: error.code, message: error.message };
    }
    throw error;
  }
}

async function CompletedStudy({ id }: { id: string }) {
  const outcome = await loadResults(id);
  return outcome.ok ? (
    <ResultsView results={outcome.results} />
  ) : (
    <ErrorView code={outcome.code} message={outcome.message} studyId={id} />
  );
}
