import { createStudyAction } from "./actions";
import { listStudies, type Study } from "@/lib/api";

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

export default async function Home() {
  const { studies, error } = await loadStudies();

  return (
    <main className="mx-auto w-full max-w-3xl px-4 py-10">
      <h1 className="text-2xl font-semibold">Studies</h1>
      <p className="mt-1 text-sm text-neutral-500">
        Phase 0 scaffolding — create a study record and read it back from Postgres.
      </p>

      <form
        action={createStudyAction}
        className="mt-6 flex flex-wrap items-end gap-3 rounded-lg border border-neutral-200 p-4 dark:border-neutral-800"
      >
        <label className="flex flex-col gap-1 text-sm">
          <span className="text-neutral-500">Modality</span>
          <select
            name="modality"
            defaultValue="xray"
            className="rounded-md border border-neutral-300 bg-transparent px-3 py-2 dark:border-neutral-700"
          >
            <option value="xray">X-ray</option>
            <option value="mri">MRI</option>
          </select>
        </label>

        <label className="flex flex-col gap-1 text-sm">
          <span className="text-neutral-500">Body part</span>
          <input
            name="body_part"
            placeholder="wrist"
            className="rounded-md border border-neutral-300 bg-transparent px-3 py-2 dark:border-neutral-700"
          />
        </label>

        <button
          type="submit"
          className="rounded-md bg-neutral-900 px-4 py-2 text-sm font-medium text-white hover:bg-neutral-700 dark:bg-white dark:text-neutral-900 dark:hover:bg-neutral-200"
        >
          Create study
        </button>
      </form>

      {error ? (
        <p className="mt-6 rounded-md border border-amber-300 bg-amber-50 px-4 py-3 text-sm text-amber-900 dark:border-amber-900/50 dark:bg-amber-950/30 dark:text-amber-200">
          {error} Start the API and refresh.
        </p>
      ) : studies.length === 0 ? (
        <p className="mt-6 text-sm text-neutral-500">No studies yet.</p>
      ) : (
        <ul className="mt-6 divide-y divide-neutral-200 dark:divide-neutral-800">
          {studies.map((study) => (
            <li key={study.id} className="flex items-center justify-between gap-4 py-3">
              <div>
                <p className="font-medium">
                  {study.modality.toUpperCase()}
                  {study.body_part ? ` — ${study.body_part}` : ""}
                </p>
                <p className="font-mono text-xs text-neutral-500">{study.id}</p>
              </div>
              <span className="rounded-full border border-neutral-300 px-2.5 py-0.5 text-xs text-neutral-600 dark:border-neutral-700 dark:text-neutral-400">
                {study.status}
              </span>
            </li>
          ))}
        </ul>
      )}
    </main>
  );
}
