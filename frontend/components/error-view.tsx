import { ERROR_COPY } from "@/lib/copy";
import type { ErrorCode } from "@/lib/api";

import { reanalyzeAction } from "@/app/actions";

/**
 * Renders a backend failure through the error contract (System Design §26).
 *
 * `message` comes from the API and is shown as-is — it is product copy fixed
 * by the design document, and re-writing it here would give the same failure
 * two different wordings depending on where the user saw it.
 */
export function ErrorView({
  code,
  message,
  studyId,
}: {
  code: ErrorCode;
  message: string | null;
  studyId: string;
}) {
  const copy = ERROR_COPY[code] ?? ERROR_COPY.INTERNAL_ERROR;

  return (
    <section>
      <h2 className="text-lg font-medium">{copy.title}</h2>

      <div className="mt-3 rounded-lg border border-red-300 bg-red-50 px-5 py-4 text-sm text-red-900 dark:border-red-900/50 dark:bg-red-950/30 dark:text-red-200">
        <p className="font-medium">{message ?? copy.detail}</p>
        {message ? <p className="mt-2">{copy.detail}</p> : null}
        <p className="mt-3 font-mono text-xs opacity-70">{code}</p>
      </div>

      {copy.retryable ? (
        <form action={reanalyzeAction} className="mt-4">
          <input type="hidden" name="study_id" value={studyId} />
          <button
            type="submit"
            className="rounded-md border border-neutral-300 px-4 py-2 text-sm dark:border-neutral-700"
          >
            Retry analysis
          </button>
        </form>
      ) : (
        // UNSUPPORTED is terminal-and-do-not-retry; offering a retry that can
        // never succeed is worse than offering nothing.
        <p className="mt-4 text-sm text-neutral-500">
          Retrying will not change this result. Upload a different image instead.
        </p>
      )}
    </section>
  );
}
