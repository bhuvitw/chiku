import { ARTIFACT_CAVEAT, MODEL_SCOPE_CAVEAT, PREDICTION_COPY } from "@/lib/copy";
import type { Results } from "@/lib/api";

const TONE: Record<string, string> = {
  alert:
    "border-amber-300 bg-amber-50 text-amber-900 dark:border-amber-900/50 dark:bg-amber-950/30 dark:text-amber-200",
  clear:
    "border-emerald-300 bg-emerald-50 text-emerald-900 dark:border-emerald-900/50 dark:bg-emerald-950/30 dark:text-emerald-200",
  unknown:
    "border-neutral-300 bg-neutral-50 text-neutral-800 dark:border-neutral-700 dark:bg-neutral-900 dark:text-neutral-200",
};

/**
 * Never renders "100.0%".
 *
 * A 0.9998 probability rounds to 100.0% at one decimal place, which reads as
 * certainty from a model whose measured test accuracy is 94%. The model has
 * been calibrated (ECE 0.025) precisely so its numbers mean something, and
 * rounding one of them up to absolute confidence throws that away. The same
 * clamp applies at the bottom of the range.
 */
function formatConfidence(confidence: number | undefined): string {
  if (confidence === undefined) return "—";
  const percent = confidence * 100;
  if (percent >= 99.95) return ">99.9%";
  if (percent <= 0.05) return "<0.1%";
  return `${percent.toFixed(1)}%`;
}

export function ResultsView({ results }: { results: Results }) {
  const { result } = results;
  const copy = PREDICTION_COPY[result.prediction];
  const abstained = result.prediction === "unable_to_assess";

  return (
    <section>
      <h2 className="text-lg font-medium">Result</h2>

      <div className={`mt-3 rounded-lg border px-5 py-4 ${TONE[copy.tone]}`}>
        <p className="text-xl font-semibold">{copy.label}</p>
        <p className="mt-1 text-sm">{copy.detail}</p>

        {abstained ? (
          // FR-04: the abstention response carries no confidence field at all.
          // Rendering a number here — even a hedged one — would undo the point
          // of abstaining.
          <p className="mt-3 text-sm">{result.message}</p>
        ) : (
          <dl className="mt-4 grid grid-cols-2 gap-x-6 gap-y-2 text-sm">
            <dt className="text-current/70">Confidence</dt>
            <dd className="text-right font-mono">{formatConfidence(result.confidence)}</dd>
          </dl>
        )}
      </div>

      <dl className="mt-4 grid grid-cols-[auto_1fr] gap-x-6 gap-y-1 text-xs text-neutral-500">
        <dt>Model</dt>
        <dd className="font-mono break-all">{result.model_version}</dd>
        <dt>Study</dt>
        <dd className="font-mono break-all">{results.study_id}</dd>
      </dl>

      {!abstained ? (
        <div className="mt-5 rounded-md border border-amber-300 bg-amber-50 px-4 py-3 text-sm text-amber-900 dark:border-amber-900/50 dark:bg-amber-950/30 dark:text-amber-200">
          <p className="font-medium">Known failure mode on casts and implants</p>
          <p className="mt-1">{ARTIFACT_CAVEAT}</p>
        </div>
      ) : null}

      <p className="mt-4 text-xs text-neutral-500">{MODEL_SCOPE_CAVEAT}</p>
      {/* PRD §12: carried by the API on every result, rendered verbatim. */}
      <p className="mt-2 text-xs font-medium text-neutral-600 dark:text-neutral-400">
        {results.disclaimer}
      </p>
    </section>
  );
}
