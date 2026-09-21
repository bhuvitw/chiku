"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";

import type { StudyStatusPayload } from "@/lib/api";

const TERMINAL = new Set(["COMPLETED", "FAILED", "UNSUPPORTED"]);
const POLL_MS = 1200;
/** Roughly two minutes. A job that has not moved by then is not going to. */
const MAX_POLLS = 100;

/**
 * The processing view (System Design §18).
 *
 * Polls rather than streams because the API contract is polling — that was
 * fixed in Phase 2 so switching the worker on later changes nothing here.
 * When the study reaches a terminal state it calls `router.refresh()`, which
 * re-runs the server component that decides which view to render; this
 * component never decides that itself.
 */
export function ProcessingView({ initial }: { initial: StudyStatusPayload }) {
  const router = useRouter();
  const [status, setStatus] = useState(initial);
  const [stalled, setStalled] = useState(false);

  useEffect(() => {
    if (TERMINAL.has(status.status)) {
      router.refresh();
      return;
    }

    let cancelled = false;
    let polls = 0;

    const timer = setInterval(async () => {
      polls += 1;
      if (polls > MAX_POLLS) {
        clearInterval(timer);
        if (!cancelled) setStalled(true);
        return;
      }
      const response = await fetch(`/api/studies/${status.study_id}/status`, {
        cache: "no-store",
      });
      if (response.status === 401) {
        // The session expired mid-poll. Stop and let the server component
        // redirect, rather than retrying 401s for two minutes.
        clearInterval(timer);
        router.refresh();
        return;
      }
      if (!response.ok || cancelled) return;
      const next = (await response.json()) as StudyStatusPayload;
      if (cancelled) return;
      setStatus(next);
      if (TERMINAL.has(next.status)) {
        clearInterval(timer);
        router.refresh();
      }
    }, POLL_MS);

    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, [status.status, status.study_id, router]);

  const progress = Math.min(100, Math.max(0, status.progress));

  return (
    <section aria-live="polite">
      <h2 className="text-lg font-medium">Analysing</h2>
      <p className="mt-1 text-sm text-neutral-500">
        {stalled
          ? "This job has not reported progress for some time."
          : "The image is being preprocessed and scored. This usually takes a few seconds."}
      </p>

      <div
        role="progressbar"
        aria-valuenow={progress}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-label="Analysis progress"
        className="mt-4 h-2 w-full overflow-hidden rounded-full bg-neutral-200 dark:bg-neutral-800"
      >
        <div
          className="h-full rounded-full bg-neutral-900 transition-[width] duration-500 dark:bg-white"
          style={{ width: `${progress}%` }}
        />
      </div>

      <p className="mt-2 font-mono text-xs text-neutral-500">
        {status.status} · {progress}%
        {status.job ? ` · job ${status.job.status.toLowerCase()}` : ""}
      </p>

      {stalled ? (
        <button
          onClick={() => router.refresh()}
          className="mt-4 rounded-md border border-neutral-300 px-4 py-2 text-sm dark:border-neutral-700"
        >
          Check again
        </button>
      ) : null}
    </section>
  );
}
