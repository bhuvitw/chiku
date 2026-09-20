"use client";

export default function Error({ reset }: { error: Error; reset: () => void }) {
  return (
    <main className="mx-auto w-full max-w-3xl px-4 py-10">
      <h1 className="text-2xl font-semibold">Something went wrong</h1>
      <p className="mt-2 text-sm text-neutral-500">
        The request could not be completed. No analysis result was produced.
      </p>
      <button
        onClick={reset}
        className="mt-4 rounded-md border border-neutral-300 px-4 py-2 text-sm dark:border-neutral-700"
      >
        Try again
      </button>
    </main>
  );
}
