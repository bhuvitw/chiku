import { signInAction } from "@/app/actions";

export const dynamic = "force-dynamic";

export default async function LoginPage({ searchParams }: PageProps<"/login">) {
  const params = await searchParams;
  const next = typeof params.next === "string" ? params.next : "/";
  const invalid = params.error === "invalid";

  return (
    <main className="mx-auto w-full max-w-sm px-4 py-16">
      <h1 className="text-2xl font-semibold">Sign in</h1>
      <p className="mt-2 text-sm text-neutral-500">
        A development stand-in so routes have a session to check against. It does not
        authenticate anyone and does not separate one person&apos;s studies from another&apos;s.
      </p>

      <form action={signInAction} className="mt-6 flex flex-col gap-3">
        <input type="hidden" name="next" value={next} />
        <label className="flex flex-col gap-1 text-sm">
          <span className="text-neutral-500">Email</span>
          <input
            name="email"
            type="email"
            required
            autoComplete="email"
            placeholder="you@example.com"
            className="rounded-md border border-neutral-300 bg-transparent px-3 py-2 dark:border-neutral-700"
          />
        </label>

        {invalid ? (
          <p role="alert" className="text-sm text-red-600 dark:text-red-400">
            Enter a valid email address.
          </p>
        ) : null}

        <button
          type="submit"
          className="rounded-md bg-neutral-900 px-4 py-2 text-sm font-medium text-white hover:bg-neutral-700 dark:bg-white dark:text-neutral-900 dark:hover:bg-neutral-200"
        >
          Continue
        </button>
      </form>
    </main>
  );
}
