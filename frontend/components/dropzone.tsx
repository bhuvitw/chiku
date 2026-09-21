"use client";

import { useRef, useState, useTransition } from "react";

/** Mirrors backend `allowed_content_types`; DICOM is refused server-side too. */
const ACCEPTED = ["image/png", "image/jpeg"];
/** Mirrors `max_upload_bytes`. */
const MAX_BYTES = 64 * 1024 * 1024;
/** Mirrors `min_image_pixels`, so an obviously-too-small image fails here first. */
const MIN_PIXELS = 128;

/**
 * Client-side checks are a courtesy, never a gate: every one of these is
 * re-run server-side, because a file that reaches the API has not necessarily
 * been through this component. The point is a fast, specific message instead
 * of a round trip that returns the same verdict a second later.
 */
async function localReasonToReject(file: File): Promise<string | null> {
  if (!ACCEPTED.includes(file.type)) {
    return `${file.type || "This file type"} is not supported. Upload a PNG or JPEG X-ray.`;
  }
  if (file.size > MAX_BYTES) {
    return `This file is ${(file.size / 1024 / 1024).toFixed(1)} MB; the limit is 64 MB.`;
  }
  if (file.size === 0) {
    return "This file is empty.";
  }

  const dimensions = await new Promise<{ width: number; height: number } | null>((resolve) => {
    const url = URL.createObjectURL(file);
    const image = new Image();
    image.onload = () => {
      URL.revokeObjectURL(url);
      resolve({ width: image.naturalWidth, height: image.naturalHeight });
    };
    image.onerror = () => {
      URL.revokeObjectURL(url);
      resolve(null);
    };
    image.src = url;
  });

  if (dimensions === null) return "This file could not be read as an image.";
  if (Math.min(dimensions.width, dimensions.height) < MIN_PIXELS) {
    return `This image is ${dimensions.width}x${dimensions.height}; the minimum is ${MIN_PIXELS}px on each side.`;
  }
  return null;
}

export function Dropzone({
  action,
}: {
  action: (formData: FormData) => Promise<{ error: string } | void>;
}) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const [rejection, setRejection] = useState<string | null>(null);
  const [filename, setFilename] = useState<string | null>(null);
  const [pending, startTransition] = useTransition();

  async function submit(file: File) {
    setRejection(null);
    const reason = await localReasonToReject(file);
    if (reason) {
      setRejection(reason);
      setFilename(null);
      return;
    }
    setFilename(file.name);
    const formData = new FormData();
    formData.append("file", file);
    startTransition(async () => {
      const result = await action(formData);
      // A server-side rejection surfaces here rather than throwing: the
      // error contract's copy belongs next to the dropzone that caused it.
      if (result && "error" in result) {
        setRejection(result.error);
        setFilename(null);
      }
    });
  }

  return (
    <div>
      <div
        onDragOver={(event) => {
          event.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(event) => {
          event.preventDefault();
          setDragging(false);
          const file = event.dataTransfer.files?.[0];
          if (file) void submit(file);
        }}
        onClick={() => inputRef.current?.click()}
        onKeyDown={(event) => {
          if (event.key === "Enter" || event.key === " ") {
            event.preventDefault();
            inputRef.current?.click();
          }
        }}
        role="button"
        tabIndex={0}
        aria-label="Upload an X-ray image"
        aria-busy={pending}
        className={[
          "flex cursor-pointer flex-col items-center justify-center gap-2 rounded-lg border-2 border-dashed px-6 py-14 text-center transition-colors",
          dragging
            ? "border-neutral-900 bg-neutral-50 dark:border-white dark:bg-neutral-900"
            : "border-neutral-300 hover:border-neutral-400 dark:border-neutral-700 dark:hover:border-neutral-600",
          pending ? "pointer-events-none opacity-60" : "",
        ].join(" ")}
      >
        <input
          ref={inputRef}
          type="file"
          accept={ACCEPTED.join(",")}
          className="sr-only"
          onChange={(event) => {
            const file = event.target.files?.[0];
            if (file) void submit(file);
            event.target.value = "";
          }}
        />
        <p className="text-sm font-medium">
          {pending
            ? `Uploading ${filename ?? "image"}…`
            : "Drop an X-ray here, or click to choose one"}
        </p>
        <p className="text-xs text-neutral-500">PNG or JPEG, at least 128px per side, up to 64 MB</p>
      </div>

      {rejection ? (
        <p
          role="alert"
          className="mt-3 rounded-md border border-red-300 bg-red-50 px-4 py-3 text-sm text-red-900 dark:border-red-900/50 dark:bg-red-950/30 dark:text-red-200"
        >
          {rejection}
        </p>
      ) : null}
    </div>
  );
}
