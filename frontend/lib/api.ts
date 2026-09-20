/**
 * Server-side client for the backend API.
 *
 * Mirrors the single error envelope defined in System Design §3 so a failure
 * always arrives as a code the UI can map to the copy required by §26, never
 * as an untyped exception.
 */

const API_BASE_URL = process.env.API_INTERNAL_URL ?? "http://localhost:8000";

export type Modality = "xray" | "mri";

export type StudyStatus =
  | "UPLOADED"
  | "VALIDATING"
  | "PREPROCESSING"
  | "ANALYZING"
  | "GENERATING_3D"
  | "COMPLETED"
  | "FAILED"
  | "UNSUPPORTED";

export interface Study {
  id: string;
  modality: Modality;
  body_part: string | null;
  status: StudyStatus;
  created_at: string;
  updated_at: string;
}

export class ApiError extends Error {
  readonly code: string;

  constructor(code: string, message: string) {
    super(message);
    this.name = "ApiError";
    this.code = code;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...init?.headers },
  });

  if (!response.ok) {
    const body = await response.json().catch(() => null);
    const error = (body as { error?: { code?: string; message?: string } } | null)?.error;
    throw new ApiError(
      error?.code ?? "INTERNAL_ERROR",
      error?.message ?? "An unexpected error occurred.",
    );
  }

  return response.json() as Promise<T>;
}

export function listStudies(): Promise<Study[]> {
  return request<Study[]>("/api/v1/studies");
}

export function createStudy(input: {
  modality: Modality;
  body_part: string | null;
}): Promise<Study> {
  return request<Study>("/api/v1/studies", {
    method: "POST",
    body: JSON.stringify(input),
  });
}
