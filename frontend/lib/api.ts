/**
 * Server-side client for the backend API.
 *
 * Mirrors the single error envelope defined in System Design §3 so a failure
 * always arrives as a code the UI can map to the copy required by §26, never
 * as an untyped exception.
 */

import "server-only";

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

export type JobStatus = "QUEUED" | "RUNNING" | "SUCCEEDED" | "FAILED";

/** PRD FR-04. `unable_to_assess` is a result, not an error. */
export type PredictionLabel = "possible_fracture" | "no_fracture" | "unable_to_assess";

/** System Design §26. The four codes the UI must render distinct copy for. */
export type ErrorCode =
  | "UNSUPPORTED_MODALITY"
  | "LOW_IMAGE_QUALITY"
  | "LOW_CONFIDENCE"
  | "PROCESSING_FAILED"
  | "VALIDATION_ERROR"
  | "NOT_FOUND"
  | "UNAUTHORIZED"
  | "INTERNAL_ERROR";

export interface Study {
  id: string;
  modality: Modality;
  body_part: string | null;
  status: StudyStatus;
  created_at: string;
  updated_at: string;
}

export interface StudyImage {
  id: string;
  study_id: string;
  original_filename: string | null;
  content_type: string;
  width: number;
  height: number;
  byte_size: number;
  sha256: string;
  created_at: string;
}

export interface Job {
  id: string;
  study_id: string;
  job_type: string;
  status: JobStatus;
  progress: number;
  error_code: ErrorCode | null;
  created_at: string;
  completed_at: string | null;
}

export interface StudyStatusPayload {
  study_id: string;
  status: StudyStatus;
  progress: number;
  job: Job | null;
  error_code: ErrorCode | null;
  message: string | null;
}

export interface Prediction {
  prediction: PredictionLabel;
  model_version: string;
  /** Absent on abstention — the API omits it rather than sending null. */
  confidence?: number;
  reason?: string;
  message?: string;
}

export interface Results {
  study_id: string;
  status: StudyStatus;
  result: Prediction;
  disclaimer: string;
}

export class ApiError extends Error {
  readonly code: ErrorCode;

  constructor(code: ErrorCode, message: string) {
    super(message);
    this.name = "ApiError";
    this.code = code;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...init,
    // Every one of these reads per-study state that changes between requests.
    cache: "no-store",
    headers: { "Content-Type": "application/json", ...init?.headers },
  });

  if (!response.ok) {
    const body = await response.json().catch(() => null);
    const error = (body as { error?: { code?: string; message?: string } } | null)?.error;
    throw new ApiError(
      (error?.code as ErrorCode) ?? "INTERNAL_ERROR",
      error?.message ?? "An unexpected error occurred.",
    );
  }

  return response.json() as Promise<T>;
}

export function listStudies(): Promise<Study[]> {
  return request<Study[]>("/api/v1/studies");
}

export function getStudy(id: string): Promise<Study> {
  return request<Study>(`/api/v1/studies/${id}`);
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

/**
 * Multipart, so the JSON content-type header from `request` would corrupt it:
 * the boundary has to be the one `fetch` generates for this FormData.
 */
export async function uploadImage(studyId: string, file: File): Promise<StudyImage> {
  const form = new FormData();
  form.append("file", file);

  const response = await fetch(`${API_BASE_URL}/api/v1/studies/${studyId}/upload`, {
    method: "POST",
    body: form,
    cache: "no-store",
  });

  if (!response.ok) {
    const body = await response.json().catch(() => null);
    const error = (body as { error?: { code?: string; message?: string } } | null)?.error;
    throw new ApiError(
      (error?.code as ErrorCode) ?? "INTERNAL_ERROR",
      error?.message ?? "The upload could not be processed.",
    );
  }

  return response.json() as Promise<StudyImage>;
}

export function startAnalysis(studyId: string): Promise<Job> {
  return request<Job>(`/api/v1/studies/${studyId}/analyze`, { method: "POST" });
}

export function getStatus(studyId: string): Promise<StudyStatusPayload> {
  return request<StudyStatusPayload>(`/api/v1/studies/${studyId}/status`);
}

export function getResults(studyId: string): Promise<Results> {
  return request<Results>(`/api/v1/studies/${studyId}/results`);
}
