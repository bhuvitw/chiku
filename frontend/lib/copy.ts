/**
 * User-facing copy for the error contract (System Design §26).
 *
 * The backend sends both a code and a message; the message is the source of
 * truth and is rendered as-is. What lives here is only the *extra* framing
 * each state needs — what happened, and what the user can do about it — plus
 * whether a retry is worth offering at all.
 *
 * FAILED invites a retry, UNSUPPORTED does not (backend/models/enums.py), and
 * saying "try again" on a state that can never succeed is worse than silence.
 */

import type { ErrorCode, PredictionLabel } from "./api";

export interface ErrorPresentation {
  title: string;
  detail: string;
  retryable: boolean;
}

export const ERROR_COPY: Record<ErrorCode, ErrorPresentation> = {
  UNSUPPORTED_MODALITY: {
    title: "Unsupported image type",
    detail:
      "This prototype analyses PNG or JPEG X-rays only. DICOM is detected and refused rather than " +
      "read on a best-effort basis — export the image as PNG or JPEG and upload that.",
    retryable: false,
  },
  LOW_IMAGE_QUALITY: {
    title: "Image quality insufficient",
    detail:
      "The image was refused before the model ran: it is too small, or too uniform to carry the " +
      "detail a fracture would show. Nothing was analysed and no result was produced.",
    retryable: true,
  },
  LOW_CONFIDENCE: {
    title: "No confident prediction",
    detail:
      "The model's output fell inside the abstention band chosen on validation data, so it " +
      "declined to answer rather than report a number it could not stand behind.",
    retryable: false,
  },
  PROCESSING_FAILED: {
    title: "Analysis failed",
    detail: "Something went wrong while processing this study. No result was produced.",
    retryable: true,
  },
  VALIDATION_ERROR: {
    title: "Request could not be processed",
    detail: "The submitted request was not valid for this study.",
    retryable: true,
  },
  NOT_FOUND: {
    title: "Not found",
    detail: "This study does not exist, or it belongs to another account.",
    retryable: false,
  },
  UNAUTHORIZED: {
    title: "Sign-in required",
    detail: "This resource requires an authenticated session.",
    retryable: false,
  },
  INTERNAL_ERROR: {
    title: "Unexpected error",
    detail: "An unexpected error occurred. No result was produced.",
    retryable: true,
  },
};

export const PREDICTION_COPY: Record<
  PredictionLabel,
  { label: string; detail: string; tone: "alert" | "clear" | "unknown" }
> = {
  possible_fracture: {
    label: "Possible fracture",
    detail:
      "The model found features consistent with a fracture in this view. This is a " +
      "model output, not a diagnosis.",
    tone: "alert",
  },
  no_fracture: {
    label: "No fracture identified",
    detail:
      "The model did not find features consistent with a fracture in this view. A fracture " +
      "not visible in this projection would not be detected here.",
    tone: "clear",
  },
  unable_to_assess: {
    label: "Unable to assess",
    detail:
      "The model abstained rather than guess. This is a deliberate outcome, not a failure.",
    tone: "unknown",
  },
};

/**
 * PRD §12 and the Phase 1 evaluation. Shown on every non-abstained result.
 *
 * This is not generic hedging: the measured cast subgroup specificity was
 * 0.10 against 0.94 overall, which means the model calls almost every casted
 * wrist a fracture. A reader who does not know that will over-trust exactly
 * the images where it is least reliable.
 */
export const ARTIFACT_CAVEAT =
  "Measured on the held-out test split, this model is markedly less reliable on images " +
  "containing a cast or metal implant: it over-calls fractures on them (cast specificity " +
  "0.10 versus 0.94 overall). Treat a positive result on a casted or instrumented wrist " +
  "with particular caution.";

export const MODEL_SCOPE_CAVEAT =
  "Trained on paediatric wrist radiographs from a single hospital and one scanner " +
  "manufacturer. It has not been evaluated on adult imaging, other body parts, or other " +
  "equipment.";
