"use server";

import { redirect } from "next/navigation";
import { revalidatePath } from "next/cache";

import {
  ApiError,
  createStudy,
  getStatus,
  startAnalysis,
  uploadImage,
  type Modality,
  type StudyStatusPayload,
} from "@/lib/api";
import { getSession, signIn, signOut } from "@/lib/session";

async function requireSession(): Promise<void> {
  // Server Functions are reachable by direct POST, so the proxy's redirect is
  // not a guard — every action re-checks rather than assuming it ran.
  if ((await getSession()) === null) redirect("/login");
}

export async function signInAction(formData: FormData): Promise<void> {
  const email = String(formData.get("email") ?? "").trim();
  if (!email.includes("@")) {
    redirect("/login?error=invalid");
  }
  await signIn(email);
  const next = String(formData.get("next") ?? "/");
  // Only same-site paths: an attacker-supplied absolute URL here would turn
  // sign-in into an open redirect.
  redirect(next.startsWith("/") && !next.startsWith("//") ? next : "/");
}

export async function signOutAction(): Promise<void> {
  await signOut();
  redirect("/login");
}

export async function createStudyAction(formData: FormData): Promise<void> {
  await requireSession();
  const modality = String(formData.get("modality") ?? "");
  if (modality !== "xray" && modality !== "mri") {
    throw new Error("Unsupported modality");
  }

  const bodyPart = String(formData.get("body_part") ?? "").trim();

  await createStudy({
    modality: modality as Modality,
    body_part: bodyPart.length > 0 ? bodyPart : null,
  });

  revalidatePath("/");
}

/**
 * One dropped file becomes one study: create, upload, analyse, then hand the
 * caller its study page.
 *
 * Rejections come back as a value rather than a thrown error so the dropzone
 * can render the error contract's own message inline. An unexpected failure
 * still throws and reaches `error.tsx`.
 */
export async function uploadAndAnalyzeAction(
  formData: FormData,
): Promise<{ error: string } | void> {
  await requireSession();

  const file = formData.get("file");
  if (!(file instanceof File) || file.size === 0) {
    return { error: "No file was received. Try selecting the image again." };
  }

  let studyId: string;
  try {
    const study = await createStudy({ modality: "xray", body_part: "wrist" });
    studyId = study.id;
    await uploadImage(study.id, file);
  } catch (error) {
    if (error instanceof ApiError) return { error: error.message };
    throw error;
  }

  try {
    await startAnalysis(studyId);
  } catch (error) {
    // The study exists and the image is stored, so the failure belongs on the
    // study page with its error code, not swallowed into the dropzone.
    if (!(error instanceof ApiError)) throw error;
  }

  revalidatePath("/");
  redirect(`/studies/${studyId}`);
}

export async function reanalyzeAction(formData: FormData): Promise<void> {
  await requireSession();
  const studyId = String(formData.get("study_id") ?? "");
  try {
    await startAnalysis(studyId);
  } catch (error) {
    if (!(error instanceof ApiError)) throw error;
  }
  revalidatePath(`/studies/${studyId}`);
  redirect(`/studies/${studyId}`);
}

/** Polled by the processing view; returns the raw status envelope. */
export async function pollStatusAction(studyId: string): Promise<StudyStatusPayload | null> {
  await requireSession();
  try {
    return await getStatus(studyId);
  } catch {
    return null;
  }
}
