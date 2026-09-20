"use server";

import { revalidatePath } from "next/cache";

import { createStudy, type Modality } from "@/lib/api";

export async function createStudyAction(formData: FormData): Promise<void> {
  const modality = String(formData.get("modality") ?? "");
  if (modality !== "xray" && modality !== "mri") {
    // Server Functions are reachable by direct POST, so the allowed set is
    // re-checked here rather than trusted from the form markup.
    throw new Error("Unsupported modality");
  }

  const bodyPart = String(formData.get("body_part") ?? "").trim();

  await createStudy({
    modality: modality as Modality,
    body_part: bodyPart.length > 0 ? bodyPart : null,
  });

  revalidatePath("/");
}
