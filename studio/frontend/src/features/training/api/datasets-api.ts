// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { hubTokenHeader } from "@/features/hub";
import { readFastApiError } from "@/lib/format-fastapi-error";
import type {
  CheckFormatResponse,
  RenderPreviewResponse,
  UploadDatasetResponse,
} from "../types/datasets";

type CheckDatasetFormatArgs = {
  datasetName: string;
  hfToken: string | null;
  subset?: string | null;
  split?: string | null;
  isVlm?: boolean;
  preferLocalCache?: boolean;
  localPath?: string | null;
  signal?: AbortSignal;
};

export class DatasetFormatError extends Error {
  readonly errorCode: string | null;
  readonly status: number;

  constructor(
    message: string,
    status: number,
    errorCode: string | null = null,
  ) {
    super(message);
    this.name = "DatasetFormatError";
    this.status = status;
    this.errorCode = errorCode;
  }
}

async function readDatasetFormatError(
  response: Response,
): Promise<DatasetFormatError> {
  const fallbackResponse = response.clone();
  try {
    const payload = (await response.json()) as { detail?: unknown };
    const detail = payload.detail;
    if (detail && typeof detail === "object" && !Array.isArray(detail)) {
      const structured = detail as { code?: unknown; message?: unknown };
      if (typeof structured.message === "string" && structured.message) {
        return new DatasetFormatError(
          structured.message,
          response.status,
          typeof structured.code === "string" ? structured.code : null,
        );
      }
    }
  } catch {
    return new DatasetFormatError(
      await readFastApiError(fallbackResponse),
      response.status,
    );
  }
  return new DatasetFormatError(
    await readFastApiError(fallbackResponse),
    response.status,
  );
}

export async function checkDatasetFormat({
  datasetName,
  hfToken,
  subset,
  split,
  isVlm,
  preferLocalCache,
  localPath,
  signal,
}: CheckDatasetFormatArgs): Promise<CheckFormatResponse> {
  const res = await authFetch("/api/hub/datasets/check-format", {
    method: "POST",
    signal,
    headers: {
      "Content-Type": "application/json",
      ...hubTokenHeader(hfToken),
    },
    body: JSON.stringify({
      dataset_name: datasetName,
      subset: subset || undefined,
      train_split: split || "train",
      is_vlm: !!isVlm,
      prefer_local_cache: !!preferLocalCache,
      local_path: localPath || undefined,
    }),
  });

  if (!res.ok) {
    throw await readDatasetFormatError(res);
  }

  return res.json();
}

export function uploadTrainingDataset(
  file: File,
): Promise<UploadDatasetResponse> {
  const form = new FormData();
  form.append("file", file);

  return uploadTrainingDatasetForm(form);
}

export function uploadNativeTrainingDataset(
  nativePathLease: string,
): Promise<UploadDatasetResponse> {
  const form = new FormData();
  form.append("nativePathLease", nativePathLease);

  return uploadTrainingDatasetForm(form);
}

async function uploadTrainingDatasetForm(
  form: FormData,
): Promise<UploadDatasetResponse> {
  const res = await authFetch("/api/hub/datasets/upload", {
    method: "POST",
    body: form,
  });

  if (!res.ok) {
    throw new Error(await readFastApiError(res, "Upload failed"));
  }

  return res.json();
}

// ── AI Assist ──

type AiAssistMappingArgs = {
  columns: string[];
  samples: Record<string, unknown>[];
  datasetName?: string | null;
  hfToken?: string | null;
  modelName?: string | null;
  modelType?: "text" | "vision" | "audio" | "embeddings" | null;
  signal?: AbortSignal;
};

export type AiAssistMappingResponse = {
  success: boolean;
  suggested_mapping?: Record<string, string> | null;
  warning?: string | null;
  // Conversion advisor fields
  system_prompt?: string | null;
  label_mapping?: Record<string, Record<string, string>> | null;
  dataset_type?: string | null;
  is_conversational?: boolean | null;
  user_notification?: string | null;
};

export async function aiAssistMapping({
  columns,
  samples,
  datasetName,
  hfToken,
  modelName,
  modelType,
  signal,
}: AiAssistMappingArgs): Promise<AiAssistMappingResponse> {
  const res = await authFetch("/api/hub/datasets/ai-assist-mapping", {
    method: "POST",
    signal,
    headers: {
      "Content-Type": "application/json",
      ...hubTokenHeader(hfToken),
    },
    body: JSON.stringify({
      columns,
      samples: samples.slice(0, 5),
      dataset_name: datasetName || undefined,
      model_name: modelName || undefined,
      model_type: modelType || undefined,
    }),
  });

  if (!res.ok) {
    throw new Error(await readFastApiError(res, "AI assist failed"));
  }

  return res.json();
}

// ── Render preview (exact training text) ──

export type PreviewTrainingRenderArgs = {
  datasetSource: "huggingface" | "upload";
  datasetName: string | null;
  subset?: string | null;
  split?: string | null;
  uploadedFile?: string | null;
  datasetFormat: string;
  customFormatMapping?: Record<string, unknown> | null;
  modelName: string | null;
  modelLocalPath?: string | null;
  hfToken: string | null;
  maxSeqLength: number;
  numSamples?: number;
  signal?: AbortSignal;
};

export async function previewTrainingRender({
  datasetSource,
  datasetName,
  subset,
  split,
  uploadedFile,
  datasetFormat,
  customFormatMapping,
  modelName,
  modelLocalPath,
  hfToken,
  maxSeqLength,
  numSamples = 3,
  signal,
}: PreviewTrainingRenderArgs): Promise<RenderPreviewResponse> {
  const res = await authFetch("/api/train/dataset-render-preview", {
    method: "POST",
    signal,
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      dataset_source: datasetSource,
      hf_dataset: datasetName || undefined,
      subset: subset || undefined,
      train_split: split || "train",
      local_datasets: uploadedFile ? [uploadedFile] : [],
      format_type: datasetFormat,
      custom_format_mapping: customFormatMapping ?? undefined,
      model_name: modelName,
      model_local_path: modelLocalPath || undefined,
      hf_token: hfToken,
      max_seq_length: maxSeqLength,
      num_samples: numSamples,
    }),
  });

  if (!res.ok) {
    throw new Error(await readFastApiError(res, "Render preview failed"));
  }

  return res.json();
}
