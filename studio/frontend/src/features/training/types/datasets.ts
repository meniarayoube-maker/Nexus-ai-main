// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export type CheckFormatResponse = {
  requires_manual_mapping: boolean;
  detected_format: string;
  columns: string[];
  suggested_mapping?: Record<string, string> | null;
  detected_image_column?: string | null;
  detected_audio_column?: string | null;
  detected_text_column?: string | null;
  detected_speaker_column?: string | null;
  chat_column?: string | null;
  preview_samples?: Record<string, unknown>[] | null;
  total_rows?: number | null;
  is_image?: boolean;
  is_audio?: boolean;
  multimodal_columns?: string[] | null;
  warning?: string | null;
};

export type UploadDatasetResponse = {
  filename: string;
  stored_path: string;
};

export type RenderPreviewMetadataAudit = {
  column: string;
  value_preview: string;
  leaked_into_text: boolean;
};

export type RenderPreviewSample = {
  index: number;
  raw_fields: Record<string, unknown>;
  columns_in_row: string[];
  rendered_text: string;
  rendered_char_count: number;
  rendered_truncated: boolean;
  token_count_full: number;
  token_count_capped: number;
  cap_applied_at: number;
  input_ids_head: number[];
  input_ids_tail: number[];
  decoded_text: string;
  roundtrip_exact: boolean;
  metadata_audit: RenderPreviewMetadataAudit[];
};

export type RenderPreviewResponse = {
  success: boolean;
  samples: RenderPreviewSample[];
  columns_in: string[];
  columns_out: string[];
  detected_format: string;
  final_format: string;
  warnings: string[];
  errors: string[];
  tokenizer_source: string | null;
  total_rows: number | null;
};
