// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export interface TrainingRunSummary {
  id: string;
  status: "running" | "completed" | "stopped" | "error";
  model_name: string;
  project_name: string | null;
  dataset_name: string;
  display_name: string | null;
  started_at: string;
  ended_at: string | null;
  total_steps: number | null;
  final_step: number | null;
  final_loss: number | null;
  output_dir: string | null;
  can_resume: boolean;
  /** Why resume is unavailable, when the resource provenance is the cause. */
  resume_blocked_reason?: string | null;
  resumed_later: boolean;
  artifacts_available?: boolean;
  has_preview_model: boolean;
  preview_ref: string | null;
  preview_sig: string | null;
  duration_seconds: number | null;
  error_message: string | null;
  loss_sparkline: number[] | null;
}

export interface TrainingRunListResponse {
  runs: TrainingRunSummary[];
  total: number;
}

export interface TrainingRunMetrics {
  step_history: number[];
  loss_history: number[];
  loss_step_history: number[];
  lr_history: number[];
  lr_step_history: number[];
  grad_norm_history: number[];
  grad_norm_step_history: number[];
  eval_loss_history: number[];
  eval_step_history: number[];
  final_epoch: number | null;
  final_num_tokens: number | null;
}

export interface TrainingRunDetailResponse {
  run: TrainingRunSummary;
  config: Record<string, unknown>;
  metrics: TrainingRunMetrics;
}

export interface TrainingRunDeleteResponse {
  status: string;
  message: string;
  artifacts_deleted: boolean;
  artifacts_kept_reason: "shared_output_dir" | "purge_failed" | null;
}

export interface BatchCompositionRecord {
  step: number;
  micro_batches: number[][];
  micro_seqs: number[];
  row_ids: number[];
  num_micro_batches: number;
  num_rows: number;
  loss?: number | null;
  smoothed_loss?: number | null;
  grad_norm?: number | null;
  learning_rate?: number | null;
  micro_t?: number[];
  micro_via_preflight?: boolean[];
}

export interface BatchCompositionResponse {
  run_id: string;
  exists: boolean;
  records: BatchCompositionRecord[];
  total_records: number;
}

export interface PerExampleLossRecord {
  row_id: number | null;
  example_id: unknown;
  source: unknown;
  level: unknown;
  batch_id: unknown;
  loss: number | null;
  num_loss_tokens: number | null;
  checkpoint: string | null;
  status: string | null;
  reason: string | null;
}

export interface PerExampleStepEntry {
  row_id: number;
  example_id: unknown;
  source: unknown;
  level: unknown;
  batch_id: unknown;
  individual_loss: number | null;
  num_loss_tokens: number | null;
  status: string | null;
}

export interface PerExampleStepView {
  trainer_step: number;
  loss: number | null;
  rule: string;
  entries: PerExampleStepEntry[];
}

export interface PerExampleLossResponse {
  run_id: string;
  exists: boolean;
  output_present: boolean;
  records: PerExampleLossRecord[];
  total_records: number;
  steps: PerExampleStepView[];
  generation?: PerExampleGeneration | null;
}

export interface PerExampleGeneration {
  status: string;
  done: number;
  total: number;
  message: string;
}

export interface PerExampleGenerateResponse {
  run_id: string;
  accepted: boolean;
  status: string;
  message: string;
  job: PerExampleGeneration | null;
}
