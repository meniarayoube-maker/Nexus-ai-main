// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Batch composition tracking is a session-only pilot diagnostic (default off).
// The UI toggle must reach the wire payload verbatim, and the store setter
// must resolve the packing conflict immediately (tracking on => packing off)
// so the backend never receives the refused combination from this UI.

import assert from "node:assert/strict";
import test from "node:test";

import {
  installLocalStorageFake,
  registerStoreStubResolver,
} from "./helpers/kit.ts";

import type { TrainingConfigState } from "../src/features/training/types/config.ts";

registerStoreStubResolver();
installLocalStorageFake();
const { buildTrainingStartPayload } = await import(
  "../src/features/training/api/mappers.ts"
);
const { initialTrainingConfigState } = await import(
  "../src/features/training/stores/training-config-policy.ts"
);
const { useTrainingConfigStore } = await import(
  "../src/features/training/stores/training-config-store.ts"
);

const CONFIG: TrainingConfigState = {
  ...initialTrainingConfigState,
  modelType: "text",
  selectedModel: "unsloth/gemma-3-270m-it",
  projectName: "composition",
  trainingMethod: "lora",
  datasetSource: "huggingface",
  datasetFormat: "auto",
  dataset: "unsloth/test",
  datasetSubset: null,
  datasetSplit: "train",
  datasetEvalSplit: null,
  datasetStreaming: false,
  datasetManualMapping: {},
  datasetSystemPrompt: "",
  datasetLabelMapping: {},
  datasetAdvisorNotification: null,
  datasetSliceStart: null,
  datasetSliceEnd: null,
  uploadedFile: null,
  uploadedEvalFile: null,
  epochs: 1,
  contextLength: 2048,
  learningRate: 2e-4,
  embeddingLearningRate: null,
  optimizerType: "adamw_8bit",
  lrSchedulerType: "linear",
  loraRank: 16,
  loraAlpha: 16,
  loraDropout: 0,
  loraVariant: "lora",
  batchSize: 2,
  gradientAccumulation: 4,
  weightDecay: 0.001,
  warmupSteps: 5,
  maxSteps: 60,
  saveSteps: 100,
  evalSteps: 0,
  packing: false,
  trackBatchComposition: false,
  trainOnCompletions: true,
  gradientCheckpointing: "unsloth",
  randomSeed: 3407,
  enableWandb: false,
  wandbToken: "",
  wandbProject: "",
  enableTensorboard: false,
  tensorboardDir: "",
  logFrequency: 1,
  isCheckingVision: false,
  isVisionModel: false,
  isEmbeddingModel: false,
  isAudioModel: false,
  isLoadingModelDefaults: false,
  modelDefaultsError: null,
  modelDefaultsAppliedFor: null,
  isCheckingDataset: false,
  isDatasetImage: false,
  isDatasetAudio: false,
  trustRemoteCode: false,
  approvedRemoteCodeFingerprint: null,
  finetuneVisionLayers: false,
  finetuneLanguageLayers: true,
  finetuneAttentionModules: true,
  finetuneMLPModules: true,
  targetModules: ["q_proj", "v_proj"],
  maxPositionEmbeddings: null,
  visionImageSize: null,
  s3Config: null,
};

test("composition tracking defaults off in state and on the wire", () => {
  assert.equal(initialTrainingConfigState.trackBatchComposition, false);

  const payload = buildTrainingStartPayload(CONFIG, null);
  assert.equal(payload.track_batch_composition, false);
});

test("an enabled toggle reaches the wire payload verbatim", () => {
  const payload = buildTrainingStartPayload(
    { ...CONFIG, trackBatchComposition: true },
    null,
  );

  assert.equal(payload.track_batch_composition, true);
});

test("enabling tracking turns packing off in the store", () => {
  useTrainingConfigStore.setState({ packing: true, trackBatchComposition: false });

  useTrainingConfigStore.getState().setTrackBatchComposition(true);

  const state = useTrainingConfigStore.getState();
  assert.equal(state.trackBatchComposition, true);
  assert.equal(
    state.packing,
    false,
    "packing must resolve off or the backend refuses the run",
  );

  useTrainingConfigStore.getState().setTrackBatchComposition(false);
  assert.equal(useTrainingConfigStore.getState().trackBatchComposition, false);
});
