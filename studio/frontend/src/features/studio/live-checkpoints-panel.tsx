// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Live Checkpoints panel — during training:
 *  - lists expected checkpoint steps (from save_steps)
 *  - shows output directory
 *  - exposes Stop & Save / Stop without save (uses existing stop API)
 *
 * Note: HuggingFace Trainer does not support "save mid-step without stopping"
 * from outside the loop. Stop & Save is the supported way to force a checkpoint.
 */

import { Button } from "@/components/ui/button";
import { Spinner } from "@/components/ui/spinner";
import {
  getBatchComposition,
  getPerExampleLoss,
  useTrainingActions,
  useTrainingConfigStore,
  useTrainingRuntimeStore,
} from "@/features/training";
import type { PerExampleLossResponse } from "@/features/training";
import { downloadFile, isDownloadCancelled } from "@/lib/native-files";
import { cn } from "@/lib/utils";
import {
  Database02Icon,
  Download01Icon,
  FloppyDiskIcon,
  Folder01Icon,
  StopIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useEffect, useMemo, useRef, useState } from "react";
import { useShallow } from "zustand/react/shallow";

type LiveCheckpointsPanelProps = {
  currentStep: number;
  totalSteps: number;
  outputDir: string | null;
  isTrainingRunning: boolean;
  /** Run/job id used to fetch the batch-composition sidecar. */
  runId?: string | null;
  saveStepsOverride?: number | null;
  className?: string;
};

type CompositionState =
  | { status: "idle" }
  | { status: "loading" }
  | { status: "ready"; records: number }
  | { status: "empty" }
  | { status: "error"; message: string };

function buildCheckpointSteps(
  currentStep: number,
  totalSteps: number,
  saveSteps: number,
): number[] {
  if (saveSteps <= 0 || currentStep <= 0) {
    return [];
  }
  const steps: number[] = [];
  for (let s = saveSteps; s <= currentStep; s += saveSteps) {
    steps.push(s);
  }
  if (
    totalSteps > 0 &&
    currentStep >= totalSteps &&
    (steps.length === 0 || steps[steps.length - 1] !== totalSteps)
  ) {
    steps.push(totalSteps);
  }
  return steps;
}

export function LiveCheckpointsPanel({
  currentStep,
  totalSteps,
  outputDir,
  isTrainingRunning,
  runId,
  saveStepsOverride,
  className,
}: LiveCheckpointsPanelProps) {
  const { stopTrainingRun } = useTrainingActions();
  const stopRequested = useTrainingRuntimeStore((s) => s.stopRequested);
  const [busy, setBusy] = useState(false);
  const [composition, setComposition] = useState<CompositionState>({
    status: "idle",
  });
  const compositionAbort = useRef<AbortController | null>(null);

  // Fresh run/output dir => fresh composition state.
  useEffect(() => {
    compositionAbort.current?.abort();
    compositionAbort.current = null;
    setComposition({ status: "idle" });
  }, [runId, outputDir]);

  useEffect(
    () => () => {
      compositionAbort.current?.abort();
    },
    [],
  );

  const handleCompositionDownload = async () => {
    if (!runId || composition.status === "loading") return;
    compositionAbort.current?.abort();
    const controller = new AbortController();
    compositionAbort.current = controller;
    setComposition({ status: "loading" });
    try {
      const res = await getBatchComposition(runId, controller.signal);
      if (controller.signal.aborted || compositionAbort.current !== controller) {
        return;
      }
      if (!res.exists || res.records.length === 0) {
        setComposition({ status: "empty" });
        return;
      }
      // Full fidelity with the raw sidecar: every key the API serves is
      // written verbatim (metrics included), so a download always matches
      // the on-disk file instead of a stripped subset.
      const lines = res.records.map((record) =>
        JSON.stringify({
          step: record.step,
          micro_batches: record.micro_batches,
          micro_seqs: record.micro_seqs ?? [],
          micro_t: record.micro_t ?? [],
          micro_via_preflight: record.micro_via_preflight ?? [],
          row_ids: record.row_ids,
          num_micro_batches: record.num_micro_batches,
          num_rows: record.num_rows,
          loss: record.loss ?? null,
          smoothed_loss: record.smoothed_loss ?? null,
          grad_norm: record.grad_norm ?? null,
          learning_rate: record.learning_rate ?? null,
        }),
      );
      await downloadFile(
        `${lines.join("\n")}\n`,
        `batch_composition_${runId}.jsonl`,
        "application/jsonl",
      );
      setComposition({ status: "ready", records: res.total_records });
    } catch (err) {
      if (controller.signal.aborted || isDownloadCancelled(err)) {
        setComposition({ status: "idle" });
        return;
      }
      setComposition({
        status: "error",
        message: err instanceof Error ? err.message : "Download failed.",
      });
    }
  };

  const formSaveSteps = useTrainingConfigStore(
    useShallow((s) => s.saveSteps ?? 0),
  );
  const saveSteps =
    typeof saveStepsOverride === "number" && saveStepsOverride > 0
      ? saveStepsOverride
      : formSaveSteps > 0
        ? formSaveSteps
        : 0;

  const checkpoints = useMemo(
    () => buildCheckpointSteps(currentStep, totalSteps, saveSteps),
    [currentStep, totalSteps, saveSteps],
  );

  const nextCheckpoint =
    saveSteps > 0 && currentStep < totalSteps
      ? Math.ceil((currentStep + 1) / saveSteps) * saveSteps
      : null;

  const latest =
    checkpoints.length > 0 ? checkpoints[checkpoints.length - 1] : null;

  const handleStop = async (saveCheckpoint: boolean) => {
    if (busy || stopRequested || !isTrainingRunning) return;
    setBusy(true);
    useTrainingRuntimeStore.getState().setStopRequested(true);
    try {
      const ok = await stopTrainingRun(saveCheckpoint);
      if (!ok) {
        useTrainingRuntimeStore.getState().setStopRequested(false);
      }
    } catch {
      useTrainingRuntimeStore.getState().setStopRequested(false);
    } finally {
      setBusy(false);
    }
  };

  if (!isTrainingRunning && checkpoints.length === 0 && !outputDir) {
    return null;
  }
  if (currentStep <= 0 && !outputDir && !isTrainingRunning) {
    return null;
  }

  return (
    <section
      className={cn(
        "elevated-card flex flex-col gap-3 bg-card p-4 sm:p-5",
        className,
      )}
    >
      <div className="flex items-center gap-2.5">
        <span className="inline-flex size-8 shrink-0 items-center justify-center rounded-full bg-muted/60">
          <HugeiconsIcon
            icon={FloppyDiskIcon}
            strokeWidth={1.5}
            className="size-4 text-foreground/80"
          />
        </span>
        <div className="min-w-0">
          <h3 className="text-ui-13 font-semibold text-foreground">
            Checkpoints
          </h3>
          <p className="text-ui-11 text-muted-foreground/85">
            {saveSteps > 0
              ? `Auto-save every ${saveSteps} steps · or stop & save now`
              : "Stop & save to force a checkpoint"}
          </p>
        </div>
      </div>

      {outputDir && (
        <div className="flex items-start gap-2 rounded-lg border border-border/50 bg-muted/15 px-3 py-2 text-ui-11">
          <HugeiconsIcon
            icon={Folder01Icon}
            className="mt-0.5 size-3.5 shrink-0 text-muted-foreground"
          />
          <div className="min-w-0 flex-1">
            <div className="text-muted-foreground">Save location</div>
            <div className="break-all font-mono text-ui-11 text-foreground/90">
              {outputDir}
            </div>
            <p className="mt-1 text-ui-10 text-muted-foreground/80">
              Checkpoints are folders like{" "}
              <span className="font-mono">checkpoint-70</span> inside this path.
            </p>
              {runId && (
                <div className="mt-2 flex items-center gap-2 flex-wrap">
                  <Button
                    type="button"
                    size="sm"
                    variant="outline"
                    disabled={composition.status === "loading"}
                    onClick={() => void handleCompositionDownload()}
                    className="gap-1.5 h-7 text-ui-11"
                  >
                    {composition.status === "loading" ? (
                      <Spinner className="size-3.5" />
                    ) : (
                      <HugeiconsIcon icon={Download01Icon} className="size-3.5" />
                    )}
                    {composition.status === "loading"
                      ? "Loading…"
                      : "Batch composition"}
                  </Button>
                  {composition.status === "ready" && (
                    <span className="text-ui-11 text-muted-foreground">
                      {composition.records.toLocaleString()} steps downloaded
                    </span>
                  )}
                  {composition.status === "empty" && (
                    <span className="text-ui-11 text-muted-foreground">
                      No composition tracked for this run (tracking is opt-in).
                    </span>
                  )}
                  {composition.status === "error" && (
                    <span className="text-ui-11 text-destructive">
                      {composition.message}
                    </span>
                  )}
                </div>
              )}
              {runId && outputDir && (
                <PerExampleLossSection
                  key={`per-example-${runId}`}
                  runId={runId}
                  outputDir={outputDir}
                />
              )}
          </div>
        </div>
      )}

      {checkpoints.length > 0 ? (
        <div className="flex flex-col gap-2">
          <div className="text-ui-11 text-muted-foreground">
            Saved so far ({checkpoints.length})
            {latest != null && (
              <span className="ml-1.5 font-medium text-foreground">
                · latest: checkpoint-{latest}
              </span>
            )}
          </div>
          <div className="flex flex-wrap gap-1.5">
            {checkpoints.map((step) => {
              const isLatest = step === latest;
              return (
                <span
                  key={step}
                  className={cn(
                    "inline-flex items-center rounded-md border px-2 py-0.5 font-mono text-ui-11",
                    isLatest
                      ? "border-primary/40 bg-primary/10 text-foreground"
                      : "border-border/60 bg-muted/30 text-muted-foreground",
                  )}
                >
                  checkpoint-{step}
                </span>
              );
            })}
          </div>
        </div>
      ) : (
        <p className="text-ui-12 text-muted-foreground">
          {saveSteps > 0
            ? `No checkpoint yet. First auto-save around step ${saveSteps}.`
            : "No checkpoint yet."}
        </p>
      )}

      {isTrainingRunning &&
        nextCheckpoint != null &&
        nextCheckpoint <= totalSteps && (
          <p className="text-ui-11 text-muted-foreground/90">
            Next auto-save at step{" "}
            <span className="font-medium text-foreground">{nextCheckpoint}</span>
            {totalSteps > 0 && (
              <span>
                {" "}
                ({Math.max(0, nextCheckpoint - currentStep)} steps left)
              </span>
            )}
          </p>
        )}

      {isTrainingRunning && (
        <div className="flex flex-col gap-2 border-t border-border/40 pt-3 sm:flex-row sm:flex-wrap">
          <Button
            type="button"
            size="sm"
            disabled={busy || stopRequested}
            onClick={() => void handleStop(true)}
            className="gap-1.5"
          >
            <HugeiconsIcon icon={FloppyDiskIcon} className="size-3.5" />
            {stopRequested || busy
              ? "Saving & stopping…"
              : "Stop & Save Checkpoint"}
          </Button>
          <Button
            type="button"
            size="sm"
            variant="outline"
            disabled={busy || stopRequested}
            onClick={() => void handleStop(false)}
            className="gap-1.5"
          >
            <HugeiconsIcon icon={StopIcon} className="size-3.5" />
            Stop without saving
          </Button>
          <p className="w-full text-ui-10 text-muted-foreground/80">
            Trainer only writes a full checkpoint at save_steps or when you stop
            with save. There is no “save and keep training” mid-step without
            stopping.
          </p>
        </div>
      )}
    </section>
  );
}

type PerExampleState =
  | { status: "idle" }
  | { status: "loading" }
  | { status: "ready"; data: PerExampleLossResponse }
  | { status: "empty"; outputPresent: boolean }
  | { status: "error"; message: string };

function formatCellValue(value: unknown, limit = 48): string {
  if (value === null || value === undefined) return "—";
  const text = typeof value === "string" ? value : JSON.stringify(value);
  return text.length > limit ? `${text.slice(0, limit)}…` : text;
}

function formatLoss(value: number | null | undefined): string {
  if (typeof value !== "number" || !Number.isFinite(value)) return "—";
  return value.toFixed(4);
}

/**
 * Per-Example Loss Attribution — read-only viewer over an already-generated
 * `per_example_loss.jsonl`. Never triggers scoring: when the file is absent
 * it says so explicitly (with the output status) instead of failing silently.
 * Wording is deliberately non-causal ("Highest per-example loss"): scores
 * measure example difficulty at the checkpoint, not causal proof.
 */
function PerExampleLossSection({
  runId,
  outputDir,
}: {
  runId: string;
  outputDir: string;
}) {
  const [state, setState] = useState<PerExampleState>({ status: "idle" });
  const [selectedStep, setSelectedStep] = useState<number | null>(null);
  const [search, setSearch] = useState("");
  const [ascending, setAscending] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    setState({ status: "idle" });
    setSelectedStep(null);
    setSearch("");
    setAscending(false);
    return () => {
      abortRef.current?.abort();
      abortRef.current = null;
    };
  }, [runId]);

  useEffect(
    () => () => {
      abortRef.current?.abort();
    },
    [],
  );

  const handleLoad = async () => {
    if (state.status === "loading") return;
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    setState({ status: "loading" });
    try {
      const res = await getPerExampleLoss(runId, controller.signal);
      if (controller.signal.aborted || abortRef.current !== controller) {
        return;
      }
      if (!res.exists || res.records.length === 0) {
        setState({ status: "empty", outputPresent: res.output_present });
        return;
      }
      setSelectedStep(null);
      setSearch("");
      setAscending(false);
      setState({ status: "ready", data: res });
    } catch (err) {
      if (controller.signal.aborted || isDownloadCancelled(err)) {
        setState({ status: "idle" });
        return;
      }
      setState({
        status: "error",
        message: err instanceof Error ? err.message : "Load failed.",
      });
    }
  };

  const handleDownload = async () => {
    if (state.status !== "ready") return;
    const lines = state.data.records.map((record) =>
      JSON.stringify({
        row_id: record.row_id,
        example_id: record.example_id,
        source: record.source,
        level: record.level,
        batch_id: record.batch_id,
        loss: record.loss ?? null,
        num_loss_tokens: record.num_loss_tokens ?? null,
        checkpoint: record.checkpoint ?? null,
        status: record.status ?? null,
        reason: record.reason ?? null,
      }),
    );
    try {
      await downloadFile(
        `${lines.join("\n")}\n`,
        `per_example_loss_${runId}.jsonl`,
        "application/jsonl",
      );
    } catch (err) {
      if (!isDownloadCancelled(err)) {
        setState({
          status: "error",
          message: err instanceof Error ? err.message : "Download failed.",
        });
      }
    }
  };

  const steps =
    state.status === "ready"
      ? [...state.data.steps].sort((a, b) => a.trainer_step - b.trainer_step)
      : [];

  // Default to the step holding the overall highest individual loss — the
  // most useful entry point for "which examples look hardest".
  const defaultStep = useMemo(() => {
    let bestStep: number | null = null;
    let bestLoss = Number.NEGATIVE_INFINITY;
    for (const stepView of steps) {
      for (const entry of stepView.entries) {
        if (
          typeof entry.individual_loss === "number" &&
          Number.isFinite(entry.individual_loss) &&
          entry.individual_loss > bestLoss
        ) {
          bestLoss = entry.individual_loss;
          bestStep = stepView.trainer_step;
        }
      }
    }
    return bestStep ?? steps[0]?.trainer_step ?? null;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [state]);

  const activeStep =
    selectedStep !== null &&
    steps.some((stepView) => stepView.trainer_step === selectedStep)
      ? selectedStep
      : defaultStep;
  const activeView = steps.find(
    (stepView) => stepView.trainer_step === activeStep,
  );

  const visibleEntries = useMemo(() => {
    if (!activeView) return [];
    const query = search.trim().toLowerCase();
    const filtered = query
      ? activeView.entries.filter((entry) => {
          const haystack =
            `${formatCellValue(entry.example_id, 200)} ${entry.row_id}`.toLowerCase();
          return haystack.includes(query);
        })
      : [...activeView.entries];
    filtered.sort((a, b) => {
      const left = a.individual_loss;
      const right = b.individual_loss;
      const leftValid = typeof left === "number" && Number.isFinite(left);
      const rightValid = typeof right === "number" && Number.isFinite(right);
      if (!leftValid && !rightValid) return 0;
      if (!leftValid) return 1;
      if (!rightValid) return -1;
      return ascending ? left - right : right - left;
    });
    return filtered;
  }, [activeView, search, ascending]);

  return (
    <div className="mt-2 rounded-lg border border-border/50 bg-muted/15 px-3 py-2 text-ui-11">
      <div className="flex items-center gap-2 flex-wrap">
        <HugeiconsIcon
          icon={Database02Icon}
          className="size-3.5 shrink-0 text-muted-foreground"
        />
        <span className="font-medium text-foreground">
          Per-Example Loss Attribution
        </span>
        <span className="text-ui-10 text-muted-foreground/80">
          Highest per-example loss — difficulty signal, not causal proof.
        </span>
      </div>

      {(state.status === "idle" || state.status === "error") && (
        <div className="mt-2 flex items-center gap-2 flex-wrap">
          <Button
            type="button"
            size="sm"
            variant="outline"
            onClick={() => void handleLoad()}
            className="gap-1.5 h-7 text-ui-11"
          >
            {state.status === "error" ? "Retry" : "Load results"}
          </Button>
          {state.status === "error" && (
            <span className="text-ui-11 text-destructive">{state.message}</span>
          )}
        </div>
      )}

      {state.status === "loading" && (
        <div className="mt-2 flex items-center gap-2 text-ui-11 text-muted-foreground">
          <Spinner className="size-3.5" />
          Loading…
        </div>
      )}

      {state.status === "empty" && (
        <div className="mt-2 space-y-1">
          <p className="text-ui-11 text-muted-foreground">
            Per-example loss has not been generated for this training output
            yet.
          </p>
          <p className="text-ui-10 text-muted-foreground/80 font-mono break-all">
            {state.outputPresent
              ? `Training output found at ${outputDir} — generate scores to enable this view.`
              : "No training output found for this run."}
          </p>
          <Button
            type="button"
            size="sm"
            variant="outline"
            onClick={() => void handleLoad()}
            className="gap-1.5 h-7 text-ui-11"
          >
            Retry
          </Button>
        </div>
      )}

      {state.status === "ready" && (
        <div className="mt-2 space-y-2">
          <div className="flex items-center gap-2 flex-wrap">
            <span className="text-ui-11 text-muted-foreground">
              {state.data.total_records.toLocaleString()} examples
              {steps.length > 0 &&
                ` · ${steps.length} attributed steps`}
            </span>
            <Button
              type="button"
              size="sm"
              variant="outline"
              onClick={() => void handleDownload()}
              className="gap-1.5 h-7 text-ui-11"
            >
              <HugeiconsIcon icon={Download01Icon} className="size-3.5" />
              Download per_example_loss.jsonl
            </Button>
          </div>

          {steps.length > 0 && activeView && (
            <div className="space-y-2">
              <div className="flex items-center gap-2 flex-wrap">
                <label
                  htmlFor={`per-example-step-${runId}`}
                  className="text-ui-11 text-muted-foreground"
                >
                  Step
                </label>
                <select
                  id={`per-example-step-${runId}`}
                  className="h-7 rounded-md border border-border/60 bg-background px-2 text-ui-11"
                  value={activeStep ?? ""}
                  onChange={(event) =>
                    setSelectedStep(Number(event.target.value))
                  }
                >
                  {steps.map((stepView) => (
                    <option
                      key={stepView.trainer_step}
                      value={stepView.trainer_step}
                    >
                      Step {stepView.trainer_step}
                      {typeof stepView.loss === "number" &&
                      Number.isFinite(stepView.loss)
                        ? ` — loss ${stepView.loss.toFixed(4)}`
                        : ""}
                    </option>
                  ))}
                </select>
                <input
                  type="text"
                  value={search}
                  onChange={(event) => setSearch(event.target.value)}
                  placeholder="Search example_id…"
                  className="h-7 w-44 rounded-md border border-border/60 bg-background px-2 text-ui-11 placeholder:text-muted-foreground/60"
                />
                <Button
                  type="button"
                  size="sm"
                  variant="ghost"
                  onClick={() => setAscending((value) => !value)}
                  className="h-7 text-ui-11"
                  title="Toggle loss sort direction"
                >
                  Loss {ascending ? "↑" : "↓"}
                </Button>
              </div>

              {visibleEntries.length > 0 ? (
                <table className="w-full border-collapse text-ui-11">
                  <thead>
                    <tr className="text-left text-muted-foreground">
                      <th className="border-b border-border/50 py-1 pr-2 font-medium">
                        Rank
                      </th>
                      <th className="border-b border-border/50 py-1 pr-2 font-medium">
                        Example ID
                      </th>
                      <th className="border-b border-border/50 py-1 font-medium">
                        Individual Loss
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {visibleEntries.map((entry, rank) => (
                      <tr
                        key={`${entry.row_id}-${rank}`}
                        className="border-b border-border/30 last:border-0"
                        title={`source: ${formatCellValue(entry.source)} · level: ${formatCellValue(entry.level)} · batch: ${formatCellValue(entry.batch_id)}`}
                      >
                        <td className="py-1 pr-2 font-mono text-muted-foreground">
                          {rank + 1}
                        </td>
                        <td className="py-1 pr-2 font-mono break-all">
                          {formatCellValue(entry.example_id)}
                        </td>
                        <td className="py-1 font-mono">
                          {formatLoss(entry.individual_loss)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              ) : (
                <p className="text-ui-11 text-muted-foreground">
                  No examples match this step and search.
                </p>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
