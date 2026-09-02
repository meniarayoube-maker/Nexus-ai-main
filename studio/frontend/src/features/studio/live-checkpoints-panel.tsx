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
import {
  useTrainingActions,
  useTrainingConfigStore,
  useTrainingRuntimeStore,
} from "@/features/training";
import { cn } from "@/lib/utils";
import {
  FloppyDiskIcon,
  Folder01Icon,
  StopIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useMemo, useState } from "react";
import { useShallow } from "zustand/react/shallow";

type LiveCheckpointsPanelProps = {
  currentStep: number;
  totalSteps: number;
  outputDir: string | null;
  isTrainingRunning: boolean;
  saveStepsOverride?: number | null;
  className?: string;
};

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
  saveStepsOverride,
  className,
}: LiveCheckpointsPanelProps) {
  const { stopTrainingRun } = useTrainingActions();
  const stopRequested = useTrainingRuntimeStore((s) => s.stopRequested);
  const [busy, setBusy] = useState(false);

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
          <div className="min-w-0">
            <div className="text-muted-foreground">Save location</div>
            <div className="break-all font-mono text-ui-11 text-foreground/90">
              {outputDir}
            </div>
            <p className="mt-1 text-ui-10 text-muted-foreground/80">
              Checkpoints are folders like{" "}
              <span className="font-mono">checkpoint-70</span> inside this path.
            </p>
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
