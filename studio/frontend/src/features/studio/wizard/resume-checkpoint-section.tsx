// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Resume from Checkpoint section — shown near model selection in the Training Wizard.
 *
 * Lists past training runs that can be resumed. When the user clicks Continue,
 * we call the existing resumeTrainingRun(runId) helper which:
 *   - loads the original config
 *   - sets resume_from_checkpoint
 *   - restarts training from the last completed step with the same settings
 *
 * The whole section hides itself when there are no resumable runs.
 */

import { Button } from "@/components/ui/button";
import { Spinner } from "@/components/ui/spinner";
import { listTrainingRuns } from "@/features/training/api/history-api";
import { resumeTrainingRun } from "@/features/training/lib/resume-training-run";
import type { TrainingRunSummary } from "@/features/training/types/history";
import { cn } from "@/lib/utils";
import { PlayIcon, RefreshIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import {
  type CSSProperties,
  useCallback,
  useEffect,
  useState,
} from "react";

function formatRelativeTime(iso: string | null): string {
  if (!iso) return "";
  const date = new Date(iso);
  const diffMs = Date.now() - date.getTime();
  const mins = Math.floor(diffMs / 60_000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins}m ago`;
  const hours = Math.floor(mins / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  return `${days}d ago`;
}

function statusLabel(status: TrainingRunSummary["status"]): string {
  switch (status) {
    case "stopped":
      return "Stopped";
    case "completed":
      return "Completed";
    case "error":
      return "Error";
    case "running":
      return "Running";
    default:
      return status;
  }
}

export function ResumeCheckpointSection() {
  const [runs, setRuns] = useState<TrainingRunSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [resumingId, setResumingId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const result = await listTrainingRuns(30, 0);
      // Only show runs that the backend says can be resumed
      const resumable = result.runs.filter((r) => r.can_resume);
      setRuns(resumable);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load runs");
      setRuns([]);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const handleResume = async (runId: string) => {
    if (resumingId) return;
    setResumingId(runId);
    try {
      await resumeTrainingRun(runId);
    } finally {
      setResumingId(null);
    }
  };

  // Hide completely when empty and not loading / no error
  if (!loading && !error && runs.length === 0) {
    return null;
  }

  return (
    <section className="@container/train-card elevated-card flex flex-col gap-5 bg-card p-5 pb-6">
      <div className="flex flex-col items-stretch gap-3 @md/train-card:flex-row @md/train-card:items-center @md/train-card:justify-between">
        <div className="flex min-w-0 items-center gap-3">
          <span
            className="train-section-chip inline-flex size-9 shrink-0 items-center justify-center rounded-full"
            style={{ "--chip-tint": "var(--chart-2)" } as CSSProperties}
          >
            <HugeiconsIcon
              icon={PlayIcon}
              strokeWidth={1.5}
              className="size-[18px]"
            />
          </span>
          <div className="min-w-0">
            <h2 className="select-none text-ui-13p5 font-semibold leading-ui-18 tracking-[-0.012em] text-foreground">
              Resume from Checkpoint
            </h2>
            <p className="text-ui-11p5 leading-ui-15 text-muted-foreground/85">
              Continue a previous run from its last checkpoint with the same
              settings.
            </p>
          </div>
        </div>
        <Button
          type="button"
          variant="ghost"
          size="sm"
          onClick={() => void load()}
          disabled={loading}
          className="shrink-0 self-end @md/train-card:self-auto"
          aria-label="Refresh checkpoint list"
        >
          <HugeiconsIcon
            icon={RefreshIcon}
            className={cn("size-4", loading && "animate-spin")}
          />
        </Button>
      </div>

      {loading && (
        <div className="flex items-center gap-2 py-2 text-ui-12 text-muted-foreground">
          <Spinner className="size-4" />
          Loading checkpoints…
        </div>
      )}

      {error && (
        <div className="rounded-lg border border-destructive/30 bg-destructive/5 px-3 py-2 text-ui-12 text-destructive">
          {error}
          <Button
            type="button"
            variant="link"
            size="sm"
            className="ml-2 h-auto p-0"
            onClick={() => void load()}
          >
            Retry
          </Button>
        </div>
      )}

      {!loading && !error && runs.length > 0 && (
        <ul className="flex flex-col gap-2">
          {runs.map((run) => {
            const stepText =
              run.final_step != null && run.total_steps != null
                ? `Step ${run.final_step}/${run.total_steps}`
                : run.final_step != null
                  ? `Step ${run.final_step}`
                  : null;
            const isBusy = resumingId === run.id;

            return (
              <li
                key={run.id}
                className="flex flex-col gap-2 rounded-xl border border-border/60 bg-muted/20 px-3.5 py-3 @md/train-card:flex-row @md/train-card:items-center @md/train-card:justify-between"
              >
                <div className="min-w-0 flex-1">
                  <div className="truncate text-ui-13 font-medium text-foreground">
                    {run.display_name || run.model_name}
                  </div>
                  <div className="mt-0.5 flex flex-wrap items-center gap-x-2 gap-y-0.5 text-ui-11 text-muted-foreground">
                    {stepText && <span>{stepText}</span>}
                    {stepText && <span aria-hidden="true">·</span>}
                    <span>{statusLabel(run.status)}</span>
                    {run.ended_at && (
                      <>
                        <span aria-hidden="true">·</span>
                        <span>{formatRelativeTime(run.ended_at)}</span>
                      </>
                    )}
                    {run.project_name && (
                      <>
                        <span aria-hidden="true">·</span>
                        <span className="truncate">{run.project_name}</span>
                      </>
                    )}
                  </div>
                  {run.resume_blocked_reason && (
                    <p className="mt-1 text-ui-11 text-amber-600 dark:text-amber-400">
                      {run.resume_blocked_reason}
                    </p>
                  )}
                </div>

                <Button
                  type="button"
                  size="sm"
                  disabled={isBusy || !run.can_resume}
                  onClick={() => void handleResume(run.id)}
                  className="shrink-0 gap-1.5"
                >
                  {isBusy ? (
                    <>
                      <Spinner className="size-3.5" />
                      Resuming…
                    </>
                  ) : (
                    <>
                      <HugeiconsIcon icon={PlayIcon} className="size-3.5" />
                      Continue Training
                    </>
                  )}
                </Button>
              </li>
            );
          })}
        </ul>
      )}
    </section>
  );
}
