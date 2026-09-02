// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Free-form save path for training checkpoints.
 * Shown in the training wizard under model selection.
 */

import { Input } from "@/components/ui/input";
import { useCustomOutputPathStore } from "@/features/training/stores/custom-output-path-store";
import { Folder01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import type { CSSProperties } from "react";

export function CustomOutputPathSection() {
  const path = useCustomOutputPathStore((s) => s.path);
  const setPath = useCustomOutputPathStore((s) => s.setPath);

  return (
    <section className="@container/train-card elevated-card flex flex-col gap-4 bg-card p-5 pb-6">
      <div className="flex min-w-0 items-center gap-3">
        <span
          className="train-section-chip inline-flex size-9 shrink-0 items-center justify-center rounded-full"
          style={{ "--chip-tint": "var(--chart-3)" } as CSSProperties}
        >
          <HugeiconsIcon
            icon={Folder01Icon}
            strokeWidth={1.5}
            className="size-[18px]"
          />
        </span>
        <div className="min-w-0">
          <h2 className="select-none text-ui-13p5 font-semibold leading-ui-18 tracking-[-0.012em] text-foreground">
            Checkpoint save location
          </h2>
          <p className="text-ui-11p5 leading-ui-15 text-muted-foreground/85">
            Leave empty for the default Studio outputs folder. Absolute paths
            are allowed (e.g. Google Drive on Colab).
          </p>
        </div>
      </div>

      <div className="flex flex-col gap-2">
        <label
          htmlFor="custom-output-path"
          className="text-ui-11 font-medium uppercase tracking-[0.05em] text-muted-foreground/70"
        >
          Output directory
        </label>
        <Input
          id="custom-output-path"
          value={path}
          onChange={(e) => setPath(e.target.value)}
          placeholder="/content/drive/MyDrive/UnslothStudio/runs"
          className="font-mono text-ui-12"
          spellCheck={false}
          autoComplete="off"
        />
        <p className="text-ui-11 text-muted-foreground/80">
          Examples:{" "}
          <span className="font-mono">/content/drive/MyDrive/my-runs</span>
          {" · "}
          <span className="font-mono">D:\models\checkpoints</span>
          {" · "}
          relative folder under Studio outputs
        </p>
        <p className="text-ui-11 text-muted-foreground/80">
          Remote upload (Hugging Face Hub) uses your HF token after training via
          Export — set the token in Settings. Dataset load from S3 uses S3 keys
          in the dataset section; checkpoint folders themselves are written to
          this path on disk.
        </p>
      </div>
    </section>
  );
}
