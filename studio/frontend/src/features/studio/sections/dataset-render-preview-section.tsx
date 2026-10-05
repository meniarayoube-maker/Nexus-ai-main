// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Rendered training text (Pilot v1, text-only).
 *
 * Runs the exact formatting pipeline training runs over a few sample rows —
 * no training, no model weights — and shows per sample:
 *  1. the original dataset fields,
 *  2. the final text after the chat template (pre-tokenization),
 *  3. what actually enters the tokenizer (input_ids + counts + decode check),
 *  4. a metadata audit proving columns like example_id / source / level /
 *     batch_id did not leak into the training text.
 */

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Spinner } from "@/components/ui/spinner";
import {
  previewTrainingRender,
  type PreviewTrainingRenderArgs,
  type RenderPreviewResponse,
  type RenderPreviewSample,
} from "@/features/training";
import { FloppyDiskIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useCallback, useEffect, useRef, useState } from "react";

function truncate(value: string, limit: number): string {
  if (value.length <= limit) return value;
  return `${value.slice(0, limit)}… [${value.length - limit} more chars]`;
}

function formatRawValue(value: unknown): string {
  if (value === null || value === undefined) return "--";
  if (typeof value === "string") return truncate(value, 500);
  try {
    return truncate(JSON.stringify(value), 500);
  } catch {
    return String(value);
  }
}

function formatIds(ids: number[]): string {
  if (ids.length === 0) return "--";
  return ids.join(" ");
}

function SampleCard({ sample }: { sample: RenderPreviewSample }) {
  const leaked = sample.metadata_audit.filter((a) => a.leaked_into_text);
  const truncated =
    sample.token_count_capped < sample.token_count_full;
  return (
    <div className="rounded-xl ring-1 ring-border/60 bg-background/60 px-4 py-3 space-y-3">
      <div className="flex items-center gap-2 flex-wrap">
        <span className="font-heading text-ui-13 font-semibold">
          Sample #{sample.index + 1}
        </span>
        <Badge variant="outline" className="font-mono text-ui-11 h-5">
          {sample.token_count_full.toLocaleString()} tokens
          {truncated &&
            ` (capped at ${sample.token_count_capped.toLocaleString()})`}
        </Badge>
        <Badge
          variant="outline"
          className="text-ui-11 h-5"
          title="decode(encode(text)) === text"
        >
          roundtrip {sample.roundtrip_exact ? "✓" : "✗"}
        </Badge>
        {sample.metadata_audit.length > 0 &&
          (leaked.length === 0 ? (
            <Badge
              variant="outline"
              className="text-ui-11 h-5 border-emerald-300 text-emerald-700 dark:border-emerald-700 dark:text-emerald-400"
            >
              metadata clean ✓
            </Badge>
          ) : (
            <Badge
              variant="outline"
              className="text-ui-11 h-5 border-destructive/50 text-destructive"
            >
              ⚠ {leaked.length} metadata leak{leaked.length > 1 ? "s" : ""}
            </Badge>
          ))}
      </div>

      <div>
        <p className="text-ui-11 font-medium uppercase tracking-[0.05em] text-muted-foreground/70 mb-1">
          1 · Original fields
        </p>
        <dl className="space-y-1">
          {Object.entries(sample.raw_fields).map(([key, value]) => (
            <div key={key} className="flex gap-2 text-ui-12 min-w-0">
              <dt className="font-mono text-muted-foreground shrink-0 max-w-40 truncate">
                {key}:
              </dt>
              <dd className="font-mono break-all text-foreground/90">
                {formatRawValue(value)}
              </dd>
            </div>
          ))}
        </dl>
      </div>

      <div>
        <p className="text-ui-11 font-medium uppercase tracking-[0.05em] text-muted-foreground/70 mb-1">
          2 · Final text after chat template ({sample.rendered_char_count.toLocaleString()} chars
          {sample.rendered_truncated ? ", truncated for display" : ""})
        </p>
        <pre className="font-mono text-ui-12 leading-relaxed whitespace-pre-wrap break-words rounded-lg bg-muted/30 px-3 py-2 max-h-56 overflow-auto">
          {sample.rendered_text}
        </pre>
      </div>

      <div>
        <p className="text-ui-11 font-medium uppercase tracking-[0.05em] text-muted-foreground/70 mb-1">
          3 · Tokenizer input
        </p>
        <p className="font-mono text-ui-11 break-all rounded-lg bg-muted/30 px-3 py-2 max-h-28 overflow-auto">
          {formatIds(sample.input_ids_head)}
          {sample.input_ids_tail.length > 0 && (
            <span className="text-muted-foreground">
              {" … "}
              {formatIds(sample.input_ids_tail)}
            </span>
          )}
        </p>
        <p className="mt-1 text-ui-11 text-muted-foreground">
          decode check:{" "}
          <span className="font-mono">
            {truncate(sample.decoded_text, 200) || "--"}
          </span>
        </p>
      </div>

      {sample.metadata_audit.length > 0 && (
        <div>
          <p className="text-ui-11 font-medium uppercase tracking-[0.05em] text-muted-foreground/70 mb-1">
            4 · Metadata audit
          </p>
          <ul className="space-y-1">
            {sample.metadata_audit.map((entry) => (
              <li
                key={entry.column}
                className="flex items-center gap-2 text-ui-12"
              >
                <span className="font-mono text-muted-foreground">
                  {entry.column}
                </span>
                <span className="font-mono truncate text-foreground/80">
                  {truncate(entry.value_preview, 80)}
                </span>
                {entry.leaked_into_text ? (
                  <Badge
                    variant="outline"
                    className="text-ui-10 h-5 border-destructive/50 text-destructive shrink-0"
                  >
                    in training text ⚠
                  </Badge>
                ) : (
                  <Badge
                    variant="outline"
                    className="text-ui-10 h-5 border-emerald-300 text-emerald-700 dark:border-emerald-700 dark:text-emerald-400 shrink-0"
                  >
                    not in text ✓
                  </Badge>
                )}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

export function DatasetRenderPreviewSection({
  argsKey,
  args,
  disabledReason,
}: {
  argsKey: string;
  args: PreviewTrainingRenderArgs | null;
  disabledReason: string | null;
}) {
  const [result, setResult] = useState<RenderPreviewResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const controllerRef = useRef<AbortController | null>(null);

  // Drop stale results whenever the inputs (dataset / format / mapping / model) change.
  useEffect(() => {
    controllerRef.current?.abort();
    controllerRef.current = null;
    setResult(null);
    setError(null);
    setLoading(false);
  }, [argsKey]);

  useEffect(
    () => () => {
      controllerRef.current?.abort();
    },
    [],
  );

  const handleRender = useCallback(async () => {
    if (!args || loading) return;
    controllerRef.current?.abort();
    const controller = new AbortController();
    controllerRef.current = controller;
    setLoading(true);
    setError(null);
    try {
      const res = await previewTrainingRender({
        ...args,
        signal: controller.signal,
      });
      if (controller.signal.aborted || controllerRef.current !== controller) {
        return;
      }
      if (!res.success) {
        setResult(null);
        setError(
          res.errors.length > 0
            ? res.errors.join("; ")
            : "Render preview failed.",
        );
        return;
      }
      setResult(res);
    } catch (err) {
      if (controller.signal.aborted) return;
      setError(err instanceof Error ? err.message : "Render preview failed.");
    } finally {
      if (controllerRef.current === controller) {
        controllerRef.current = null;
        setLoading(false);
      }
    }
  }, [args, loading]);

  return (
    <div className="rounded-xl corner-squircle ring-1 ring-border/60 bg-muted/30 px-5 py-4 mb-4 space-y-3">
      <div className="flex items-center gap-2.5">
        <HugeiconsIcon icon={FloppyDiskIcon} className="size-4 text-foreground/70" />
        <div className="min-w-0">
          <h3 className="text-ui-13 font-semibold text-foreground">
            Rendered training text
          </h3>
          <p className="text-ui-11 text-muted-foreground/85">
            Exact text sent to the model after the chat template, before
            tokenization — same pipeline as training, no training started.
          </p>
        </div>
      </div>

      {disabledReason ? (
        <p className="text-ui-12 text-muted-foreground">{disabledReason}</p>
      ) : (
        <>
          <Button
            type="button"
            size="sm"
            variant="outline"
            disabled={loading || !args}
            onClick={() => void handleRender()}
            className="gap-1.5"
          >
            {loading && <Spinner className="size-3.5" />}
            {loading ? "Rendering…" : "Render 3 samples"}
          </Button>

          {error && (
            <div className="rounded-lg border border-destructive/30 bg-destructive/5 px-3 py-2 text-ui-12 text-destructive">
              {error}
            </div>
          )}

          {result && (
            <div className="space-y-3">
              <p className="text-ui-11 text-muted-foreground">
                Tokenizer:{" "}
                <span className="font-mono">
                  {result.tokenizer_source ?? "--"}
                </span>
                {" · "}Columns:{" "}
                <span className="font-mono">
                  {result.columns_in.join(", ") || "--"}
                </span>{" "}
                →{" "}
                <span className="font-mono">
                  {result.columns_out.join(", ") || "--"}
                </span>
                {" · "}Format: {result.final_format}
                {result.total_rows != null &&
                  ` · ${result.total_rows.toLocaleString()} rows total`}
              </p>
              {result.warnings.length > 0 && (
                <p className="text-ui-11 text-amber-600 dark:text-amber-400">
                  {result.warnings.join("; ")}
                </p>
              )}
              {result.samples.map((sample) => (
                <SampleCard key={sample.index} sample={sample} />
              ))}
            </div>
          )}
        </>
      )}
    </div>
  );
}
