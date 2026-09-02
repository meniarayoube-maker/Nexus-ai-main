// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";

/**
 * Optional free-form checkpoint / run output directory.
 * Absolute paths (e.g. /content/drive/MyDrive/runs) are allowed by the backend
 * when using resolve_training_write_dir.
 */
type CustomOutputPathState = {
  path: string;
  setPath: (path: string) => void;
  clear: () => void;
};

export const useCustomOutputPathStore = create<CustomOutputPathState>((set) => ({
  path: "",
  setPath: (path) => set({ path }),
  clear: () => set({ path: "" }),
}));

export function getCustomOutputPath(): string | null {
  const trimmed = useCustomOutputPathStore.getState().path.trim();
  return trimmed.length > 0 ? trimmed : null;
}
