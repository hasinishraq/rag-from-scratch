import { AlertDialog } from "radix-ui";
import type { ReactNode } from "react";

import { Button } from "@/components/ui/button";

/** Asks before an action that cannot be undone. Escape and Cancel both leave things as they are. */
export function ConfirmDialog({
  open,
  title,
  children,
  confirmLabel,
  busy = false,
  onConfirm,
  onCancel,
}: {
  open: boolean;
  title: string;
  children: ReactNode;
  confirmLabel: string;
  busy?: boolean;
  onConfirm: () => void;
  onCancel: () => void;
}) {
  return (
    <AlertDialog.Root open={open} onOpenChange={(next) => !next && !busy && onCancel()}>
      <AlertDialog.Portal>
        <AlertDialog.Overlay className="fixed inset-0 z-40 bg-ink/30" />
        <AlertDialog.Content className="fixed left-1/2 top-1/2 z-50 flex w-[min(28rem,calc(100vw-2rem))] -translate-x-1/2 -translate-y-1/2 flex-col gap-3 rounded-2xl border border-ink bg-paper p-6 outline-none">
          <AlertDialog.Title className="text-lg font-semibold">{title}</AlertDialog.Title>
          <AlertDialog.Description asChild>
            <div className="text-base text-graphite">{children}</div>
          </AlertDialog.Description>
          <div className="mt-3 flex justify-end gap-2">
            <AlertDialog.Cancel asChild>
              <Button variant="ghost" disabled={busy}>
                Cancel
              </Button>
            </AlertDialog.Cancel>
            <Button onClick={onConfirm} disabled={busy}>
              {busy ? "Removing" : confirmLabel}
            </Button>
          </div>
        </AlertDialog.Content>
      </AlertDialog.Portal>
    </AlertDialog.Root>
  );
}
