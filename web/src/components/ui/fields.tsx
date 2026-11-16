import { Select as SelectPrimitive, Switch as SwitchPrimitive } from "radix-ui";
import { Check, ChevronDown } from "lucide-react";
import type * as React from "react";

import { cn } from "@/lib/utils";

const control =
  "w-full rounded-lg border border-rule bg-paper px-3 text-base outline-none transition-colors " +
  "placeholder:text-graphite focus-visible:border-ink disabled:opacity-50";

export function Input({ className, ...props }: React.ComponentProps<"input">) {
  return <input className={cn(control, "h-10", className)} {...props} />;
}

export function Textarea({ className, ...props }: React.ComponentProps<"textarea">) {
  return <textarea className={cn(control, "resize-none py-2.5", className)} {...props} />;
}

export function Field({
  label,
  hint,
  children,
  className,
}: {
  label: string;
  hint?: React.ReactNode;
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <label className={cn("flex flex-col gap-1.5", className)}>
      <span className="text-sm font-medium">{label}</span>
      {children}
      {hint ? <span className="text-sm text-graphite">{hint}</span> : null}
    </label>
  );
}

export function Switch({
  className,
  ...props
}: React.ComponentProps<typeof SwitchPrimitive.Root>) {
  return (
    <SwitchPrimitive.Root
      className={cn(
        "relative inline-flex h-5 w-9 shrink-0 items-center rounded-full border border-ink " +
          "bg-paper transition-colors data-[state=checked]:bg-ink",
        className,
      )}
      {...props}
    >
      <SwitchPrimitive.Thumb
        className={
          "block size-3.5 translate-x-0.5 rounded-full bg-ink transition-transform " +
          "data-[state=checked]:translate-x-[18px] data-[state=checked]:bg-paper"
        }
      />
    </SwitchPrimitive.Root>
  );
}

export function Select({
  value,
  onValueChange,
  options,
  placeholder,
  className,
}: {
  value: string;
  onValueChange: (v: string) => void;
  options: { value: string; label: string }[];
  placeholder?: string;
  className?: string;
}) {
  return (
    <SelectPrimitive.Root value={value || undefined} onValueChange={onValueChange}>
      <SelectPrimitive.Trigger
        className={cn(control, "flex h-10 items-center justify-between gap-2 text-left", className)}
      >
        <SelectPrimitive.Value placeholder={placeholder} />
        <SelectPrimitive.Icon>
          <ChevronDown className="size-4 text-graphite" />
        </SelectPrimitive.Icon>
      </SelectPrimitive.Trigger>
      <SelectPrimitive.Portal>
        <SelectPrimitive.Content
          position="popper"
          sideOffset={6}
          className={
            "z-50 max-h-80 min-w-[var(--radix-select-trigger-width)] overflow-hidden " +
            "rounded-lg border border-ink bg-paper text-ink"
          }
        >
          <SelectPrimitive.Viewport className="p-1">
            {options.map((o) => (
              <SelectPrimitive.Item
                key={o.value}
                value={o.value}
                className={
                  "relative flex cursor-default select-none items-center rounded-md py-2 pl-8 " +
                  "pr-3 text-base outline-none data-[highlighted]:bg-ink data-[highlighted]:text-paper"
                }
              >
                <SelectPrimitive.ItemIndicator className="absolute left-2.5">
                  <Check className="size-4" />
                </SelectPrimitive.ItemIndicator>
                <SelectPrimitive.ItemText>{o.label}</SelectPrimitive.ItemText>
              </SelectPrimitive.Item>
            ))}
          </SelectPrimitive.Viewport>
        </SelectPrimitive.Content>
      </SelectPrimitive.Portal>
    </SelectPrimitive.Root>
  );
}
