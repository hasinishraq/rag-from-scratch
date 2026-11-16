import { cva, type VariantProps } from "class-variance-authority";
import { Slot } from "radix-ui";
import type * as React from "react";

import { cn } from "@/lib/utils";

const buttonVariants = cva(
  "btn inline-flex shrink-0 items-center justify-center gap-2 whitespace-nowrap font-medium " +
    "disabled:pointer-events-none disabled:opacity-40 [&_svg]:size-4 [&_svg]:shrink-0",
  {
    variants: {
      variant: {
        solid: "btn-solid bg-ink text-paper",
        outline: "btn-outline border border-ink bg-transparent hover:text-paper",
        ghost: "btn-ghost bg-transparent",
        quiet: "bg-transparent text-graphite underline-offset-4 hover:text-ink hover:underline",
      },
      size: {
        sm: "h-8 rounded-md px-3 text-sm",
        md: "h-10 rounded-lg px-4 text-base",
        lg: "h-12 rounded-xl px-6 text-base",
        icon: "size-9 rounded-md",
      },
    },
    defaultVariants: { variant: "solid", size: "md" },
  },
);

export function Button({
  className,
  variant,
  size,
  asChild = false,
  ...props
}: React.ComponentProps<"button"> & VariantProps<typeof buttonVariants> & { asChild?: boolean }) {
  const Comp = asChild ? Slot.Root : "button";
  return <Comp className={cn(buttonVariants({ variant, size }), className)} {...props} />;
}
