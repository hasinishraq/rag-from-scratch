import { useMemo } from "react";

import { bayerTexture, Dither, type Field } from "@/components/Dither";
import { cn } from "@/lib/utils";

// The current page is an ink block that breaks up into dots towards its right edge. It forms
// dot by dot when a page is opened (tone rising through the Bayer thresholds) and its edge
// keeps drifting slightly, like the imagery on the pages.
const SOLID_SHARE = 0.8; // left part that is fully ink, so the label always sits on solid ground
const FORM_MS = 420;
const HOVER_DOTS = bayerTexture(0.1);

function activeField(progress: () => number): Field {
  return (x, y, t, w) => {
    const u = x / w;
    const drift = Math.sin(y * 0.9 + t * 2.1) * 0.03 + Math.sin(u * 7 - t * 1.4) * 0.025;
    const tail = u < SOLID_SHARE ? 1 : Math.max(0, 1 - (u - SOLID_SHARE + drift) / (1 - SOLID_SHARE));
    return tail * progress();
  };
}

function ActiveBlock() {
  const field = useMemo(() => {
    const born = performance.now();
    const still = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const eased = () => {
      const p = still ? 1 : Math.min(1, (performance.now() - born) / FORM_MS);
      return 1 - (1 - p) ** 3;
    };
    return activeField(eased);
  }, []);
  return (
    <span className="absolute inset-0 overflow-hidden rounded-md" aria-hidden>
      <Dither field={field} cell={2} fps={24} />
    </span>
  );
}

export function SideNav<R extends string>({
  routes,
  current,
}: {
  routes: readonly { route: R; label: string }[];
  current: R;
}) {
  return (
    <nav className="flex shrink-0 gap-1 overflow-x-auto md:flex-col" aria-label="Pages">
      {routes.map((r, i) => {
        const active = r.route === current;
        return (
          <a
            key={r.route}
            href={`#/${r.route}`}
            aria-current={active ? "page" : undefined}
            className={cn(
              "nav-item group relative flex shrink-0 items-baseline gap-2.5 rounded-md py-2 pl-3 pr-9 text-base",
              active ? "font-medium text-paper" : "text-ink",
            )}
          >
            {active ? (
              <ActiveBlock key={r.route} />
            ) : (
              <span
                className="nav-wash absolute inset-0 rounded-md bg-ink"
                style={{ maskImage: HOVER_DOTS, WebkitMaskImage: HOVER_DOTS }}
                aria-hidden
              />
            )}
            <span
              className={cn(
                "relative hidden w-4 text-xs tabular-nums md:inline",
                active ? "text-paper/70" : "text-graphite group-hover:text-ink",
              )}
              aria-hidden
            >
              {String(i + 1).padStart(2, "0")}
            </span>
            <span className="relative">{r.label}</span>
          </a>
        );
      })}
    </nav>
  );
}
