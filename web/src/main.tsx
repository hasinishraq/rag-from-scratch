import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import App from "@/App";
import { bayerTexture } from "@/components/Dither";
import "@/index.css";

// Bayer dot textures for CSS (button hover and press states), at three densities.
for (const [name, level] of [["sparse", 0.12], ["light", 0.25], ["half", 0.5]] as const) {
  document.documentElement.style.setProperty(`--dots-${name}`, bayerTexture(level));
}

const root = document.getElementById("root");
if (root) {
  createRoot(root).render(
    <StrictMode>
      <App />
    </StrictMode>,
  );
}
