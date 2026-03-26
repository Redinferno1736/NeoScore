"use client";

import dynamic from "next/dynamic";
import { useTheme } from "next-themes";
import { useState, useEffect } from "react";

// Dynamically import to avoid SSR issues with WebGL / canvas
const DarkVeil = dynamic(() => import("./DarkVeil"), { ssr: false });

export function LandingBackground() {
  const { resolvedTheme } = useTheme();
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  // Overlay color: Deep Forest for light, Obsidian Moss for dark
  // We use the exact brand hex codes with opacity so the WebGL animation bleeds through
  const overlayColor = mounted && resolvedTheme === "dark"
    ? "rgba(8, 20, 18, 0.78)"   // Obsidian Moss #081412 @ 78%
    : "rgba(16, 44, 38, 0.74)"; // Deep Forest  #102C26 @ 74%

  return (
    // Fixed full-viewport container, below everything (z-index -10)
    <div className="fixed inset-0 -z-10 overflow-hidden pointer-events-none">
      {/* WebGL canvas fills the container */}
      <DarkVeil
        speed={0.4}
        hueShift={145}
        warpAmount={0.3}
        noiseIntensity={0.02}
        scanlineIntensity={0}
        scanlineFrequency={0}
        resolutionScale={0.75}
      />

      {/* Brand tint overlay — sits on top of the canvas, below page content */}
      <div
        className="absolute inset-0 transition-colors duration-700"
        style={{ backgroundColor: overlayColor, mixBlendMode: "multiply" }}
      />

      {/* Secondary gradient fade at the bottom so cards blend into the bg */}
      <div className="absolute bottom-0 left-0 right-0 h-48 bg-gradient-to-t from-bg/80 to-transparent" />
    </div>
  );
}
