"use client";

import { usePathname } from "next/navigation";
import { motion } from "framer-motion";

export function AnimatedBackground() {
  const pathname = usePathname();

  // Landing page uses DarkVeil WebGL — skip the CSS background there
  if (pathname === "/") return null;

  // Reduce intensity on specific paths to aid readability
  const isFocusIntense = pathname.startsWith("/results") || pathname.startsWith("/home") || pathname.startsWith("/history") || pathname.startsWith("/loans");
  const opacityClass = isFocusIntense ? "opacity-10" : "opacity-40";

  return (
    <div className="fixed inset-0 overflow-hidden pointer-events-none -z-50 transition-opacity duration-1000 bg-bg">
      
      {/* Base soft radial glow */}
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,_var(--color-accent)_0%,_transparent_25%)] opacity-5" />

      {/* Organic shifting mesh gradient */}
      <div className={`absolute top-0 left-0 w-full h-[150%] transition-opacity duration-1000 ${opacityClass}`}>
         <div 
           className="absolute inset-0 animate-mesh mix-blend-screen opacity-30"
           style={{
             backgroundImage: "radial-gradient(at 10% 20%, color-mix(in srgb, var(--color-accent) 15%, transparent) 0px, transparent 50%), radial-gradient(at 90% 80%, color-mix(in srgb, var(--color-accent) 8%, transparent) 0px, transparent 50%), radial-gradient(at 50% 10%, color-mix(in srgb, var(--color-text) 5%, transparent) 0px, transparent 50%)",
             backgroundSize: "200% 200%",
             filter: "blur(80px)"
           }}
         />
      </div>

    </div>
  );
}
