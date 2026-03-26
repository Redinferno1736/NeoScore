"use client";

import * as React from "react"
import { cn } from "@/lib/utils"
import { motion, HTMLMotionProps } from "framer-motion"

export interface CardProps extends HTMLMotionProps<"div"> {}

const Card = React.forwardRef<HTMLDivElement, CardProps>(
  ({ className, style, ...props }, ref) => (
    <motion.div
      ref={ref}
      // Only animate transform — GPU-composited, zero repaint cost
      whileHover={{ y: -4 }}
      transition={{ duration: 0.4, ease: [0.16, 1, 0.3, 1] }}
      className={cn(
        "theme-card rounded-sm text-text relative",
        className
      )}
      style={{
        // Force GPU compositing layer on mount — prevents first-frame jank
        transform: "translateZ(0)",
        willChange: "transform",
        ...style,
      }}
      {...props}
    />
  )
)
Card.displayName = "Card"

export { Card }
