"use client";

import * as React from "react"
import { cn } from "@/lib/utils"

export interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: "default" | "outline" | "ghost" | "accent"
  size?: "default" | "sm" | "lg" | "icon"
}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant = "default", size = "default", children, ...props }, ref) => {
    
    return (
      <button
        ref={ref}
        className={cn(
          "relative inline-flex items-center justify-center whitespace-nowrap rounded-sm text-sm font-semibold tracking-widest uppercase focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-accent focus-visible:ring-offset-1 focus-visible:ring-offset-bg disabled:pointer-events-none disabled:opacity-50 transition-all duration-200 ease-out overflow-hidden group",
          {
            "bg-text text-bg hover:opacity-90 shadow-md": variant === "default",
            "border border-accent/60 bg-accent/5 text-accent hover:bg-accent hover:text-bg hover:shadow-[0_0_20px_rgba(5,255,170,0.3)]": variant === "accent",
            "border border-border bg-transparent hover:bg-card text-text hover:border-accent/40": variant === "outline",
            "hover:bg-card text-text": variant === "ghost",
            "h-10 px-6 py-2": size === "default",
            "h-8 px-4 text-xs": size === "sm",
            "h-12 px-10 text-sm": size === "lg",
            "h-10 w-10": size === "icon",
          },
          className
        )}
        {...props}
      >
        <span className="relative z-10 flex items-center gap-2">{children}</span>
        {variant === 'accent' && (
          <div className="absolute inset-0 z-0 bg-accent/20 blur-[8px] opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
        )}
      </button>
    )
  }
)
Button.displayName = "Button"

export { Button }
