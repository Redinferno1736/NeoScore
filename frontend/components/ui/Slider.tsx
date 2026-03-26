import * as React from "react"
import { cn } from "@/lib/utils"

export interface SliderProps extends Omit<React.InputHTMLAttributes<HTMLInputElement>, 'onChange'> {
  value: number;
  min: number;
  max: number;
  step?: number;
  onChange: (value: number) => void;
}

const Slider = React.forwardRef<HTMLInputElement, SliderProps>(
  ({ className, value, min, max, step = 1, onChange, ...props }, ref) => {
    return (
      <div className={cn("relative w-full flex items-center", className)}>
        <input
          type="range"
          ref={ref}
          min={min}
          max={max}
          step={step}
          value={value}
          onChange={(e) => onChange(Number(e.target.value))}
          className="w-full h-2 bg-muted/20 rounded-lg appearance-none cursor-pointer accent-accent focus:outline-none focus:ring-2 focus:ring-accent max-w-full"
          {...props}
        />
      </div>
    )
  }
)
Slider.displayName = "Slider"

export { Slider }
