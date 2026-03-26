import * as React from "react"
import { cn } from "@/lib/utils"
import { Minus, Plus } from "lucide-react"

export interface StepperProps {
  value: number
  min?: number
  max?: number
  onChange: (value: number) => void
  disabled?: boolean
  className?: string
}

const Stepper = React.forwardRef<HTMLDivElement, StepperProps>(
  ({ value, min = 0, max = 100, onChange, disabled = false, className }, ref) => {
    return (
      <div
        ref={ref}
        className={cn("flex items-center space-x-2 bg-card border border-border rounded-xl p-1", className)}
      >
        <button
          type="button"
          onClick={() => value > min && onChange(value - 1)}
          disabled={disabled || value <= min}
          className="h-8 w-8 rounded-lg flex items-center justify-center text-text hover:bg-muted/10 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          <Minus size={16} />
        </button>
        <div className="w-8 text-center font-medium text-text">{value}</div>
        <button
          type="button"
          onClick={() => value < max && onChange(value + 1)}
          disabled={disabled || value >= max}
          className="h-8 w-8 rounded-lg flex items-center justify-center text-text hover:bg-muted/10 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          <Plus size={16} />
        </button>
      </div>
    )
  }
)
Stepper.displayName = "Stepper"

export { Stepper }
