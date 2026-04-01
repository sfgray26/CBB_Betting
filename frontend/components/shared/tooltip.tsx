"use client"

import * as React from "react"
import * as TooltipPrimitive from "@radix-ui/react-tooltip"
import { cn } from "@/lib/utils"

export function Tooltip({
  children,
  content,
  className,
}: {
  children: React.ReactNode
  content: React.ReactNode
  className?: string
}) {
  return (
    <TooltipPrimitive.Provider delayDuration={100}>
      <TooltipPrimitive.Root>
        <TooltipPrimitive.Trigger asChild>
          {children}
        </TooltipPrimitive.Trigger>
        <TooltipPrimitive.Portal>
          <TooltipPrimitive.Content
            sideOffset={4}
            className={cn(
              "z-50 px-2.5 py-1.5 text-xs bg-zinc-800 text-zinc-200 border border-zinc-700 rounded-md shadow-lg animate-in fade-in zoom-in-95 duration-100",
              className
            )}
          >
            {content}
            <TooltipPrimitive.Arrow className="fill-zinc-800" />
          </TooltipPrimitive.Content>
        </TooltipPrimitive.Portal>
      </TooltipPrimitive.Root>
    </TooltipPrimitive.Provider>
  )
}
