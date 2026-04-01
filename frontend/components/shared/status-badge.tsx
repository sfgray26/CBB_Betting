"use client"

import * as React from "react"
import * as Tooltip from "@radix-ui/react-tooltip"
import { Activity, AlertTriangle, XCircle, Ban } from "lucide-react"
import { cn } from "@/lib/utils"

interface StatusBadgeProps {
  status: string | null | undefined
  className?: string
  showTooltip?: boolean
}

export function StatusBadge({ status, className, showTooltip = true }: StatusBadgeProps) {
  const s = (status || "ACTIVE").toLowerCase()

  // Color & Icon mapping
  let bgColor = "bg-zinc-700"
  let textColor = "text-zinc-400"
  let Icon: React.ElementType | null = null
  let tooltipText = "Status unavailable from Yahoo"

  if (["active", "", "start", "probable"].includes(s)) {
    bgColor = "bg-emerald-500/15"
    textColor = "text-emerald-400"
    Icon = Activity
    if (s === "start") tooltipText = "Confirmed starter for today"
    else if (s === "probable") tooltipText = "Probable starting pitcher"
    else tooltipText = "In the lineup / no injury designation"
  } else if (["dtd", "questionable"].includes(s)) {
    bgColor = "bg-amber-500/15"
    textColor = "text-amber-400"
    Icon = AlertTriangle
    if (s === "dtd") tooltipText = "Day-to-Day — monitor before lock"
    else tooltipText = `Listed as ${s.toUpperCase()} — check beat reporters`
  } else if (["il", "il10", "il60", "out", "dl", "ir"].includes(s)) {
    bgColor = "bg-rose-500/15"
    textColor = "text-rose-400"
    Icon = XCircle
    if (s === "il10") tooltipText = "On the 10-day Injured List"
    else if (s === "il60") tooltipText = "On the 60-day Injured List"
    else if (s === "out") tooltipText = "Out — do not start"
    else tooltipText = "Injured List"
  } else if (s === "no_start" || s === "unknown") {
    tooltipText = "No game today"
  } else if (s === "bench") {
    Icon = Ban
    tooltipText = "On the bench for today's game"
  }

  // Display text rules
  let displayText = status?.toUpperCase() || "ACTIVE"
  if (s === "il10") displayText = "IL-10"
  else if (s === "il60") displayText = "IL-60"
  else if (["no_start", "unknown"].includes(s)) displayText = "NO START"
  else if (s === "dtd") displayText = "DTD"
  else if (s === "active" || s === "") displayText = "ACTIVE"
  else if (s === "probable") displayText = "STARTING"

  const badge = (
    <div
      className={cn(
        "inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-[10px] font-bold tracking-wide uppercase whitespace-nowrap",
        bgColor,
        textColor,
        className
      )}
    >
      {Icon && <Icon className="h-3 w-3" />}
      {displayText}
    </div>
  )

  if (!showTooltip) return badge

  return (
    <Tooltip.Provider delayDuration={100}>
      <Tooltip.Root>
        <Tooltip.Trigger asChild>
          <div className="inline-block cursor-help">{badge}</div>
        </Tooltip.Trigger>
        <Tooltip.Portal>
          <Tooltip.Content
            sideOffset={4}
            className="z-50 px-2.5 py-1.5 text-xs bg-zinc-800 text-zinc-200 border border-zinc-700 rounded-md shadow-lg animate-in fade-in zoom-in-95 duration-100"
          >
            {tooltipText}
            <Tooltip.Arrow className="fill-zinc-800" />
          </Tooltip.Content>
        </Tooltip.Portal>
      </Tooltip.Root>
    </Tooltip.Provider>
  )
}
