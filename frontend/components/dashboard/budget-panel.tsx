"use client"

import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card"
import { DollarSign } from "lucide-react"
import type { BudgetData } from "@/lib/types"

interface BudgetPanelProps {
  budget: BudgetData
}

function paceColor(pace: BudgetData["ip_pace"]) {
  if (pace === "BEHIND") return "text-status-lost"
  if (pace === "AHEAD") return "text-status-safe"
  return "text-status-bubble"
}

function paceLabel(pace: BudgetData["ip_pace"]) {
  if (pace === "BEHIND") return "BEHIND"
  if (pace === "AHEAD") return "AHEAD"
  return "ON TRACK"
}

function acquisitionColor(used: number, limit: number) {
  if (limit === 0) return "bg-text-muted"
  const pct = used / limit
  if (pct >= 0.8) return "bg-status-lost"
  if (pct >= 0.5) return "bg-status-bubble"
  return "bg-status-safe"
}

function StatRow({
  label,
  value,
  sub,
}: {
  label: string
  value: React.ReactNode
  sub?: React.ReactNode
}) {
  return (
    <div className="flex items-center justify-between py-1.5 border-b border-border-subtle last:border-0">
      <span className="text-text-secondary text-xs">{label}</span>
      <div className="text-right">
        <span className="text-text-primary text-sm font-mono tabular-nums">{value}</span>
        {sub && <span className="ml-1.5 text-xs">{sub}</span>}
      </div>
    </div>
  )
}

export function BudgetPanel({ budget }: BudgetPanelProps) {
  const acqPct =
    budget.acquisition_limit > 0
      ? Math.min(100, (budget.acquisitions_used / budget.acquisition_limit) * 100)
      : 0

  const ipPct =
    budget.ip_minimum > 0
      ? Math.min(100, (budget.ip_accumulated / budget.ip_minimum) * 100)
      : 0

  return (
    <Card className="bg-bg-surface border-border-subtle">
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-text-primary text-sm">
          <DollarSign className="h-4 w-4 text-accent-gold" />
          Constraint Budget
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        {/* Acquisitions */}
        <div>
          <div className="flex items-center justify-between mb-1">
            <span className="text-text-secondary text-xs">Acquisitions</span>
            <span className="text-text-primary text-xs font-mono tabular-nums">
              {budget.acquisitions_used} / {budget.acquisition_limit}
            </span>
          </div>
          <div className="h-1.5 w-full bg-bg-inset rounded-full overflow-hidden">
            <div
              className={`h-full rounded-full transition-all ${acquisitionColor(
                budget.acquisitions_used,
                budget.acquisition_limit,
              )}`}
              style={{ width: `${acqPct}%` }}
            />
          </div>
          {budget.acquisition_warning && (
            <p className="text-status-bubble text-[10px] mt-0.5">Running low</p>
          )}
        </div>

        {/* IL Slots */}
        <StatRow
          label="IL Slots"
          value={`${budget.il_used} / ${budget.il_total}`}
          sub={
            budget.il_used >= budget.il_total ? (
              <span className="text-status-lost">Full</span>
            ) : (
              <span className="text-text-muted">{budget.il_total - budget.il_used} open</span>
            )
          }
        />

        {/* IP Progress */}
        <div>
          <div className="flex items-center justify-between mb-1">
            <span className="text-text-secondary text-xs">Innings Pitched</span>
            <span className={`text-xs font-semibold ${paceColor(budget.ip_pace)}`}>
              {paceLabel(budget.ip_pace)}
            </span>
          </div>
          <div className="h-1.5 w-full bg-bg-inset rounded-full overflow-hidden">
            <div
              className={`h-full rounded-full transition-all ${
                budget.ip_pace === "BEHIND"
                  ? "bg-red-500"
                  : budget.ip_pace === "AHEAD"
                  ? "bg-green-500"
                  : "bg-amber-400"
              }`}
              style={{ width: `${ipPct}%` }}
            />
          </div>
          <div className="flex items-center justify-between mt-0.5">
            <span className="text-text-muted text-[10px]">
              {budget.ip_accumulated.toFixed(1)} IP accumulated
            </span>
            <span className="text-text-muted text-[10px]">
              min {budget.ip_minimum.toFixed(0)} IP
            </span>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
