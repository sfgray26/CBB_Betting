"use client"

import { useState } from "react"
import { useSuspenseQuery, useIsFetching } from "@tanstack/react-query"
import { endpoints } from "@/lib/api"
import {
  type DashboardData,
  type LineupGap,
} from "@/lib/types"
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { BudgetPanel } from "@/components/dashboard/budget-panel"
import {
  AlertCircle,
  TrendingUp,
  TrendingDown,
  Activity,
  Users,
  Star,
  ArrowRight,
  Calendar,
  RefreshCw,
  X,
} from "lucide-react"
import Link from "next/link"
import { cn } from "@/lib/utils"
import { Tooltip } from "@/components/shared/tooltip"

// ─── Helpers ──────────────────────────────────────────────────────────────────

function formatRelativeTime(iso: string | null | undefined): string {
  if (!iso) return "unknown"
  const diff = Date.now() - new Date(iso).getTime()
  const minutes = Math.floor(diff / 60000)
  const hours = Math.floor(diff / 3600000)
  if (minutes < 1) return "just now"
  if (minutes < 60) return `${minutes}m ago`
  if (hours < 24) return `${hours}h ago`
  return `${Math.floor(hours / 24)}d ago`
}

function severityDotClass(severity: LineupGap["severity"]) {
  switch (severity) {
    case "critical":
      return "bg-status-lost"
    case "warning":
      return "bg-status-bubble"
    case "optimization":
      return "bg-accent-gold"
    default:
      return "bg-text-muted"
  }
}

function formatContributionKey(key: string): string {
  const labels: Record<string, string> = {
    HR_B: "HR",
    K_B: "K",
    K_P: "Ks",
    HR_P: "HRA",
    K_9: "K/9",
    NSB: "NSB",
    R: "R",
    H: "H",
    HR: "HR",
    RBI: "RBI",
    TB: "TB",
    AVG: "AVG",
    OPS: "OPS",
    W: "W",
    L: "L",
    ERA: "ERA",
    WHIP: "WHIP",
    QS: "QS",
    SV: "SV",
    NSV: "SV",
  }
  return `${labels[key] ?? key} fit`
}

function NeedScoreTierBadge({ score }: { score?: number }) {
  if (score == null) return null
  if (score >= 20) {
    return (
      <span className="text-[10px] px-1.5 py-0.5 bg-accent-gold/10 text-accent-gold border border-accent-gold/30 rounded font-semibold uppercase tracking-wider">
        PREMIUM
      </span>
    )
  }
  if (score >= 15) {
    return (
      <span className="text-[10px] px-1.5 py-0.5 bg-text-muted/10 text-text-secondary border border-text-muted/30 rounded font-semibold uppercase tracking-wider">
        STRONG
      </span>
    )
  }
  return null
}

function NeedScoreTooltipContent({
  score,
  contributions,
}: {
  score?: number
  contributions?: Record<string, number>
}) {
  if (score == null) return null
  const tier =
    score >= 20 ? "Premium target" : score >= 15 ? "Strong target" : "Standard target"
  const breakdown = contributions
    ? Object.entries(contributions)
        .map(([k, v]) => `${formatContributionKey(k)}: ${v >= 0 ? "+" : ""}${v.toFixed(1)}`)
        .join(" | ")
    : null

  return (
    <div className="space-y-1.5 max-w-[240px]">
      <p className="font-semibold text-text-primary">
        {score.toFixed(2)} — {tier}
      </p>
      {breakdown && <p className="text-text-secondary">{breakdown}</p>}
      <p className="text-text-muted text-[10px]">
        Scores range 0-30. {">"}20 = premium target
      </p>
    </div>
  )
}

// ─── Dashboard data hook (shared query key) ───────────────────────────────────

function useDashboardData() {
  return useSuspenseQuery({
    queryKey: ["dashboard"],
    queryFn: endpoints.getDashboard,
    staleTime: 2 * 60_000,
    refetchInterval: 5 * 60_000,
  })
}

// ─── Dashboard Header ─────────────────────────────────────────────────────────

export function DashboardHeader() {
  const [staleDismissed, setStaleDismissed] = useState(false)
  const { data: response } = useDashboardData()
  const isFetchingDashboard = useIsFetching({ queryKey: ["dashboard"] }) > 0

  const dashboard: DashboardData | null = response?.success ? response.data : null
  const timestamp: string | undefined = response?.timestamp

  const showStaleBanner =
    !staleDismissed &&
    dashboard?.stale_warning &&
    (dashboard?.has_mlb_games_today !== false)

  return (
    <>
      {isFetchingDashboard && (
        <div className="mb-4 text-xs text-text-secondary flex items-center gap-1.5">
          <span className="inline-block h-2 w-2 rounded-full bg-accent-gold animate-pulse" />
          Refreshing…
        </div>
      )}

      {showStaleBanner && (
        <div className="mb-4 bg-status-bubble/10 border border-status-bubble/30 rounded-lg p-3 flex items-start gap-3">
          <AlertCircle className="h-4 w-4 text-status-bubble flex-shrink-0 mt-0.5" />
          <div className="flex-1 min-w-0">
            <p className="text-sm text-status-bubble">
              ⚠️ Data may be stale — last updated {formatRelativeTime(dashboard?.last_sync)}.
              Starting lineups may have changed.
            </p>
          </div>
          <button
            onClick={() => setStaleDismissed(true)}
            className="text-status-bubble hover:text-text-primary transition-colors flex-shrink-0"
            aria-label="Dismiss"
          >
            <X className="h-4 w-4" />
          </button>
        </div>
      )}

      <div className="mb-8">
        <h1 className="text-xl font-semibold text-text-primary mb-2">Dashboard</h1>
        <p className="text-text-secondary text-sm">
          Last updated:{" "}
          {new Date(timestamp ?? Date.now()).toLocaleString("en-US", {
            timeZone: "America/New_York",
            dateStyle: "short",
            timeStyle: "short",
          })}
          {" ET"}
          {!timestamp ? " (approx)" : ""}
        </p>
      </div>
    </>
  )
}

// ─── Lineup Status Bar ────────────────────────────────────────────────────────

export function LineupStatusBar() {
  const { data: response } = useDashboardData()
  const dashboard = response?.success ? response.data : null

  if (!dashboard) return null

  return (
    <div className="mb-6 p-4 bg-bg-surface border border-border-subtle rounded-lg flex items-center gap-4">
      <Activity className="h-4 w-4 text-accent-gold shrink-0" />
      <span className="text-sm text-text-secondary">
        Lineup:{" "}
        <span className="text-text-primary font-medium">{dashboard.lineup_filled_count}</span>
        {" / "}
        <span className="text-text-primary font-medium">{dashboard.lineup_total_count}</span>
        {" slots filled"}
      </span>
      {dashboard.healthy_count !== undefined && (
        <span className="ml-auto text-xs text-text-muted">
          {dashboard.healthy_count} healthy · {dashboard.injured_count} injured
        </span>
      )}
    </div>
  )
}

// ─── Lineup Gaps Widget ───────────────────────────────────────────────────────

export function LineupGapsWidget() {
  const { data: response } = useDashboardData()
  const gaps = response?.success ? response.data.lineup_gaps : []

  const regularGaps = gaps.filter((g) => g.severity !== "optimization")
  const optimizationGaps = gaps.filter((g) => g.severity === "optimization")

  return (
    <Card className="bg-bg-surface border-border-subtle">
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-text-primary text-sm">
          <Users className="h-4 w-4 text-accent-gold" />
          Lineup Gaps
          {gaps.length > 0 && (
            <Badge variant="volatile" className="ml-auto text-xs">
              {gaps.length}
            </Badge>
          )}
        </CardTitle>
      </CardHeader>
      <CardContent>
        {gaps.length === 0 ? (
          <p className="text-text-muted text-sm">No lineup gaps detected.</p>
        ) : (
          <div className="space-y-4">
            {regularGaps.length > 0 && (
              <ul className="space-y-2">
                {regularGaps.map((gap, i) =>
                  gap.position === 'ROSTER' ? (
                    <li key={i} className="flex items-start gap-2 p-2 bg-status-lost/5 border border-status-lost/20 rounded">
                      <AlertCircle className="mt-0.5 h-4 w-4 text-status-lost shrink-0" />
                      <div className="flex-1 min-w-0">
                        <p className="text-status-lost text-sm font-bold">{gap.message}</p>
                        {gap.action_url && (
                          <Link
                            href={gap.action_url}
                            className="inline-flex items-center gap-1 text-xs text-accent-gold mt-1 hover:underline"
                          >
                            Go to Roster <ArrowRight className="h-3 w-3" />
                          </Link>
                        )}
                      </div>
                    </li>
                  ) : (
                    <li key={i} className="flex items-start gap-2">
                      <span
                        className={`mt-0.5 h-2 w-2 rounded-full shrink-0 ${severityDotClass(gap.severity)}`}
                      />
                      <div className="flex-1 min-w-0">
                        <p className="text-text-secondary text-sm font-medium">{gap.position}</p>
                        <p className="text-text-tertiary text-xs">{gap.message}</p>
                        {gap.suggested_add && (
                          <p className="text-accent-gold text-xs mt-0.5">
                            Add: {gap.suggested_add}
                          </p>
                        )}
                      </div>
                    </li>
                  )
                )}
              </ul>
            )}

            {optimizationGaps.length > 0 && (
              <div className={regularGaps.length > 0 ? "pt-3 border-t border-border-subtle" : ""}>
                <p className="text-[10px] font-bold tracking-widest uppercase text-accent-gold mb-2">
                  Sub-Optimal Placement
                </p>
                <ul className="space-y-2">
                  {optimizationGaps.map((gap, i) => (
                    <li key={i} className="flex items-start gap-2">
                      <span className="mt-0.5 h-2 w-2 rounded-full shrink-0 bg-accent-gold" />
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2 flex-wrap">
                          <p className="text-text-secondary text-sm font-medium">{gap.position}</p>
                          <span className="text-[10px] px-1.5 py-0.5 bg-accent-gold/10 text-accent-gold border border-accent-gold/30 rounded font-semibold uppercase tracking-wider">
                            Optimize
                          </span>
                        </div>
                        <p className="text-text-tertiary text-xs">{gap.message}</p>
                        {gap.suggested_add && (
                          <p className="text-accent-gold text-xs mt-0.5">
                            Add: {gap.suggested_add}
                          </p>
                        )}
                      </div>
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        )}
        <div className="mt-3 pt-3 border-t border-border-subtle">
          <Link
            href="/war-room/roster"
            className="inline-flex items-center gap-1 text-[11px] text-text-secondary hover:text-text-primary transition-colors"
          >
            View Roster <ArrowRight className="h-3 w-3" />
          </Link>
        </div>
      </CardContent>
    </Card>
  )
}

// ─── Injury Flags Widget ──────────────────────────────────────────────────────

export function InjuryFlagsWidget() {
  const { data: response } = useDashboardData()
  const flags = response?.success ? response.data.injury_flags : []

  return (
    <Card className="bg-bg-surface border-border-subtle">
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-text-primary text-sm">
          <AlertCircle className="h-4 w-4 text-accent-gold" />
          Injury Alerts
          {flags.length > 0 && (
            <Badge variant="volatile" className="ml-auto text-xs">
              {flags.length}
            </Badge>
          )}
        </CardTitle>
      </CardHeader>
      <CardContent>
        {flags.length === 0 ? (
          <p className="text-text-muted text-sm">No active injury alerts.</p>
        ) : (
          <ul className="space-y-3">
            {flags.map((flag, i) => (
              <li key={i} className="flex items-start gap-2">
                <span
                  className={`mt-0.5 h-2 w-2 rounded-full shrink-0 ${
                    flag.severity === "critical"
                      ? "bg-status-lost"
                      : flag.severity === "warning"
                      ? "bg-status-bubble"
                      : "bg-text-muted"
                  }`}
                />
                <div>
                  <div className="flex items-center gap-2">
                    <p className="text-text-secondary text-sm font-medium">{flag.name}</p>
                    <Badge variant="secondary" className="text-[10px] px-1 py-0">
                      {flag.status}
                    </Badge>
                  </div>
                  {flag.injury_note && (
                    <p className="text-text-tertiary text-xs mt-0.5">{flag.injury_note}</p>
                  )}
                  <p className="text-status-lost text-xs mt-0.5">{flag.action_needed}</p>
                  {flag.estimated_return && (
                    <p className="text-text-muted text-xs">ETA: {flag.estimated_return}</p>
                  )}
                </div>
              </li>
            ))}
          </ul>
        )}
      </CardContent>
    </Card>
  )
}

// ─── Waiver Targets Widget (independent endpoint) ─────────────────────────────

export function WaiverTargetsWidget() {
  const { data: response } = useSuspenseQuery({
    queryKey: ["dashboard-waiver-targets"],
    queryFn: endpoints.getDashboardWaiverTargets,
    staleTime: 2 * 60_000,
    refetchInterval: 5 * 60_000,
  })
  const targets = response?.success ? response.targets : []

  return (
    <Card className="bg-bg-surface border-border-subtle">
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-text-primary text-sm">
          <TrendingUp className="h-4 w-4 text-accent-gold" />
          Top Waiver Targets
        </CardTitle>
      </CardHeader>
      <CardContent>
        {targets.length === 0 ? (
          <p className="text-text-muted text-sm">No waiver targets available.</p>
        ) : (
          <ul className="space-y-3">
            {targets.slice(0, 5).map((t, i) => (
              <li key={i} className="flex items-start justify-between gap-2">
                <div className="min-w-0">
                  <div className="flex items-center gap-2 flex-wrap">
                    <p className="text-text-secondary text-sm font-medium">{t.name}</p>
                    <NeedScoreTierBadge score={t.need_score} />
                    {(t.starts_this_week ?? 0) >= 2 && (
                      <span className="text-[10px] px-1.5 py-0.5 bg-status-safe/10 text-status-safe border border-status-safe/30 rounded font-semibold uppercase tracking-wider">
                        2-Start
                      </span>
                    )}
                    <span className="text-text-muted text-xs">{t.team}</span>
                    <span className="text-text-tertiary text-xs">
                      {t.positions.join(", ")}
                    </span>
                  </div>
                  <p className="text-text-tertiary text-xs mt-0.5">{t.reason}</p>
                  <div className="flex items-center gap-2 mt-0.5">
                    <p className="text-text-muted text-xs">
                      {(t.percent_owned ?? 0) > 0
                        ? `${t.percent_owned.toFixed(0)}% owned`
                        : "— owned"}
                    </p>
                    {t.need_score != null && (
                      <Tooltip
                        content={
                          <NeedScoreTooltipContent
                            score={t.need_score}
                            contributions={t.category_contributions}
                          />
                        }
                      >
                        <span className="text-[10px] text-text-secondary tabular-nums cursor-help underline decoration-dotted">
                          Need: {t.need_score.toFixed(2)}
                        </span>
                      </Tooltip>
                    )}
                    {t.small_sample && (
                      <span className="text-[10px] px-1.5 py-0.5 bg-status-bubble/10 text-status-bubble border border-status-bubble/30 rounded font-semibold">
                        ⚠️ Small Sample
                      </span>
                    )}
                  </div>
                </div>
                <Badge
                  variant="default"
                  className={`shrink-0 text-[10px] ${
                    t.tier === "must_add"
                      ? "border-status-lost text-status-lost"
                      : t.tier === "strong_add"
                      ? "border-status-bubble text-status-bubble"
                      : "border-text-muted text-text-muted"
                  }`}
                >
                  {t.tier.replace("_", " ")}
                </Badge>
              </li>
            ))}
          </ul>
        )}
      </CardContent>
    </Card>
  )
}

// ─── Streaks Widget (independent endpoint) ────────────────────────────────────

export function StreaksWidget() {
  const { data: response } = useSuspenseQuery({
    queryKey: ["dashboard-streaks"],
    queryFn: endpoints.getDashboardStreaks,
    staleTime: 2 * 60_000,
    refetchInterval: 5 * 60_000,
  })
  const hot = response?.success ? response.hot_streaks : []
  const cold = response?.success ? response.cold_streaks : []

  return (
    <Card className="bg-bg-surface border-border-subtle">
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-text-primary text-sm">
          <Activity className="h-4 w-4 text-accent-gold" />
          Player Trends
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {hot.length === 0 && cold.length === 0 ? (
          <p className="text-text-muted text-sm">No streak data available.</p>
        ) : (
          <>
            {hot.length > 0 && (
              <div>
                <p className="text-xs font-semibold text-text-muted uppercase tracking-wider mb-2 flex items-center gap-1">
                  <TrendingUp className="h-3 w-3 text-status-safe" /> Hot
                </p>
                <ul className="space-y-1.5">
                  {hot.slice(0, 3).map((p, i) => (
                    <li key={i} className="flex items-center justify-between">
                      <div>
                        <span className="text-text-secondary text-sm">{p.name}</span>
                        <span className="text-text-muted text-xs ml-2">{p.team}</span>
                      </div>
                      <span className="text-status-safe text-xs">
                        Δ {p.trend_score.toFixed(1)}
                      </span>
                    </li>
                  ))}
                </ul>
              </div>
            )}
            {cold.length > 0 && (
              <div>
                <p className="text-xs font-semibold text-text-muted uppercase tracking-wider mb-2 flex items-center gap-1">
                  <TrendingDown className="h-3 w-3 text-status-behind" /> Cold
                </p>
                <ul className="space-y-1.5">
                  {cold.slice(0, 3).map((p, i) => (
                    <li key={i} className="flex items-center justify-between">
                      <div>
                        <span className="text-text-secondary text-sm">{p.name}</span>
                        <span className="text-text-muted text-xs ml-2">{p.team}</span>
                      </div>
                      <span className="text-status-behind text-xs">
                        Δ {p.trend_score.toFixed(1)}
                      </span>
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </>
        )}
      </CardContent>
    </Card>
  )
}

// ─── Budget Widget (independent endpoint) ─────────────────────────────────────

export function BudgetWidget() {
  const { data: response } = useSuspenseQuery({
    queryKey: ["budget"],
    queryFn: endpoints.getBudget,
    staleTime: 5 * 60_000,
    refetchInterval: 10 * 60_000,
  })

  if (!response?.budget) return null

  return <BudgetPanel budget={response.budget} />
}

// ─── Probable Pitchers Widget ─────────────────────────────────────────────────

export function ProbablePitchersWidget() {
  const { data: response, refetch } = useDashboardData()
  const pitchers = response?.success ? response.data.probable_pitchers : []

  return (
    <Card className="lg:col-span-2">
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-text-primary text-sm">
          <Calendar className="h-4 w-4 text-accent-gold" />
          Probable Pitchers
        </CardTitle>
      </CardHeader>
      <CardContent>
        {pitchers.length === 0 ? (
          <div className="space-y-3">
            <p className="text-status-bubble text-sm flex items-center gap-2">
              <AlertCircle className="h-4 w-4 flex-shrink-0" />
              ⚠️ Pitcher data temporarily unavailable. Your pitching slots are still active on Yahoo.
            </p>
            <button
              onClick={() => refetch()}
              className="inline-flex items-center gap-1.5 text-[11px] text-text-secondary hover:text-text-primary transition-colors"
            >
              <RefreshCw className="h-3 w-3" /> Retry
            </button>
          </div>
        ) : (
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-3">
            {pitchers.map((p) => (
              <div
                key={`${p.name}-${p.game_date}`}
                className="p-3 bg-bg-surface border border-border-subtle rounded-md hover:bg-bg-elevated transition-colors"
              >
                <p className="text-text-primary text-sm font-medium">{p.name}</p>
                <p className="text-text-muted text-xs mt-0.5">
                  {p.team} vs {p.opponent || "TBD"} · {p.game_date}
                </p>
                <div className="flex items-center gap-2 mt-1">
                  <span
                    className={cn(
                      "text-[10px] px-1.5 py-0.5 rounded font-semibold",
                      p.matchup_quality === "favorable"
                        ? "bg-status-safe/10 text-status-safe"
                        : p.matchup_quality === "unfavorable"
                        ? "bg-status-lost/10 text-status-lost"
                        : "bg-status-bubble/10 text-status-bubble"
                    )}
                  >
                    {p.matchup_quality}
                  </span>
                  <span className="text-[10px] text-text-muted">
                    Stream: {p.stream_score.toFixed(1)}
                  </span>
                </div>
                <p className="text-text-secondary text-xs mt-1">{p.reason}</p>
              </div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  )
}

// ─── Two-Start Pitchers Widget ────────────────────────────────────────────────

export function TwoStartPitchersWidget() {
  const { data: response } = useDashboardData()
  const pitchers = response?.success ? response.data.two_start_pitchers : []

  if (pitchers.length === 0) return null

  return (
    <Card className="lg:col-span-2">
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-text-primary text-sm">
          <Star className="h-4 w-4 text-accent-gold" />
          Two-Start Pitchers This Week
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-3">
          {pitchers.map((p) => (
            <div
              key={`${p.name}-${p.game_date}`}
              className="p-3 bg-bg-surface border border-border-subtle rounded-md hover:bg-bg-elevated transition-colors"
            >
              <p className="text-text-primary text-sm font-medium">{p.name}</p>
              <p className="text-text-muted text-xs mt-0.5">
                {p.team} vs {p.opponent || "TBD"} · {p.game_date}
              </p>
              <p className="text-text-secondary text-xs mt-1">{p.reason}</p>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  )
}
