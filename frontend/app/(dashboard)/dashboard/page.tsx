"use client"

import { useQuery, keepPreviousData } from "@tanstack/react-query"
import { endpoints } from "@/lib/api"
import { type DashboardData, type LineupGap, type InjuryFlag, type WaiverTarget, type StreakPlayer } from "@/lib/types"
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card"
import { Alert, AlertTitle, AlertDescription } from "@/components/ui/alert"
import { Badge } from "@/components/ui/badge"
import { BudgetPanel } from "@/components/dashboard/budget-panel"
import {
  AlertCircle,
  TrendingUp,
  TrendingDown,
  Activity,
  Users,
  Star,
} from "lucide-react"

export default function DashboardPage() {
  const { data: response, isLoading, isFetching, isError, error: queryError } = useQuery({
    queryKey: ['dashboard'],
    queryFn: endpoints.getDashboard,
    staleTime: 2 * 60_000,
    refetchInterval: 5 * 60_000,
    placeholderData: keepPreviousData,
  })

  const { data: budgetResponse } = useQuery({
    queryKey: ['budget'],
    queryFn: endpoints.getBudget,
    staleTime: 5 * 60_000,
    refetchInterval: 10 * 60_000,
  })

  const dashboard: DashboardData | null = response?.success ? response.data : null
  // timestamp lives at the top-level envelope, not inside response.data
  const timestamp: string | undefined = response?.timestamp

  const error: string | null = isError || queryError
    ? (queryError instanceof Error ? queryError.message : "Failed to load dashboard")
    : response && !response.success ? "Failed to load dashboard data" : null

  if (isLoading && !dashboard) {
    return <DashboardSkeleton />
  }

  if (error) {
    return (
      <div className="container mx-auto py-8 px-4">
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertTitle>Error loading dashboard</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      </div>
    )
  }

  if (!dashboard) {
    return (
      <div className="container mx-auto py-8 px-4">
        <Alert>
          <AlertCircle className="h-4 w-4" />
          <AlertTitle>No data available</AlertTitle>
          <AlertDescription>Dashboard data could not be loaded.</AlertDescription>
        </Alert>
      </div>
    )
  }

  return (
    <div className="container mx-auto py-8 px-4">
      {isFetching && (
        <div className="mb-4 text-xs text-text-secondary flex items-center gap-1.5">
          <span className="inline-block h-2 w-2 rounded-full bg-accent-gold animate-pulse" />
          Refreshing…
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
          })}{" ET"}
          {!timestamp ? " (approx)" : ""}
        </p>
      </div>

      {/* Lineup status bar */}
      <div className="mb-6 p-4 bg-bg-surface border border-border-subtle rounded-lg flex items-center gap-4">
        <Activity className="h-4 w-4 text-accent-gold shrink-0" />
        <span className="text-sm text-text-secondary">
          Lineup: <span className="text-text-primary font-medium">{dashboard.lineup_filled_count}</span>
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

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Lineup Gaps */}
        <LineupGapsCard gaps={dashboard.lineup_gaps} />

        {/* Injury Flags */}
        <InjuryFlagsCard flags={dashboard.injury_flags} />

        {/* Waiver Targets */}
        <WaiverTargetsCard targets={dashboard.waiver_targets} />

        {/* Hot Streaks */}
        <StreaksCard
          hot={dashboard.hot_streaks}
          cold={dashboard.cold_streaks}
        />

        {/* Constraint Budget */}
        {budgetResponse?.budget && (
          <BudgetPanel budget={budgetResponse.budget} />
        )}

        {/* Two-Start Pitchers */}
        {dashboard.two_start_pitchers?.length > 0 && (
          <Card className="lg:col-span-2">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-text-primary text-sm">
                <Star className="h-4 w-4 text-accent-gold" />
                Two-Start Pitchers This Week
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-3">
                {dashboard.two_start_pitchers.map((p) => (
                  <div
                    key={`${p.name}-${p.game_date}`}
                    className="p-3 bg-bg-surface border border-border-subtle rounded-md hover:bg-bg-elevated transition-colors"
                  >
                    <p className="text-text-primary text-sm font-medium">{p.name}</p>
                    <p className="text-text-muted text-xs mt-0.5">
                      {p.team} vs {p.opponent || 'TBD'} · {p.game_date}
                    </p>
                    <p className="text-text-secondary text-xs mt-1">{p.reason}</p>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  )
}

// ─── Sub-components ───────────────────────────────────────────────────────────

function LineupGapsCard({ gaps }: { gaps: LineupGap[] }) {
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
          <ul className="space-y-2">
            {gaps.map((gap, i) => (
              <li key={i} className="flex items-start gap-2">
                <span
                  className={`mt-0.5 h-2 w-2 rounded-full shrink-0 ${
                    gap.severity === "critical"
                      ? "bg-status-lost"
                      : gap.severity === "warning"
                      ? "bg-status-bubble"
                      : "bg-text-muted"
                  }`}
                />
                <div>
                  <p className="text-text-secondary text-sm font-medium">{gap.position}</p>
                  <p className="text-text-tertiary text-xs">{gap.message}</p>
                  {gap.suggested_add && (
                    <p className="text-accent-gold text-xs mt-0.5">Add: {gap.suggested_add}</p>
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

function InjuryFlagsCard({ flags }: { flags: InjuryFlag[] }) {
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

function WaiverTargetsCard({ targets }: { targets: WaiverTarget[] }) {
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
                <div>
                  <div className="flex items-center gap-2">
                    <p className="text-text-secondary text-sm font-medium">{t.name}</p>
                    <span className="text-text-muted text-xs">{t.team}</span>
                    <span className="text-text-tertiary text-xs">
                      {t.positions.join(", ")}
                    </span>
                  </div>
                  <p className="text-text-tertiary text-xs mt-0.5">{t.reason}</p>
                  <p className="text-text-muted text-xs">
                    {(t.percent_owned ?? 0) > 0 ? `${t.percent_owned.toFixed(0)}% owned` : '— owned'}
                  </p>
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

function StreaksCard({ hot, cold }: { hot: StreakPlayer[]; cold: StreakPlayer[] }) {
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
                        {p.last_7_avg.toFixed(1)} avg/7d
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
                        {p.last_7_avg.toFixed(1)} avg/7d
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

// Loading skeleton
function DashboardSkeleton() {
  return (
    <div className="container mx-auto py-8 px-4">
      <div className="h-7 w-40 mb-2 bg-bg-elevated animate-pulse rounded" />
      <div className="h-4 w-56 mb-8 bg-bg-elevated animate-pulse rounded" />
      <div className="h-14 mb-6 bg-bg-elevated animate-pulse rounded-lg" />
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {[1, 2, 3, 4].map((i) => (
          <div key={i} className="h-52 bg-bg-elevated animate-pulse rounded-lg" />
        ))}
      </div>
    </div>
  )
}
