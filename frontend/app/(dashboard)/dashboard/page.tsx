export const dynamic = "force-dynamic"

import { Suspense } from "react"
import { ErrorBoundary } from "@/components/error-boundary"
import {
  DashboardHeader,
  LineupStatusBar,
  LineupGapsWidget,
  InjuryFlagsWidget,
  WaiverTargetsWidget,
  StreaksWidget,
  BudgetWidget,
  ProbablePitchersWidget,
  TwoStartPitchersWidget,
} from "./_components/dashboard-client"
import {
  DashboardHeaderSkeleton,
  LineupStatusSkeleton,
  LineupGapsSkeleton,
  InjuryFlagsSkeleton,
  WaiverTargetsSkeleton,
  StreaksSkeleton,
  BudgetSkeleton,
  ProbablePitchersSkeleton,
  TwoStartPitchersSkeleton,
} from "./_components/widget-skeletons"

// ─── Widget error fallback ────────────────────────────────────────────────────

function WidgetError({ title }: { title: string }) {
  return (
    <div className="p-4 bg-bg-surface border border-status-lost/30 rounded-lg">
      <p className="text-status-lost text-sm font-medium">{title}</p>
      <p className="text-text-secondary text-xs mt-1">
        Failed to load. The widget will retry automatically.
      </p>
    </div>
  )
}

// ─── Dashboard Page (Server Component) ────────────────────────────────────────

export default function DashboardPage() {
  return (
    <div className="container mx-auto py-8 px-4">
      {/* Header + stale banner */}
      <ErrorBoundary fallback={<WidgetError title="Dashboard" />}>
        <Suspense fallback={<DashboardHeaderSkeleton />}>
          <DashboardHeader />
        </Suspense>
      </ErrorBoundary>

      {/* Lineup status bar */}
      <ErrorBoundary fallback={<WidgetError title="Lineup Status" />}>
        <Suspense fallback={<LineupStatusSkeleton />}>
          <LineupStatusBar />
        </Suspense>
      </ErrorBoundary>

      {/* Widget grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <ErrorBoundary fallback={<WidgetError title="Lineup Gaps" />}>
          <Suspense fallback={<LineupGapsSkeleton />}>
            <LineupGapsWidget />
          </Suspense>
        </ErrorBoundary>

        <ErrorBoundary fallback={<WidgetError title="Injury Alerts" />}>
          <Suspense fallback={<InjuryFlagsSkeleton />}>
            <InjuryFlagsWidget />
          </Suspense>
        </ErrorBoundary>

        <ErrorBoundary fallback={<WidgetError title="Waiver Targets" />}>
          <Suspense fallback={<WaiverTargetsSkeleton />}>
            <WaiverTargetsWidget />
          </Suspense>
        </ErrorBoundary>

        <ErrorBoundary fallback={<WidgetError title="Player Trends" />}>
          <Suspense fallback={<StreaksSkeleton />}>
            <StreaksWidget />
          </Suspense>
        </ErrorBoundary>

        <ErrorBoundary fallback={<WidgetError title="Constraint Budget" />}>
          <Suspense fallback={<BudgetSkeleton />}>
            <BudgetWidget />
          </Suspense>
        </ErrorBoundary>

        <ErrorBoundary fallback={<WidgetError title="Probable Pitchers" />}>
          <Suspense fallback={<ProbablePitchersSkeleton />}>
            <ProbablePitchersWidget />
          </Suspense>
        </ErrorBoundary>

        <ErrorBoundary fallback={<WidgetError title="Two-Start Pitchers" />}>
          <Suspense fallback={<TwoStartPitchersSkeleton />}>
            <TwoStartPitchersWidget />
          </Suspense>
        </ErrorBoundary>
      </div>
    </div>
  )
}
