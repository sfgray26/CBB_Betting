"use client"

import { useEffect, useState } from "react"
import { api } from "@/lib/api"
import type { DashboardResponse, DashboardData, UserPreferences } from "@/lib/types"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Badge } from "@/components/ui/badge"
import { Skeleton } from "@/components/ui/skeleton"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import {
  AlertCircle,
  TrendingUp,
  TrendingDown,
  Users,
  Activity,
  Calendar,
  Zap,
  Trophy,
} from "lucide-react"

// Dashboard panel components
import { LineupGapsPanel } from "./components/LineupGapsPanel"
import { StreaksPanel } from "./components/StreaksPanel"
import { WaiverTargetsPanel } from "./components/WaiverTargetsPanel"
import { InjuryFlagsPanel } from "./components/InjuryFlagsPanel"
import { MatchupPreviewPanel } from "./components/MatchupPreviewPanel"
import { ProbablePitchersPanel } from "./components/ProbablePitchersPanel"

export default function DashboardPage() {
  const [dashboard, setDashboard] = useState<DashboardData | null>(null)
  const [preferences, setPreferences] = useState<UserPreferences | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    loadDashboard()
  }, [])

  async function loadDashboard() {
    try {
      setLoading(true)
      const response = await api.getDashboard()
      if (response.success) {
        setDashboard(response.data)
        setPreferences(response.preferences)
      } else {
        setError("Failed to load dashboard data")
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load dashboard")
    } finally {
      setLoading(false)
    }
  }

  if (loading) {
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
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold mb-2">Fantasy Baseball Dashboard</h1>
        <p className="text-muted-foreground">
          Last updated: {new Date(dashboard.timestamp).toLocaleString()}
        </p>
      </div>

      {/* Quick Stats Row */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
        <QuickStatCard
          title="Lineup Status"
          value={`${dashboard.lineup_filled_count}/${dashboard.lineup_total_count}`}
          description="Positions filled"
          icon={<Users className="h-4 w-4" />}
          trend={dashboard.lineup_gaps.length === 0 ? "good" : "warning"}
        />
        <QuickStatCard
          title="Healthy Players"
          value={`${dashboard.healthy_count}`}
          description={`${dashboard.injured_count} on IL/DL`}
          icon={<Activity className="h-4 w-4" />}
          trend={dashboard.injured_count === 0 ? "good" : "warning"}
        />
        <QuickStatCard
          title="Hot Streaks"
          value={`${dashboard.hot_streaks.length}`}
          description="Players trending up"
          icon={<TrendingUp className="h-4 w-4" />}
          trend="good"
        />
        <QuickStatCard
          title="Waiver Targets"
          value={`${dashboard.waiver_targets.length}`}
          description="Recommended adds"
          icon={<Zap className="h-4 w-4" />}
          trend="neutral"
        />
      </div>

      {/* Main Dashboard Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Lineup Gaps - Full Width Priority */}
        {dashboard.lineup_gaps.length > 0 && (
          <div className="lg:col-span-2">
            <LineupGapsPanel gaps={dashboard.lineup_gaps} />
          </div>
        )}

        {/* Left Column */}
        <div className="space-y-6">
          <StreaksPanel
            hotStreaks={dashboard.hot_streaks}
            coldStreaks={dashboard.cold_streaks}
          />
          <WaiverTargetsPanel targets={dashboard.waiver_targets} />
        </div>

        {/* Right Column */}
        <div className="space-y-6">
          <InjuryFlagsPanel
            flags={dashboard.injury_flags}
            healthyCount={dashboard.healthy_count}
            injuredCount={dashboard.injured_count}
          />
          <MatchupPreviewPanel preview={dashboard.matchup_preview} />
          <ProbablePitchersPanel
            pitchers={dashboard.probable_pitchers}
            twoStarters={dashboard.two_start_pitchers}
          />
        </div>
      </div>
    </div>
  )
}

// Quick Stat Card Component
function QuickStatCard({
  title,
  value,
  description,
  icon,
  trend,
}: {
  title: string
  value: string
  description: string
  icon: React.ReactNode
  trend: "good" | "warning" | "bad" | "neutral"
}) {
  const trendColors = {
    good: "text-green-600 bg-green-50",
    warning: "text-yellow-600 bg-yellow-50",
    bad: "text-red-600 bg-red-50",
    neutral: "text-blue-600 bg-blue-50",
  }

  return (
    <Card>
      <CardContent className="p-6">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-sm font-medium text-muted-foreground">{title}</p>
            <p className="text-2xl font-bold mt-1">{value}</p>
            <p className="text-xs text-muted-foreground mt-1">{description}</p>
          </div>
          <div className={`p-3 rounded-full ${trendColors[trend]}`}>
            {icon}
          </div>
        </div>
      </CardContent>
    </Card>
  )
}

// Loading Skeleton
function DashboardSkeleton() {
  return (
    <div className="container mx-auto py-8 px-4">
      <Skeleton className="h-10 w-64 mb-2" />
      <Skeleton className="h-4 w-48 mb-8" />

      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
        {[1, 2, 3, 4].map((i) => (
          <Skeleton key={i} className="h-24" />
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {[1, 2, 3, 4, 5, 6].map((i) => (
          <Skeleton key={i} className="h-64" />
        ))}
      </div>
    </div>
  )
}
