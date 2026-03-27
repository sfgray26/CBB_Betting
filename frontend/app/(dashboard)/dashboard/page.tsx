"use client"

import { useEffect, useState } from "react"
import { endpoints } from "@/lib/api"
import type { DashboardResponse, DashboardData, UserPreferences } from "@/lib/types"
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Alert, AlertTitle, AlertDescription } from "@/components/ui/alert"
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

// Dashboard panel components - simplified inline versions

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
      const response = await endpoints.getDashboard()
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
      
      {dashboard.lineup_gaps.length > 0 && (
        <div className="mb-6 p-4 bg-amber-50 border border-amber-200 rounded-lg">
          <h3 className="font-semibold text-amber-800 flex items-center gap-2">
            <AlertCircle className="h-5 w-5" />
            Lineup Gaps Detected: {dashboard.lineup_gaps.length} unfilled
          </h3>
        </div>
      )}

      {/* Main Dashboard Content */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Hot Streaks */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-orange-600">
              <TrendingUp className="h-5 w-5" />
              Hot Streaks ({dashboard.hot_streaks.length})
            </CardTitle>
          </CardHeader>
          <div className="p-6 pt-0">
            {dashboard.hot_streaks.length === 0 ? (
              <p className="text-muted-foreground">No hot streaks detected</p>
            ) : (
              <div className="space-y-2">
                {dashboard.hot_streaks.map((player) => (
                  <div key={player.player_id} className="flex justify-between p-2 bg-orange-50 rounded">
                    <span className="font-medium">{player.name}</span>
                    <span className="text-orange-600">+{player.trend_score.toFixed(2)} z</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        </Card>

        {/* Cold Streaks */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-blue-600">
              <TrendingDown className="h-5 w-5" />
              Cold Streaks ({dashboard.cold_streaks.length})
            </CardTitle>
          </CardHeader>
          <div className="p-6 pt-0">
            {dashboard.cold_streaks.length === 0 ? (
              <p className="text-muted-foreground">No cold streaks detected</p>
            ) : (
              <div className="space-y-2">
                {dashboard.cold_streaks.map((player) => (
                  <div key={player.player_id} className="flex justify-between p-2 bg-blue-50 rounded">
                    <span className="font-medium">{player.name}</span>
                    <span className="text-blue-600">{player.trend_score.toFixed(2)} z</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        </Card>

        {/* Waiver Targets */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Zap className="h-5 w-5" />
              Waiver Targets
            </CardTitle>
          </CardHeader>
          <div className="p-6 pt-0">
            {dashboard.waiver_targets.length === 0 ? (
              <p className="text-muted-foreground">No waiver targets available</p>
            ) : (
              <div className="space-y-2">
                {dashboard.waiver_targets.slice(0, 5).map((target) => (
                  <div key={target.player_id} className="flex justify-between p-2 border rounded">
                    <div>
                      <span className="font-medium">{target.name}</span>
                      <span className="text-sm text-muted-foreground ml-2">{target.tier}</span>
                    </div>
                    <span className="font-semibold">{target.priority_score.toFixed(1)}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        </Card>

        {/* Injury Report */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Activity className="h-5 w-5" />
              Injury Report
            </CardTitle>
          </CardHeader>
          <div className="p-6 pt-0">
            {dashboard.injury_flags.length === 0 ? (
              <p className="text-green-600 font-medium">All players healthy</p>
            ) : (
              <div className="space-y-2">
                {dashboard.injury_flags.map((flag) => (
                  <div key={flag.player_id} className="flex justify-between p-2 bg-red-50 rounded">
                    <div>
                      <span className="font-medium">{flag.name}</span>
                      <Badge variant="default" className="ml-2 bg-red-100 text-red-800">{flag.status}</Badge>
                    </div>
                    <span className="text-sm text-red-600">{flag.severity}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        </Card>
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
      <div className="h-10 w-64 mb-2 bg-gray-200 animate-pulse rounded" />
      <div className="h-4 w-48 mb-8 bg-gray-200 animate-pulse rounded" />

      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
        {[1, 2, 3, 4].map((i) => (
          <div key={i} className="h-24 bg-gray-200 animate-pulse rounded" />
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {[1, 2, 3, 4, 5, 6].map((i) => (
          <div key={i} className="h-64 bg-gray-200 animate-pulse rounded" />
        ))}
      </div>
    </div>
  )
}
