"use client"

import { useQuery, keepPreviousData } from "@tanstack/react-query"
import { endpoints } from "@/lib/api"
import type { DashboardData } from "@/lib/types"
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card"
import { Alert, AlertTitle, AlertDescription } from "@/components/ui/alert"
import {
  AlertCircle,
  Database,
  LayoutDashboard,
} from "lucide-react"

export default function DashboardPage() {
  const { data: response, isLoading, isFetching, isError, error: queryError } = useQuery({
    queryKey: ['dashboard'],
    queryFn: endpoints.getDashboard,
    staleTime: 2 * 60_000,       // serve cached data for 2 min
    refetchInterval: 5 * 60_000, // auto-refresh every 5 min
    placeholderData: keepPreviousData,
  })

  const dashboard: DashboardData | null = response?.success ? response.data : null
  const error: string | null = isError || queryError
    ? (queryError instanceof Error ? queryError.message : "Failed to load dashboard")
    : response && !response.success ? "Failed to load dashboard data" : null

  // Only show skeleton on very first load (no cached data yet)
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
      {/* Stale indicator */}
      {isFetching && (
        <div className="mb-4 text-xs text-zinc-500 flex items-center gap-1.5">
          <span className="inline-block h-2 w-2 rounded-full bg-amber-400 animate-pulse" />
          Refreshing…
        </div>
      )}

      {/* Header */}
      <div className="mb-8">
        <h1 className="text-xl font-semibold text-zinc-100 mb-2">Dashboard</h1>
        <p className="text-muted-foreground">
          Last updated: {
            new Date(dashboard.timestamp ?? Date.now()).toLocaleString('en-US', {
              timeZone: 'America/New_York', dateStyle: 'short', timeStyle: 'short'
            }) + ' ET'
          }{!dashboard.timestamp ? ' (approx)' : ''}
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-zinc-100">
              <LayoutDashboard className="h-5 w-5 text-amber-400" />
              Dashboard Status
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-3 text-sm text-zinc-400">
            <p>The shared dashboard route remains active and continues to validate the preserved dashboard API contract.</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-zinc-100">
              <Database className="h-5 w-5 text-amber-400" />
              Data Snapshot
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-2 text-sm text-zinc-400">
            <p>User: <span className="text-zinc-200">{dashboard.user_id}</span></p>
            <p>Payload loaded successfully from the preserved dashboard endpoint.</p>
          </CardContent>
        </Card>
      </div>
    </div>
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
