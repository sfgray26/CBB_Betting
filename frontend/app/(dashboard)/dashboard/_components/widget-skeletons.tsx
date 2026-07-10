"use client"

import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card"
import {
  AlertCircle,
  TrendingUp,
  Activity,
  Users,
  Star,
  Calendar,
  DollarSign,
} from "lucide-react"

// ─── Shared skeleton building blocks ──────────────────────────────────────────

function SkeletonRow({ className = "" }: { className?: string }) {
  return (
    <div className={`h-4 bg-bg-elevated animate-pulse rounded ${className}`} />
  )
}

function SkeletonBar({ className = "" }: { className?: string }) {
  return (
    <div className={`h-2.5 bg-bg-elevated animate-pulse rounded-full ${className}`} />
  )
}

// ─── Widget-specific skeletons ────────────────────────────────────────────────

export function DashboardHeaderSkeleton() {
  return (
    <div className="mb-8">
      <div className="h-7 w-40 mb-2 bg-bg-elevated animate-pulse rounded" />
      <div className="h-4 w-56 bg-bg-elevated animate-pulse rounded" />
    </div>
  )
}

export function LineupStatusSkeleton() {
  return (
    <div className="mb-6 p-4 bg-bg-surface border border-border-subtle rounded-lg flex items-center gap-4">
      <div className="h-4 w-4 bg-bg-elevated animate-pulse rounded-full" />
      <div className="h-4 w-48 bg-bg-elevated animate-pulse rounded" />
      <div className="ml-auto h-3 w-24 bg-bg-elevated animate-pulse rounded" />
    </div>
  )
}

export function LineupGapsSkeleton() {
  return (
    <Card className="bg-bg-surface border-border-subtle">
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-text-primary text-sm">
          <Users className="h-4 w-4 text-accent-gold" />
          Lineup Gaps
          <span className="ml-auto h-5 w-8 bg-bg-elevated animate-pulse rounded" />
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <SkeletonRow className="w-full" />
        <SkeletonRow className="w-5/6" />
        <SkeletonRow className="w-4/5" />
        <div className="mt-3 pt-3 border-t border-border-subtle">
          <div className="h-3 w-24 bg-bg-elevated animate-pulse rounded" />
        </div>
      </CardContent>
    </Card>
  )
}

export function InjuryFlagsSkeleton() {
  return (
    <Card className="bg-bg-surface border-border-subtle">
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-text-primary text-sm">
          <AlertCircle className="h-4 w-4 text-accent-gold" />
          Injury Alerts
          <span className="ml-auto h-5 w-8 bg-bg-elevated animate-pulse rounded" />
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="flex items-start gap-2">
          <div className="mt-0.5 h-2 w-2 rounded-full bg-bg-elevated animate-pulse shrink-0" />
          <div className="flex-1 space-y-1.5">
            <SkeletonRow className="w-32" />
            <SkeletonRow className="w-full" />
            <SkeletonRow className="w-2/3" />
          </div>
        </div>
        <div className="flex items-start gap-2">
          <div className="mt-0.5 h-2 w-2 rounded-full bg-bg-elevated animate-pulse shrink-0" />
          <div className="flex-1 space-y-1.5">
            <SkeletonRow className="w-28" />
            <SkeletonRow className="w-full" />
          </div>
        </div>
      </CardContent>
    </Card>
  )
}

export function WaiverTargetsSkeleton() {
  return (
    <Card className="bg-bg-surface border-border-subtle">
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-text-primary text-sm">
          <TrendingUp className="h-4 w-4 text-accent-gold" />
          Top Waiver Targets
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        {[1, 2, 3].map((i) => (
          <div key={i} className="flex items-start justify-between gap-2">
            <div className="flex-1 space-y-1.5">
              <div className="flex items-center gap-2">
                <SkeletonRow className="w-24" />
                <SkeletonRow className="w-12" />
                <SkeletonRow className="w-16" />
              </div>
              <SkeletonRow className="w-full" />
              <SkeletonRow className="w-1/2" />
            </div>
            <div className="h-5 w-16 bg-bg-elevated animate-pulse rounded shrink-0" />
          </div>
        ))}
      </CardContent>
    </Card>
  )
}

export function ProjectionCoverageSkeleton() {
  return (
    <Card className="bg-bg-surface border-border-subtle">
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-text-primary text-sm">
          <Activity className="h-4 w-4 text-accent-gold" />
          Projection Coverage
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-2">
        <SkeletonRow className="w-24 h-7" />
        <SkeletonRow className="w-full" />
      </CardContent>
    </Card>
  )
}

export function StreaksSkeleton() {
  return (
    <Card className="bg-bg-surface border-border-subtle">
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-text-primary text-sm">
          <Activity className="h-4 w-4 text-accent-gold" />
          Player Trends
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div>
          <div className="flex items-center gap-1 mb-2">
            <SkeletonRow className="w-8" />
          </div>
          <div className="space-y-1.5">
            <SkeletonRow className="w-full" />
            <SkeletonRow className="w-full" />
            <SkeletonRow className="w-5/6" />
          </div>
        </div>
        <div>
          <div className="flex items-center gap-1 mb-2">
            <SkeletonRow className="w-8" />
          </div>
          <div className="space-y-1.5">
            <SkeletonRow className="w-full" />
            <SkeletonRow className="w-full" />
          </div>
        </div>
      </CardContent>
    </Card>
  )
}

export function BudgetSkeleton() {
  return (
    <Card className="bg-bg-surface border-border-subtle">
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-text-primary text-sm">
          <DollarSign className="h-4 w-4 text-accent-gold" />
          Constraint Budget
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <div>
          <div className="flex items-center justify-between mb-1">
            <SkeletonRow className="w-20" />
            <SkeletonRow className="w-16" />
          </div>
          <SkeletonBar className="w-full" />
        </div>
        <div className="flex items-center justify-between py-1.5 border-b border-border-subtle">
          <SkeletonRow className="w-16" />
          <SkeletonRow className="w-20" />
        </div>
        <div>
          <div className="flex items-center justify-between mb-1">
            <SkeletonRow className="w-24" />
            <SkeletonRow className="w-12" />
          </div>
          <SkeletonBar className="w-full" />
        </div>
        <div>
          <div className="flex items-center justify-between mb-1">
            <SkeletonRow className="w-28" />
            <SkeletonRow className="w-16" />
          </div>
          <SkeletonBar className="w-full" />
          <div className="flex items-center justify-between mt-1">
            <SkeletonRow className="w-24" />
            <SkeletonRow className="w-20" />
          </div>
        </div>
      </CardContent>
    </Card>
  )
}

export function ProbablePitchersSkeleton() {
  return (
    <Card className="lg:col-span-2">
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-text-primary text-sm">
          <Calendar className="h-4 w-4 text-accent-gold" />
          Probable Pitchers
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-3">
          {[1, 2, 3].map((i) => (
            <div key={i} className="p-3 bg-bg-surface border border-border-subtle rounded-md space-y-2">
              <SkeletonRow className="w-24" />
              <SkeletonRow className="w-full" />
              <div className="flex items-center gap-2">
                <SkeletonRow className="w-16" />
                <SkeletonRow className="w-12" />
              </div>
              <SkeletonRow className="w-3/4" />
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  )
}

export function TwoStartPitchersSkeleton() {
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
          {[1, 2, 3].map((i) => (
            <div key={i} className="p-3 bg-bg-surface border border-border-subtle rounded-md space-y-2">
              <SkeletonRow className="w-24" />
              <SkeletonRow className="w-full" />
              <SkeletonRow className="w-3/4" />
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  )
}
