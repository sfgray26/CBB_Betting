"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { AlertTriangle, Plus, ArrowRight } from "lucide-react"
import type { LineupGap } from "@/lib/types"
import Link from "next/link"

interface LineupGapsPanelProps {
  gaps: LineupGap[]
}

export function LineupGapsPanel({ gaps }: LineupGapsPanelProps) {
  if (gaps.length === 0) {
    return (
      <Card className="border-green-200 bg-green-50/50">
        <CardContent className="p-6">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-green-100 rounded-full">
              <svg className="w-5 h-5 text-green-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
              </svg>
            </div>
            <div>
              <h3 className="font-semibold text-green-900">Lineup Complete</h3>
              <p className="text-sm text-green-700">All positions are filled for today.</p>
            </div>
          </div>
        </CardContent>
      </Card>
    )
  }

  const criticalGaps = gaps.filter((g) => g.severity === "critical")
  const warningGaps = gaps.filter((g) => g.severity === "warning")

  return (
    <Card className="border-amber-200">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <AlertTriangle className="h-5 w-5 text-amber-600" />
            <CardTitle>Lineup Gaps Detected</CardTitle>
          </div>
          <Badge variant="destructive">{gaps.length} unfilled</Badge>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {criticalGaps.length > 0 && (
          <Alert variant="destructive">
            <AlertTriangle className="h-4 w-4" />
            <AlertTitle>Critical: {criticalGaps.length} position(s) empty</AlertTitle>
            <AlertDescription>
              You have no eligible players for: {criticalGaps.map((g) => g.position).join(", ")}
            </AlertDescription>
          </Alert>
        )}

        <div className="space-y-2">
          {gaps.map((gap) => (
            <div
              key={gap.position}
              className="flex items-center justify-between p-3 bg-muted rounded-lg"
            >
              <div className="flex items-center gap-3">
                <Badge
                  variant={gap.severity === "critical" ? "destructive" : "secondary"}
                >
                  {gap.position}
                </Badge>
                <span className="text-sm">{gap.message}</span>
              </div>
              {gap.suggested_add && (
                <div className="flex items-center gap-2 text-sm text-muted-foreground">
                  <span>Suggested: {gap.suggested_add}</span>
                  <Plus className="h-4 w-4" />
                </div>
              )}
            </div>
          ))}
        </div>

        <div className="flex gap-2">
          <Button asChild variant="default" size="sm">
            <Link href="/fantasy/lineup">
              Fix Lineup
              <ArrowRight className="ml-2 h-4 w-4" />
            </Link>
          </Button>
          <Button asChild variant="outline" size="sm">
            <Link href="/fantasy/waiver">
              Find Players
              <Plus className="ml-2 h-4 w-4" />
            </Link>
          </Button>
        </div>
      </CardContent>
    </Card>
  )
}
