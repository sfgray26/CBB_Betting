"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { ArrowRight, Plus, Users, Star } from "lucide-react"
import type { WaiverTarget } from "@/lib/types"
import Link from "next/link"

interface WaiverTargetsPanelProps {
  targets: WaiverTarget[]
}

export function WaiverTargetsPanel({ targets }: WaiverTargetsPanelProps) {
  return (
    <Card>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2">
            <Star className="h-5 w-5" />
            Waiver Targets
          </CardTitle>
          <Badge variant="secondary">{targets.length} available</Badge>
        </div>
      </CardHeader>
      <CardContent>
        {targets.length === 0 ? (
          <div className="text-center py-6">
            <Users className="h-8 w-8 text-muted-foreground mx-auto mb-2" />
            <p className="text-sm text-muted-foreground">No waiver targets available</p>
          </div>
        ) : (
          <div className="space-y-3">
            {targets.slice(0, 5).map((target) => (
              <WaiverTargetRow key={target.player_id} target={target} />
            ))}
          </div>
        )}

        <Button asChild variant="outline" className="w-full mt-4">
          <Link href="/fantasy/waiver">
            View All Targets
            <ArrowRight className="ml-2 h-4 w-4" />
          </Link>
        </Button>
      </CardContent>
    </Card>
  )
}

function WaiverTargetRow({ target }: { target: WaiverTarget }) {
  const tierColors = {
    must_add: "bg-red-100 text-red-800 border-red-200",
    strong_add: "bg-amber-100 text-amber-800 border-amber-200",
    streamer: "bg-blue-100 text-blue-800 border-blue-200",
  }

  const tierLabels = {
    must_add: "Must Add",
    strong_add: "Strong Add",
    streamer: "Streamer",
  }

  return (
    <div className="flex items-start justify-between p-3 border rounded-lg hover:bg-muted/50 transition-colors">
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2">
          <p className="font-medium text-sm truncate">{target.name}</p>
          <Badge variant="outline" className={`text-xs ${tierColors[target.tier]}`}>
            {tierLabels[target.tier]}
          </Badge>
        </div>
        <p className="text-xs text-muted-foreground mt-1">
          {target.team} · {target.positions.join(", ")} · {target.percent_owned}% owned
        </p>
        <p className="text-xs text-muted-foreground mt-1">{target.reason}</p>
      </div>
      <div className="text-right ml-4">
        <p className="font-semibold text-sm">{target.priority_score.toFixed(1)}</p>
        <p className="text-xs text-muted-foreground">score</p>
      </div>
    </div>
  )
}
