"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { Trophy, Target, TrendingUp, TrendingDown, Minus } from "lucide-react"
import type { MatchupPreviewData } from "@/lib/types"

interface MatchupPreviewPanelProps {
  preview: MatchupPreviewData | null
}

export function MatchupPreviewPanel({ preview }: MatchupPreviewPanelProps) {
  if (!preview) {
    return (
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="flex items-center gap-2">
            <Trophy className="h-5 w-5" />
            This Week&apos;s Matchup
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center py-6">
            <Target className="h-8 w-8 text-muted-foreground mx-auto mb-2" />
            <p className="text-sm text-muted-foreground">No active matchup</p>
            <p className="text-xs text-muted-foreground mt-1">
              Matchup data will appear when the season starts
            </p>
          </div>
        </CardContent>
      </Card>
    )
  }

  const winProbPercent = Math.round(preview.win_probability * 100)
  const isFavored = winProbPercent > 50

  return (
    <Card>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2">
            <Trophy className="h-5 w-5" />
            Week {preview.week_number} Matchup
          </CardTitle>
          <Badge variant={isFavored ? "default" : "secondary"}>
            {winProbPercent}% win prob
          </Badge>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Opponent Info */}
        <div className="flex items-center justify-between p-3 bg-muted rounded-lg">
          <div>
            <p className="text-sm text-muted-foreground">Opponent</p>
            <p className="font-semibold">{preview.opponent_team_name}</p>
          </div>
          <div className="text-right">
            <p className="text-sm text-muted-foreground">Record</p>
            <p className="font-semibold">{preview.opponent_record}</p>
          </div>
        </div>

        {/* Win Probability Bar */}
        <div>
          <div className="flex justify-between text-sm mb-1">
            <span>You</span>
            <span className={isFavored ? "text-green-600 font-medium" : ""}>
              {winProbPercent}%
            </span>
          </div>
          <Progress value={winProbPercent} className="h-2" />
          <div className="flex justify-between text-sm mt-1">
            <span className="text-muted-foreground">Win Probability</span>
            <span className="text-muted-foreground">{100 - winProbPercent}% Them</span>
          </div>
        </div>

        {/* Category Advantages */}
        {(preview.category_advantages.length > 0 || preview.category_disadvantages.length > 0) && (
          <div className="grid grid-cols-2 gap-4">
            {preview.category_advantages.length > 0 && (
              <div>
                <p className="text-xs font-medium text-green-600 mb-2 flex items-center gap-1">
                  <TrendingUp className="h-3 w-3" />
                  Advantages
                </p>
                <div className="flex flex-wrap gap-1">
                  {preview.category_advantages.map((cat) => (
                    <Badge key={cat} variant="outline" className="text-xs bg-green-50">
                      {cat}
                    </Badge>
                  ))}
                </div>
              </div>
            )}

            {preview.category_disadvantages.length > 0 && (
              <div>
                <p className="text-xs font-medium text-red-600 mb-2 flex items-center gap-1">
                  <TrendingDown className="h-3 w-3" />
                  Disadvantages
                </p>
                <div className="flex flex-wrap gap-1">
                  {preview.category_disadvantages.map((cat) => (
                    <Badge key={cat} variant="outline" className="text-xs bg-red-50">
                      {cat}
                    </Badge>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  )
}
