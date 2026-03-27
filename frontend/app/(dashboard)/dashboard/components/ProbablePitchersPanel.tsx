"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Calendar, Target, Zap, CheckCircle2, AlertCircle } from "lucide-react"
import type { ProbablePitcherInfo } from "@/lib/types"

interface ProbablePitchersPanelProps {
  pitchers: ProbablePitcherInfo[]
  twoStarters: ProbablePitcherInfo[]
}

export function ProbablePitchersPanel({ pitchers, twoStarters }: ProbablePitchersPanelProps) {
  const hasTwoStarters = twoStarters.length > 0

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="flex items-center gap-2">
          <Calendar className="h-5 w-5" />
          Probable Pitchers
          {hasTwoStarters && (
            <Badge variant="default" className="bg-amber-500">
              {twoStarters.length} Two-Start
            </Badge>
          )}
        </CardTitle>
      </CardHeader>
      <CardContent>
        <Tabs defaultValue={hasTwoStarters ? "twostart" : "starters"} className="w-full">
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="starters">
              Starters ({pitchers.length})
            </TabsTrigger>
            <TabsTrigger value="twostart" disabled={!hasTwoStarters}>
              Two-Start ({twoStarters.length})
            </TabsTrigger>
          </TabsList>

          <TabsContent value="starters" className="mt-4">
            {pitchers.length === 0 ? (
              <EmptyState message="No probable starters for today" />
            ) : (
              <div className="space-y-2">
                {pitchers.slice(0, 4).map((pitcher) => (
                  <PitcherRow key={`${pitcher.name}-${pitcher.game_date}`} pitcher={pitcher} />
                ))}
              </div>
            )}
          </TabsContent>

          <TabsContent value="twostart" className="mt-4">
            {twoStarters.length === 0 ? (
              <EmptyState message="No two-start pitchers this week" />
            ) : (
              <div className="space-y-2">
                {twoStarters.map((pitcher) => (
                  <PitcherRow key={pitcher.name} pitcher={pitcher} highlight />
                ))}
              </div>
            )}
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  )
}

function PitcherRow({ pitcher, highlight = false }: { pitcher: ProbablePitcherInfo; highlight?: boolean }) {
  const qualityColors = {
    favorable: "text-green-600 bg-green-50 border-green-200",
    neutral: "text-blue-600 bg-blue-50 border-blue-200",
    unfavorable: "text-red-600 bg-red-50 border-red-200",
  }

  const qualityIcons = {
    favorable: <CheckCircle2 className="h-3 w-3" />,
    neutral: <Target className="h-3 w-3" />,
    unfavorable: <AlertCircle className="h-3 w-3" />,
  }

  return (
    <div
      className={`flex items-center justify-between p-3 border rounded-lg ${
        highlight ? "bg-amber-50 border-amber-200" : "hover:bg-muted/50"
      }`}
    >
      <div className="flex-1">
        <div className="flex items-center gap-2">
          <p className="font-medium text-sm">{pitcher.name}</p>
          {pitcher.is_two_start && (
            <Badge variant="outline" className="text-xs bg-amber-100 text-amber-800">
              <Zap className="h-3 w-3 mr-1" />
              2-Start
            </Badge>
          )}
        </div>
        <p className="text-xs text-muted-foreground">
          {pitcher.team} vs {pitcher.opponent} · {pitcher.game_date}
        </p>
        <p className="text-xs text-muted-foreground mt-1">{pitcher.reason}</p>
      </div>
      <div className="text-right ml-4">
        <Badge
          variant="outline"
          className={`text-xs flex items-center gap-1 ${qualityColors[pitcher.matchup_quality]}`}
        >
          {qualityIcons[pitcher.matchup_quality]}
          {pitcher.matchup_quality}
        </Badge>
        <p className="text-xs text-muted-foreground mt-1">
          Score: {pitcher.stream_score.toFixed(1)}
        </p>
      </div>
    </div>
  )
}

function EmptyState({ message }: { message: string }) {
  return (
    <div className="text-center py-6">
      <Calendar className="h-8 w-8 text-muted-foreground mx-auto mb-2" />
      <p className="text-sm text-muted-foreground">{message}</p>
    </div>
  )
}
