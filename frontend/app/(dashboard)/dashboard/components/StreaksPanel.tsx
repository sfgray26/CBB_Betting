"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { TrendingUp, TrendingDown, Minus, Flame, Snowflake } from "lucide-react"
import type { StreakPlayer } from "@/lib/types"

interface StreaksPanelProps {
  hotStreaks: StreakPlayer[]
  coldStreaks: StreakPlayer[]
}

export function StreaksPanel({ hotStreaks, coldStreaks }: StreaksPanelProps) {
  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="flex items-center gap-2">
          <TrendingUp className="h-5 w-5" />
          Player Streaks
        </CardTitle>
      </CardHeader>
      <CardContent>
        <Tabs defaultValue="hot" className="w-full">
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="hot" className="flex items-center gap-2">
              <Flame className="h-4 w-4 text-orange-500" />
              Hot ({hotStreaks.length})
            </TabsTrigger>
            <TabsTrigger value="cold" className="flex items-center gap-2">
              <Snowflake className="h-4 w-4 text-blue-500" />
              Cold ({coldStreaks.length})
            </TabsTrigger>
          </TabsList>

          <TabsContent value="hot" className="mt-4">
            {hotStreaks.length === 0 ? (
              <EmptyState message="No hot streaks detected" icon={<Flame className="h-8 w-8 text-muted-foreground" />} />
            ) : (
              <div className="space-y-2">
                {hotStreaks.map((player) => (
                  <StreakRow key={player.player_id} player={player} type="hot" />
                ))}
              </div>
            )}
          </TabsContent>

          <TabsContent value="cold" className="mt-4">
            {coldStreaks.length === 0 ? (
              <EmptyState message="No cold streaks detected" icon={<Snowflake className="h-8 w-8 text-muted-foreground" />} />
            ) : (
              <div className="space-y-2">
                {coldStreaks.map((player) => (
                  <StreakRow key={player.player_id} player={player} type="cold" />
                ))}
              </div>
            )}
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  )
}

function StreakRow({ player, type }: { player: StreakPlayer; type: "hot" | "cold" }) {
  const isHot = type === "hot"
  const trendColor = isHot ? "text-orange-600" : "text-blue-600"
  const bgColor = isHot ? "bg-orange-50" : "bg-blue-50"
  const Icon = isHot ? Flame : Snowflake

  return (
    <div className={`flex items-center justify-between p-3 rounded-lg ${bgColor}`}>
      <div className="flex items-center gap-3">
        <Icon className={`h-4 w-4 ${trendColor}`} />
        <div>
          <p className="font-medium text-sm">{player.name}</p>
          <p className="text-xs text-muted-foreground">
            {player.team} · {player.positions.join(", ")}
          </p>
        </div>
      </div>
      <div className="text-right">
        <p className={`font-semibold text-sm ${trendColor}`}>
          {isHot ? "+" : ""}{player.trend_score.toFixed(2)} z
        </p>
        <p className="text-xs text-muted-foreground">
          L7: {player.last_7_avg.toFixed(3)}
        </p>
      </div>
    </div>
  )
}

function EmptyState({ message, icon }: { message: string; icon: React.ReactNode }) {
  return (
    <div className="flex flex-col items-center justify-center py-8 text-center">
      {icon}
      <p className="mt-2 text-sm text-muted-foreground">{message}</p>
    </div>
  )
}
