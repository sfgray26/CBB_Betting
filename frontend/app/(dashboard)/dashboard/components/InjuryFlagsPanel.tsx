"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Activity, AlertTriangle, Stethoscope, Calendar } from "lucide-react"
import type { InjuryFlag } from "@/lib/types"

interface InjuryFlagsPanelProps {
  flags: InjuryFlag[]
  healthyCount: number
  injuredCount: number
}

export function InjuryFlagsPanel({ flags, healthyCount, injuredCount }: InjuryFlagsPanelProps) {
  const criticalFlags = flags.filter((f) => f.severity === "critical")
  const warningFlags = flags.filter((f) => f.severity === "warning")

  return (
    <Card>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2">
            <Stethoscope className="h-5 w-5" />
            Injury Report
          </CardTitle>
          <div className="flex gap-2">
            <Badge variant="default" className="bg-green-100 text-green-800">
              {healthyCount} healthy
            </Badge>
            {injuredCount > 0 && (
              <Badge variant="destructive">{injuredCount} injured</Badge>
            )}
          </div>
        </div>
      </CardHeader>
      <CardContent>
        {flags.length === 0 ? (
          <div className="text-center py-6">
            <Activity className="h-8 w-8 text-green-500 mx-auto mb-2" />
            <p className="text-sm font-medium text-green-700">All players healthy</p>
            <p className="text-xs text-muted-foreground mt-1">
              No injuries reported on your roster
            </p>
          </div>
        ) : (
          <div className="space-y-3">
            {criticalFlags.length > 0 && (
              <Alert variant="destructive">
                <AlertTriangle className="h-4 w-4" />
                <AlertTitle>{criticalFlags.length} Critical Injury(s)</AlertTitle>
                <AlertDescription>
                  Immediate action required for: {criticalFlags.map((f) => f.name).join(", ")}
                </AlertDescription>
              </Alert>
            )}

            {flags.map((flag) => (
              <InjuryFlagRow key={flag.player_id} flag={flag} />
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  )
}

function InjuryFlagRow({ flag }: { flag: InjuryFlag }) {
  const statusColors = {
    IL: "bg-red-100 text-red-800 border-red-200",
    IL10: "bg-orange-100 text-orange-800 border-orange-200",
    IL60: "bg-red-100 text-red-800 border-red-200",
    DTD: "bg-yellow-100 text-yellow-800 border-yellow-200",
    OUT: "bg-gray-100 text-gray-800 border-gray-200",
  }

  const severityIcons = {
    critical: <AlertTriangle className="h-4 w-4 text-red-500" />,
    warning: <Activity className="h-4 w-4 text-yellow-500" />,
    info: <Activity className="h-4 w-4 text-blue-500" />,
  }

  return (
    <div className="flex items-start justify-between p-3 border rounded-lg">
      <div className="flex-1">
        <div className="flex items-center gap-2">
          {severityIcons[flag.severity]}
          <p className="font-medium text-sm">{flag.name}</p>
          <Badge variant="outline" className={`text-xs ${statusColors[flag.status] || "bg-gray-100"}`}>
            {flag.status}
          </Badge>
        </div>
        {flag.injury_note && (
          <p className="text-xs text-muted-foreground mt-1 ml-6">{flag.injury_note}</p>
        )}
        {flag.estimated_return && (
          <p className="text-xs text-muted-foreground mt-1 ml-6 flex items-center gap-1">
            <Calendar className="h-3 w-3" />
            Est. return: {flag.estimated_return}
          </p>
        )}
        <p className="text-xs font-medium mt-1 ml-6">{flag.action_needed}</p>
      </div>
    </div>
  )
}
