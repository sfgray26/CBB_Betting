"use client"

import { useEffect, useState } from "react"
import { api } from "@/lib/api"
import type { UserPreferences } from "@/lib/types"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Switch } from "@/components/ui/switch"
import { Label } from "@/components/ui/label"
import { Input } from "@/components/ui/input"
import { Slider } from "@/components/ui/slider"
import { Badge } from "@/components/ui/badge"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Separator } from "@/components/ui/separator"
import { Alert, AlertDescription } from "@/components/ui/alert"
import {
  Bell,
  LayoutDashboard,
  TrendingUp,
  Users,
  Save,
  RefreshCw,
  CheckCircle2,
  AlertCircle,
  Mail,
  MessageSquare,
  Smartphone,
} from "lucide-react"

export default function SettingsPage() {
  const [preferences, setPreferences] = useState<UserPreferences | null>(null)
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [saved, setSaved] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    loadPreferences()
  }, [])

  async function loadPreferences() {
    try {
      setLoading(true)
      const response = await api.getUserPreferences()
      if (response.success) {
        setPreferences(response.preferences)
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load preferences")
    } finally {
      setLoading(false)
    }
  }

  async function savePreferences() {
    if (!preferences) return
    
    try {
      setSaving(true)
      setSaved(false)
      await api.updateUserPreferences(preferences)
      setSaved(true)
      setTimeout(() => setSaved(false), 3000)
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to save preferences")
    } finally {
      setSaving(false)
    }
  }

  if (loading) {
    return (
      <div className="container mx-auto py-8 px-4">
        <h1 className="text-3xl font-bold mb-8">Settings</h1>
        <div className="animate-pulse space-y-4">
          {[1, 2, 3].map((i) => (
            <div key={i} className="h-64 bg-muted rounded-lg" />
          ))}
        </div>
      </div>
    )
  }

  if (!preferences) {
    return (
      <div className="container mx-auto py-8 px-4">
        <h1 className="text-3xl font-bold mb-8">Settings</h1>
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>Failed to load preferences. Please try refreshing.</AlertDescription>
        </Alert>
      </div>
    )
  }

  return (
    <div className="container mx-auto py-8 px-4 max-w-5xl">
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-3xl font-bold">Settings</h1>
          <p className="text-muted-foreground mt-1">
            Customize your dashboard experience and notification preferences
          </p>
        </div>
        <div className="flex items-center gap-2">
          {saved && (
            <Badge variant="default" className="bg-green-100 text-green-800">
              <CheckCircle2 className="h-3 w-3 mr-1" />
              Saved
            </Badge>
          )}
          <Button onClick={savePreferences} disabled={saving}>
            {saving ? (
              <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
            ) : (
              <Save className="h-4 w-4 mr-2" />
            )}
            Save Changes
          </Button>
        </div>
      </div>

      {error && (
        <Alert variant="destructive" className="mb-6">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      <Tabs defaultValue="notifications" className="space-y-6">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="notifications" className="flex items-center gap-2">
            <Bell className="h-4 w-4" />
            Notifications
          </TabsTrigger>
          <TabsTrigger value="dashboard" className="flex items-center gap-2">
            <LayoutDashboard className="h-4 w-4" />
            Dashboard
          </TabsTrigger>
          <TabsTrigger value="streaks" className="flex items-center gap-2">
            <TrendingUp className="h-4 w-4" />
            Streaks
          </TabsTrigger>
          <TabsTrigger value="waiver" className="flex items-center gap-2">
            <Users className="h-4 w-4" />
            Waiver
          </TabsTrigger>
        </TabsList>

        <TabsContent value="notifications">
          <NotificationsPanel
            notifications={preferences.notifications}
            onChange={(notifications) =>
              setPreferences({ ...preferences, notifications })
            }
          />
        </TabsContent>

        <TabsContent value="dashboard">
          <DashboardPanel
            layout={preferences.dashboard_layout}
            onChange={(dashboard_layout) =>
              setPreferences({ ...preferences, dashboard_layout })
            }
          />
        </TabsContent>

        <TabsContent value="streaks">
          <StreaksPanel
            settings={preferences.streak_settings}
            onChange={(streak_settings) =>
              setPreferences({ ...preferences, streak_settings })
            }
          />
        </TabsContent>

        <TabsContent value="waiver">
          <WaiverPanel
            preferences={preferences.waiver_preferences}
            onChange={(waiver_preferences) =>
              setPreferences({ ...preferences, waiver_preferences })
            }
          />
        </TabsContent>
      </Tabs>
    </div>
  )
}

// Notifications Panel
function NotificationsPanel({
  notifications,
  onChange,
}: {
  notifications: UserPreferences["notifications"]
  onChange: (n: UserPreferences["notifications"]) => void
}) {
  const updateField = (field: string, value: any) => {
    onChange({ ...notifications, [field]: value })
  }

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Bell className="h-5 w-5" />
            Notification Types
          </CardTitle>
          <CardDescription>
            Choose which alerts you want to receive
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {[
            { key: "lineup_deadline", label: "Lineup Deadline Alerts", desc: "Get notified when lineups are about to lock" },
            { key: "injury_alerts", label: "Injury Alerts", desc: "Breaking injury news for your rostered players" },
            { key: "waiver_suggestions", label: "Waiver Suggestions", desc: "Daily recommendations for waiver adds" },
            { key: "trade_offers", label: "Trade Offers", desc: "When you receive trade proposals" },
            { key: "hot_streak_alerts", label: "Hot Streak Alerts", desc: "When players on your roster heat up" },
          ].map(({ key, label, desc }) => (
            <div key={key} className="flex items-center justify-between">
              <div>
                <Label className="font-medium">{label}</Label>
                <p className="text-sm text-muted-foreground">{desc}</p>
              </div>
              <Switch
                checked={notifications[key as keyof typeof notifications] as boolean}
                onCheckedChange={(checked) => updateField(key, checked)}
              />
            </div>
          ))}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <MessageSquare className="h-5 w-5" />
            Notification Channels
          </CardTitle>
          <CardDescription>
            How you want to receive notifications
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex items-center gap-4">
            <div className="flex-1">
              <Label className="font-medium flex items-center gap-2">
                <MessageSquare className="h-4 w-4" />
                Discord
              </Label>
              <p className="text-sm text-muted-foreground">Alerts via Discord bot</p>
            </div>
            <Switch
              checked={notifications.channels.includes("discord")}
              onCheckedChange={(checked) => {
                const channels = checked
                  ? [...notifications.channels, "discord"]
                  : notifications.channels.filter((c) => c !== "discord")
                updateField("channels", channels)
              }}
            />
          </div>

          <div className="flex items-center gap-4">
            <div className="flex-1">
              <Label className="font-medium flex items-center gap-2">
                <Mail className="h-4 w-4" />
                Email
              </Label>
              <p className="text-sm text-muted-foreground">Alerts via email</p>
            </div>
            <Switch
              checked={notifications.email_enabled}
              onCheckedChange={(checked) => updateField("email_enabled", checked)}
            />
          </div>

          <div className="flex items-center gap-4">
            <div className="flex-1">
              <Label className="font-medium flex items-center gap-2">
                <Smartphone className="h-4 w-4" />
                Push Notifications
              </Label>
              <p className="text-sm text-muted-foreground">Browser push notifications</p>
            </div>
            <Switch
              checked={notifications.channels.includes("push")}
              onCheckedChange={(checked) => {
                const channels = checked
                  ? [...notifications.channels, "push"]
                  : notifications.channels.filter((c) => c !== "push")
                updateField("channels", channels)
              }}
            />
          </div>

          {notifications.channels.includes("discord") && (
            <div className="pt-4 border-t">
              <Label htmlFor="discord-user-id">Discord User ID (optional)</Label>
              <Input
                id="discord-user-id"
                value={notifications.discord_user_id || ""}
                onChange={(e) => updateField("discord_user_id", e.target.value)}
                placeholder="123456789"
                className="mt-2"
              />
              <p className="text-xs text-muted-foreground mt-1">
                For DM notifications. Leave blank for channel alerts only.
              </p>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}

// Dashboard Panel
function DashboardPanel({
  layout,
  onChange,
}: {
  layout: UserPreferences["dashboard_layout"]
  onChange: (l: UserPreferences["dashboard_layout"]) => void
}) {
  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <LayoutDashboard className="h-5 w-5" />
            Dashboard Panels
          </CardTitle>
          <CardDescription>
            Enable or disable dashboard panels
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {layout.panels.map((panel, index) => (
            <div key={panel.id} className="flex items-center justify-between p-3 border rounded-lg">
              <div>
                <Label className="font-medium capitalize">
                  {panel.id.replace(/_/g, " ")}
                </Label>
                <p className="text-sm text-muted-foreground">
                  Position: {panel.position} · Size: {panel.size}
                </p>
              </div>
              <Switch
                checked={panel.enabled}
                onCheckedChange={(checked) => {
                  const panels = [...layout.panels]
                  panels[index] = { ...panel, enabled: checked }
                  onChange({ ...layout, panels })
                }}
              />
            </div>
          ))}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Refresh Settings</CardTitle>
          <CardDescription>
            How often to refresh dashboard data
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div>
            <div className="flex justify-between mb-2">
              <Label>Refresh Interval</Label>
              <span className="text-sm font-medium">
                {layout.refresh_interval_seconds < 60
                  ? `${layout.refresh_interval_seconds}s`
                  : `${Math.round(layout.refresh_interval_seconds / 60)}m`}
              </span>
            </div>
            <Slider
              value={[layout.refresh_interval_seconds]}
              onValueChange={([value]) =>
                onChange({ ...layout, refresh_interval_seconds: value })
              }
              min={30}
              max={600}
              step={30}
            />
            <div className="flex justify-between text-xs text-muted-foreground mt-1">
              <span>30s</span>
              <span>10m</span>
            </div>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Theme</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex gap-2">
            {(["dark", "light", "system"] as const).map((theme) => (
              <Button
                key={theme}
                variant={layout.theme === theme ? "default" : "outline"}
                onClick={() => onChange({ ...layout, theme })}
                className="capitalize"
              >
                {theme}
              </Button>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  )
}

// Streaks Panel
function StreaksPanel({
  settings,
  onChange,
}: {
  settings: UserPreferences["streak_settings"]
  onChange: (s: UserPreferences["streak_settings"]) => void
}) {
  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <TrendingUp className="h-5 w-5" />
            Streak Detection Thresholds
          </CardTitle>
          <CardDescription>
            Define what constitutes a hot or cold streak
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <div>
            <div className="flex justify-between mb-2">
              <Label className="flex items-center gap-2 text-orange-600">
                <TrendingUp className="h-4 w-4" />
                Hot Streak Threshold
              </Label>
              <span className="text-sm font-medium">z-score: {settings.hot_threshold.toFixed(2)}</span>
            </div>
            <Slider
              value={[settings.hot_threshold]}
              onValueChange={([value]) => onChange({ ...settings, hot_threshold: value })}
              min={0}
              max={2}
              step={0.1}
            />
            <p className="text-xs text-muted-foreground mt-1">
              Players with z-scores above this value are flagged as "hot"
            </p>
          </div>

          <Separator />

          <div>
            <div className="flex justify-between mb-2">
              <Label className="flex items-center gap-2 text-blue-600">
                <TrendingUp className="h-4 w-4 rotate-180" />
                Cold Streak Threshold
              </Label>
              <span className="text-sm font-medium">z-score: {settings.cold_threshold.toFixed(2)}</span>
            </div>
            <Slider
              value={[settings.cold_threshold]}
              onValueChange={([value]) => onChange({ ...settings, cold_threshold: value })}
              min={-2}
              max={0}
              step={0.1}
            />
            <p className="text-xs text-muted-foreground mt-1">
              Players with z-scores below this value are flagged as "cold"
            </p>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Rolling Windows</CardTitle>
          <CardDescription>
            Days to include in streak calculations
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex flex-wrap gap-2">
            {[7, 14, 21, 30].map((days) => (
              <Badge
                key={days}
                variant={settings.rolling_windows.includes(days) ? "default" : "outline"}
                className="cursor-pointer"
                onClick={() => {
                  const windows = settings.rolling_windows.includes(days)
                    ? settings.rolling_windows.filter((w) => w !== days)
                    : [...settings.rolling_windows, days].sort((a, b) => a - b)
                  onChange({ ...settings, rolling_windows: windows })
                }}
              >
                {days} days
              </Badge>
            ))}
          </div>
          <p className="text-xs text-muted-foreground mt-3">
            Select which rolling averages to display for streak analysis
          </p>
        </CardContent>
      </Card>
    </div>
  )
}

// Waiver Panel
function WaiverPanel({
  preferences,
  onChange,
}: {
  preferences: UserPreferences["waiver_preferences"]
  onChange: (p: UserPreferences["waiver_preferences"]) => void
}) {
  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Users className="h-5 w-5" />
            Waiver Wire Filters
          </CardTitle>
          <CardDescription>
            Customize which players appear in waiver recommendations
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <div>
            <div className="flex justify-between mb-2">
              <Label>Minimum Ownership %</Label>
              <span className="text-sm font-medium">{preferences.min_percent_owned}%</span>
            </div>
            <Slider
              value={[preferences.min_percent_owned]}
              onValueChange={([value]) => onChange({ ...preferences, min_percent_owned: value })}
              min={0}
              max={50}
              step={5}
            />
            <p className="text-xs text-muted-foreground mt-1">
              Hide players owned by fewer than this percentage
            </p>
          </div>

          <div>
            <div className="flex justify-between mb-2">
              <Label>Maximum Ownership %</Label>
              <span className="text-sm font-medium">{preferences.max_percent_owned}%</span>
            </div>
            <Slider
              value={[preferences.max_percent_owned]}
              onValueChange={([value]) => onChange({ ...preferences, max_percent_owned: value })}
              min={50}
              max={100}
              step={5}
            />
            <p className="text-xs text-muted-foreground mt-1">
              Hide players owned by more than this percentage (likely unavailable)
            </p>
          </div>

          <Separator />

          <div className="flex items-center justify-between">
            <div>
              <Label className="font-medium">Hide Injured Players</Label>
              <p className="text-sm text-muted-foreground">Exclude players on IL/DL from recommendations</p>
            </div>
            <Switch
              checked={preferences.hide_injured}
              onCheckedChange={(checked) => onChange({ ...preferences, hide_injured: checked })}
            />
          </div>

          <Separator />

          <div>
            <div className="flex justify-between mb-2">
              <Label>Streamer Threshold (z-score)</Label>
              <span className="text-sm font-medium">{preferences.streamer_threshold.toFixed(2)}</span>
            </div>
            <Slider
              value={[preferences.streamer_threshold]}
              onValueChange={([value]) => onChange({ ...preferences, streamer_threshold: value })}
              min={-1}
              max={1}
              step={0.1}
            />
            <p className="text-xs text-muted-foreground mt-1">
              Minimum z-score to be considered a viable streamer
            </p>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
