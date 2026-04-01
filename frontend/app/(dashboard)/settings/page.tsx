"use client"

import { useEffect, useState } from "react"
import { endpoints } from "@/lib/api"
import type { UserPreferences } from "@/lib/types"
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Switch } from "@/components/ui/switch"
import { Slider } from "@/components/ui/slider"
import { Label } from "@/components/ui/label"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import {
  CheckCircle2,
  Save,
  RefreshCw,
  Bell,
  Layout,
  TrendingUp,
  Search,
} from "lucide-react"
import { cn } from "@/lib/utils"

const Z_PRESETS = [
  { id: "conservative", label: "Conservative", hot: 3.5, cold: -1.0, description: "Only high-confidence streaks" },
  { id: "balanced", label: "Balanced", hot: 2.0, cold: 0.0, description: "Standard sensitivity" },
  { id: "aggressive", label: "Aggressive", hot: 1.0, cold: 1.0, description: "Early detection of trends" },
]

const REFRESH_OPTIONS = [
  { value: "60", label: "1 minute" },
  { value: "300", label: "5 minutes" },
  { value: "600", label: "10 minutes" },
  { value: "1800", label: "30 minutes" },
  { value: "3600", label: "1 hour" },
]

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
      const response = await endpoints.getUserPreferences()
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
      const response = await endpoints.updateUserPreferences(preferences)
      if (response.success) {
        setSaved(true)
        setTimeout(() => setSaved(false), 3000)
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to save preferences")
    } finally {
      setSaving(false)
    }
  }

  const updateNested = (
    section: keyof UserPreferences,
    key: string,
    value: string | number | boolean | string[] | number[]
  ) => {
    if (!preferences) return
    setPreferences({
      ...preferences,
      [section]: {
        ...(preferences[section] as Record<string, unknown>),
        [key]: value,
      },
    })
  }

  const getActivePreset = () => {
    if (!preferences) return "custom"
    const { hot_threshold, cold_threshold } = preferences.streak_settings
    const preset = Z_PRESETS.find(p => p.hot === hot_threshold && p.cold === cold_threshold)
    return preset ? preset.id : "custom"
  }

  const applyPreset = (presetId: string) => {
    const preset = Z_PRESETS.find(p => p.id === presetId)
    if (!preset || !preferences) return
    setPreferences({
      ...preferences,
      streak_settings: {
        ...preferences.streak_settings,
        hot_threshold: preset.hot,
        cold_threshold: preset.cold
      }
    })
  }

  if (loading) {
    return (
      <div className="space-y-6 max-w-5xl mx-auto py-8 px-4">
        <div className="h-10 w-48 bg-zinc-800 animate-pulse rounded" />
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {[1, 2, 3, 4].map((i) => (
            <div key={i} className="h-64 bg-zinc-800/50 animate-pulse rounded-lg border border-zinc-800" />
          ))}
        </div>
      </div>
    )
  }

  if (!preferences) {
    return (
      <div className="container mx-auto py-8 px-4 max-w-5xl">
        <h1 className="text-3xl font-bold text-zinc-100">Settings</h1>
        <div className="mt-8 p-6 bg-rose-500/10 border border-rose-500/30 rounded-lg text-rose-400">
          <p>Failed to load preferences. Please try refreshing.</p>
          <Button variant="default" className="mt-4 border-rose-500/30 text-rose-400 hover:bg-rose-500/20" onClick={loadPreferences}>
            <RefreshCw className="h-4 w-4 mr-2" />
            Retry
          </Button>
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-8 max-w-5xl mx-auto py-8 px-4">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold text-zinc-100 tracking-tight">Settings</h1>
          <p className="text-zinc-500 mt-1">
            Configure your analytics experience and notifications
          </p>
        </div>
        <div className="flex items-center gap-3">
          {saved && (
            <Badge className="bg-emerald-500/15 text-emerald-400 border border-emerald-500/30 animate-in fade-in zoom-in duration-300">
              <CheckCircle2 className="h-3.5 w-3.5 mr-1.5" />
              Changes saved
            </Badge>
          )}
          <Button 
            onClick={savePreferences} 
            disabled={saving}
            className="bg-amber-500 hover:bg-amber-600 text-zinc-950 font-bold px-6"
          >
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
        <div className="p-4 bg-rose-500/10 border border-rose-500/30 rounded-lg text-rose-400 text-sm flex items-center gap-3">
          <RefreshCw className="h-4 w-4" />
          {error}
        </div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Notifications */}
        <Card className="bg-zinc-900/50 border-zinc-800">
          <CardHeader className="flex flex-row items-center gap-2 space-y-0">
            <Bell className="h-5 w-5 text-amber-400" />
            <CardTitle className="text-lg">Notifications</CardTitle>
          </CardHeader>
          <CardContent className="space-y-5">
            <div className="flex items-center justify-between">
              <div className="space-y-0.5">
                <Label className="text-sm font-medium">Lineup Deadline</Label>
                <p className="text-xs text-zinc-500">Alert before MLB first pitch</p>
              </div>
              <Switch 
                checked={preferences.notifications.lineup_deadline} 
                onCheckedChange={(v) => updateNested("notifications", "lineup_deadline", v)}
              />
            </div>
            <div className="flex items-center justify-between">
              <div className="space-y-0.5">
                <Label className="text-sm font-medium">Injury Alerts</Label>
                <p className="text-xs text-zinc-500">Instant notification when status changes</p>
              </div>
              <Switch 
                checked={preferences.notifications.injury_alerts} 
                onCheckedChange={(v) => updateNested("notifications", "injury_alerts", v)}
              />
            </div>
            <div className="flex items-center justify-between">
              <div className="space-y-0.5">
                <Label className="text-sm font-medium">Waiver Suggestions</Label>
                <p className="text-xs text-zinc-500">High-priority waiver targets</p>
              </div>
              <Switch 
                checked={preferences.notifications.waiver_suggestions} 
                onCheckedChange={(v) => updateNested("notifications", "waiver_suggestions", v)}
              />
            </div>
            <div className="flex items-center justify-between border-t border-zinc-800 pt-4 mt-4">
              <div className="space-y-0.5">
                <Label className="text-sm font-medium">Email Alerts</Label>
                <p className="text-xs text-zinc-500">Send summaries via email</p>
              </div>
              <Switch 
                checked={preferences.notifications.email_enabled} 
                onCheckedChange={(v) => updateNested("notifications", "email_enabled", v)}
              />
            </div>
          </CardContent>
        </Card>

        {/* Dashboard Layout */}
        <Card className="bg-zinc-900/50 border-zinc-800">
          <CardHeader className="flex flex-row items-center gap-2 space-y-0">
            <Layout className="h-5 w-5 text-sky-400" />
            <CardTitle className="text-lg">Dashboard</CardTitle>
          </CardHeader>
          <CardContent className="space-y-6">
            <div className="space-y-3">
              <Label className="text-sm font-medium">Auto-Refresh Interval</Label>
              <Select 
                value={String(preferences.dashboard_layout.refresh_interval_seconds)}
                onValueChange={(v) => updateNested("dashboard_layout", "refresh_interval_seconds", Number(v))}
              >
                <SelectTrigger className="bg-zinc-950 border-zinc-800 text-zinc-200">
                  <SelectValue placeholder="Select interval" />
                </SelectTrigger>
                <SelectContent>
                  {REFRESH_OPTIONS.map(opt => (
                    <SelectItem key={opt.value} value={opt.value}>{opt.label}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-3">
              <Label className="text-sm font-medium">Visual Theme</Label>
              <div className="flex gap-2">
                {["dark", "light", "system"].map((t) => (
                  <Button
                    key={t}
                    variant="default"
                    size="sm"
                    className={cn(
                      "flex-1 capitalize",
                      preferences.dashboard_layout.theme === t 
                        ? "bg-zinc-800 border-zinc-600 text-zinc-100" 
                        : "bg-transparent border-zinc-800 text-zinc-500 hover:text-zinc-300"
                    )}
                    onClick={() => updateNested("dashboard_layout", "theme", t)}
                  >
                    {t}
                  </Button>
                ))}
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Streak Settings */}
        <Card className="bg-zinc-900/50 border-zinc-800 md:col-span-2">
          <CardHeader className="flex flex-row items-center gap-2 space-y-0">
            <TrendingUp className="h-5 w-5 text-rose-400" />
            <CardTitle className="text-lg">Streak Detection</CardTitle>
          </CardHeader>
          <CardContent className="space-y-8">
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
              {Z_PRESETS.map((preset) => (
                <button
                  key={preset.id}
                  onClick={() => applyPreset(preset.id)}
                  className={cn(
                    "flex flex-col items-start p-4 rounded-lg border text-left transition-all",
                    getActivePreset() === preset.id
                      ? "bg-zinc-800/80 border-zinc-600 ring-1 ring-zinc-600"
                      : "bg-zinc-950 border-zinc-800 hover:border-zinc-700"
                  )}
                >
                  <span className={cn(
                    "text-sm font-bold mb-1",
                    getActivePreset() === preset.id ? "text-zinc-100" : "text-zinc-400"
                  )}>
                    {preset.label}
                  </span>
                  <span className="text-xs text-zinc-500 mb-3">{preset.description}</span>
                  <div className="flex gap-3 mt-auto">
                    <div className="flex flex-col">
                      <span className="text-[10px] uppercase text-zinc-600 font-bold">Hot</span>
                      <span className="text-xs font-mono text-emerald-400">+{preset.hot.toFixed(1)}</span>
                    </div>
                    <div className="flex flex-col">
                      <span className="text-[10px] uppercase text-zinc-600 font-bold">Cold</span>
                      <span className="text-xs font-mono text-rose-400">{preset.cold.toFixed(1)}</span>
                    </div>
                  </div>
                </button>
              ))}
            </div>

            <div className="space-y-6 border-t border-zinc-800 pt-6">
              <div className="space-y-4">
                <div className="flex justify-between items-center">
                  <Label className="text-sm font-medium">Minimum Sample Size</Label>
                  <span className="text-sm font-mono text-amber-400">{preferences.streak_settings.min_sample_days} days</span>
                </div>
                <Slider 
                  value={[preferences.streak_settings.min_sample_days]} 
                  min={3} 
                  max={21} 
                  step={1} 
                  onValueChange={([v]) => updateNested("streak_settings", "min_sample_days", v)}
                />
                <p className="text-[10px] text-zinc-600">
                  Minimum consecutive days required to trigger a streak alert.
                </p>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Waiver Wire */}
        <Card className="bg-zinc-900/50 border-zinc-800 md:col-span-2">
          <CardHeader className="flex flex-row items-center gap-2 space-y-0">
            <Search className="h-5 w-5 text-emerald-400" />
            <CardTitle className="text-lg">Waiver Preferences</CardTitle>
          </CardHeader>
          <CardContent className="space-y-8">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
              <div className="space-y-6">
                <div className="space-y-4">
                  <div className="flex justify-between items-center">
                    <Label className="text-sm font-medium">Ownership Range</Label>
                    <span className="text-sm font-mono text-zinc-400">
                      {preferences.waiver_preferences.min_percent_owned}% - {preferences.waiver_preferences.max_percent_owned}%
                    </span>
                  </div>
                  <div className="space-y-8 pt-2">
                    <div className="space-y-2">
                      <div className="flex justify-between text-[10px] text-zinc-500 uppercase font-bold">
                        <span>Min Owned</span>
                        <span>{preferences.waiver_preferences.min_percent_owned}%</span>
                      </div>
                      <Slider 
                        value={[preferences.waiver_preferences.min_percent_owned]} 
                        min={0} 
                        max={preferences.waiver_preferences.max_percent_owned} 
                        step={5} 
                        onValueChange={([v]) => updateNested("waiver_preferences", "min_percent_owned", v)}
                      />
                    </div>
                    <div className="space-y-2">
                      <div className="flex justify-between text-[10px] text-zinc-500 uppercase font-bold">
                        <span>Max Owned</span>
                        <span>{preferences.waiver_preferences.max_percent_owned}%</span>
                      </div>
                      <Slider 
                        value={[preferences.waiver_preferences.max_percent_owned]} 
                        min={preferences.waiver_preferences.min_percent_owned} 
                        max={100} 
                        step={5} 
                        onValueChange={([v]) => updateNested("waiver_preferences", "max_percent_owned", v)}
                      />
                    </div>
                  </div>
                </div>
              </div>

              <div className="space-y-6">
                <div className="flex items-center justify-between p-4 bg-zinc-950 border border-zinc-800 rounded-lg">
                  <div className="space-y-0.5">
                    <Label className="text-sm font-medium">Hide Injured Players</Label>
                    <p className="text-xs text-zinc-500">Filter out DL/IL players from waiver list</p>
                  </div>
                  <Switch 
                    checked={preferences.waiver_preferences.hide_injured} 
                    onCheckedChange={(v) => updateNested("waiver_preferences", "hide_injured", v)}
                  />
                </div>

                <div className="space-y-4">
                  <div className="flex justify-between items-center">
                    <Label className="text-sm font-medium">Streamer Threshold</Label>
                    <span className="text-sm font-mono text-emerald-400">
                      {(preferences.waiver_preferences.streamer_threshold * 100).toFixed(0)}%
                    </span>
                  </div>
                  <Slider 
                    value={[preferences.waiver_preferences.streamer_threshold * 100]} 
                    min={10} 
                    max={90} 
                    step={5} 
                    onValueChange={([v]) => updateNested("waiver_preferences", "streamer_threshold", v / 100)}
                  />
                  <p className="text-[10px] text-zinc-600">
                    Minimum win probability gain required to recommend a streaming move.
                  </p>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
