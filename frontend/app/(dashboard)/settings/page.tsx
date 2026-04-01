"use client"

import { useEffect, useState } from "react"
import { endpoints } from "@/lib/api"
import type { UserPreferences } from "@/lib/types"
import { Card, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import {
  CheckCircle2,
  Save,
  RefreshCw,
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
      await endpoints.updateUserPreferences(preferences)
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
            <div key={i} className="h-64 bg-gray-200 rounded" />
          ))}
        </div>
      </div>
    )
  }

  if (!preferences) {
    return (
      <div className="container mx-auto py-8 px-4">
        <h1 className="text-3xl font-bold mb-8">Settings</h1>
        <div className="p-4 bg-red-50 border border-red-200 rounded-lg">
          <p className="text-red-800">Failed to load preferences. Please try refreshing.</p>
        </div>
      </div>
    )
  }

  return (
    <div className="container mx-auto py-8 px-4 max-w-5xl">
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-3xl font-bold">Settings</h1>
          <p className="text-muted-foreground mt-1">
            Customize your dashboard experience
          </p>
        </div>
        <div className="flex items-center gap-2">
          {saved && (
            <Badge className="bg-green-100 text-green-800">
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
        <div className="p-4 mb-6 bg-red-50 border border-red-200 rounded-lg">
          <p className="text-red-800">{error}</p>
        </div>
      )}

      <div className="space-y-6">
        {/* Notifications */}
        <Card>
          <CardHeader>
            <CardTitle>Notifications</CardTitle>
          </CardHeader>
          <div className="p-6 pt-0 space-y-4">
            {Object.entries(preferences.notifications).map(([key, value]) => (
              <div key={key} className="flex items-center justify-between">
                <span className="capitalize">{key.replace(/_/g, " ")}</span>
                <span className="text-muted-foreground">
                  {typeof value === "boolean" ? (value ? "On" : "Off") : JSON.stringify(value)}
                </span>
              </div>
            ))}
          </div>
        </Card>

        {/* Dashboard Layout */}
        <Card>
          <CardHeader>
            <CardTitle>Dashboard Layout</CardTitle>
          </CardHeader>
          <div className="p-6 pt-0">
            <p className="text-sm text-muted-foreground mb-4">
              Refresh interval: {preferences.dashboard_layout.refresh_interval_seconds}s
            </p>
            <div className="space-y-2">
              {preferences.dashboard_layout.panels.map((panel) => (
                <div key={panel.id} className="flex items-center justify-between p-2 border rounded">
                  <span className="capitalize">{panel.id.replace(/_/g, " ")}</span>
                  <Badge variant={panel.enabled ? "default" : "secondary"}>
                    {panel.enabled ? "Enabled" : "Disabled"}
                  </Badge>
                </div>
              ))}
            </div>
          </div>
        </Card>

        {/* Streak Settings */}
        <Card>
          <CardHeader>
            <CardTitle>Streak Detection</CardTitle>
          </CardHeader>
          <div className="p-6 pt-0 space-y-4">
            <div className="flex justify-between">
              <span>Hot threshold (z-score)</span>
              <span className="font-medium">{preferences.streak_settings.hot_threshold}</span>
            </div>
            <div className="flex justify-between">
              <span>Cold threshold (z-score)</span>
              <span className="font-medium">{preferences.streak_settings.cold_threshold}</span>
            </div>
            <div className="flex justify-between">
              <span>Rolling windows</span>
              <span className="font-medium">{preferences.streak_settings.rolling_windows.join(", ")} days</span>
            </div>
          </div>
        </Card>

        {/* Waiver Settings */}
        <Card>
          <CardHeader>
            <CardTitle>Waiver Preferences</CardTitle>
          </CardHeader>
          <div className="p-6 pt-0 space-y-4">
            <div className="flex justify-between">
              <span>Min ownership %</span>
              <span className="font-medium">{preferences.waiver_preferences.min_percent_owned}%</span>
            </div>
            <div className="flex justify-between">
              <span>Max ownership %</span>
              <span className="font-medium">{preferences.waiver_preferences.max_percent_owned}%</span>
            </div>
            <div className="flex justify-between">
              <span>Hide injured</span>
              <Badge variant={preferences.waiver_preferences.hide_injured ? "default" : "secondary"}>
                {preferences.waiver_preferences.hide_injured ? "Yes" : "No"}
              </Badge>
            </div>
          </div>
        </Card>
      </div>
    </div>
  )
}
