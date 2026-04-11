# K-23: Settings Page UAT Deep Dive — "Developer JSON Dump" Analysis

**Date:** April 1, 2026  
**Analyst:** Kimi CLI (Deep Intelligence Unit)  
**Scope:** Technical root cause of Settings page "raw database JSON" UX failure  
**Status:** CRITICAL UX FAILURE — Page is read-only data dump, not interactive settings

---

## Executive Summary

The Settings page is a **"developer accidentally printed a raw database JSON file directly onto the screen."** This is not a settings page — it's a **read-only data viewer** that destroys user trust.

**The Problem:**
- `{"channels": ["discord"]}` rendered as raw text instead of Discord OAuth button
- `300 s` shown instead of "Refresh every 5 minutes" dropdown
- `0.5` z-score with no context instead of "Aggressive/Conservative" slider
- **Zero interactive controls** — toggles don't toggle, sliders don't slide

**The Opportunity:**
These settings are the **"holy grail"** for elite fantasy managers. Fixed UI = premium differentiation from Yahoo/ESPN.

---

## Part 1: Critical UI & Logic Failures

### Issue 1.1: Raw Data Leaks — Developer JSON Visible to Users

**User Impact:** Shatters illusion of finished product. Users see `{"discord"]` and `null` values.

**Root Cause Analysis:**

```typescript
// frontend/app/(dashboard)/settings/page.tsx lines 120-127
{Object.entries(preferences.notifications).map(([key, value]) => (
  <div key={key} className="flex items-center justify-between">
    <span className="capitalize">{key.replace(/_/g, " ")}</span>
    <span className="text-muted-foreground">
      {typeof value === "boolean" ? (value ? "On" : "Off") : JSON.stringify(value)}
    </span>
  </div>
))}
```

**The smoking gun:** Line 124 uses `JSON.stringify(value)` for non-boolean values!

When `value = ["discord"]`, user sees: `["discord"]`  
When `value = null`, user sees: `null`  
When `value = {foo: "bar"}`, user sees: `{"foo":"bar"}`

**The Fix:** Replace JSON.stringify with human-readable UI components:

```typescript
// BEFORE (line 124):
{typeof value === "boolean" ? (value ? "On" : "Off") : JSON.stringify(value)}

// AFTER:
{typeof value === "boolean" ? (
  <Switch checked={value} onCheckedChange={(checked) => updateNotification(key, checked)} />
) : key === "channels" ? (
  <div className="flex gap-2">
    {(value as string[]).map(channel => (
      <Badge key={channel} variant="outline">
        {channel === "discord" && <DiscordLogo className="h-3 w-3 mr-1" />}
        {channel}
      </Badge>
    ))}
    <Button size="sm" variant="outline" onClick={connectDiscord}>
      Connect Discord
    </Button>
  </div>
) : key === "discord_user_id" ? (
  value ? (
    <span className="text-emerald-600">Connected</span>
  ) : (
    <span className="text-zinc-400">Not connected</span>
  )
) : (
  String(value)
)}
```

**Complexity:** LOW  
**Priority:** CRITICAL — Core UX broken

---

### Issue 1.2: Dev-Speak Labels — "Refresh interval: 300 s"

**User Impact:** Users must do math (300s = 5min) to understand settings. Computers talking to humans.

**Root Cause Analysis:**

```typescript
// settings/page.tsx lines 137-138
<p className="text-sm text-muted-foreground mb-4">
  Refresh interval: {preferences.dashboard_layout.refresh_interval_seconds}s
</p>
```

**The Fix:** Human-readable dropdown:

```typescript
// AFTER:
const REFRESH_OPTIONS = [
  { value: 60, label: "1 minute" },
  { value: 300, label: "5 minutes" },
  { value: 600, label: "10 minutes" },
  { value: 1800, label: "30 minutes" },
  { value: 3600, label: "1 hour" },
]

// In JSX:
<Select 
  value={String(preferences.dashboard_layout.refresh_interval_seconds)}
  onValueChange={(val) => updateRefreshInterval(Number(val))}
>
  <SelectTrigger>
    <Clock className="h-4 w-4 mr-2" />
    {REFRESH_OPTIONS.find(o => o.value === preferences.dashboard_layout.refresh_interval_seconds)?.label}
  </SelectTrigger>
  <SelectContent>
    {REFRESH_OPTIONS.map(opt => (
      <SelectItem key={opt.value} value={String(opt.value)}>{opt.label}</SelectItem>
    ))}
  </SelectContent>
</Select>
```

**Complexity:** LOW  
**Priority:** HIGH — Usability

---

### Issue 1.3: Zero Interactivity — Settings Are Read-Only

**User Impact:** "Save Changes" button is a placebo. No toggles, sliders, or inputs actually work.

**Root Cause Analysis:**

Looking at the entire settings page, there are **zero interactive controls**:

| Setting | Current Display | Missing Interactive Element |
|---------|-----------------|---------------------------|
| `lineup_deadline: true` | Text "On" | Toggle switch |
| `channels: ["discord"]` | JSON `["discord"]` | OAuth button + toggle per channel |
| `refresh_interval_seconds: 300` | Text "300s" | Dropdown/select |
| `panels[].enabled: true` | Badge "Enabled" | Toggle switch per panel |
| `hot_threshold: 0.5` | Text "0.5" | Slider or Aggressive/Conservative selector |
| `min_percent_owned: 0` | Text "0%" | Dual-handle range slider (0-100%) |

**The Fix:** Implement interactive controls for each setting type:

```typescript
// Example: Notification toggles with actual state updates
function NotificationToggle({ 
  label, 
  checked, 
  onChange 
}: { 
  label: string; 
  checked: boolean; 
  onChange: (checked: boolean) => void 
}) {
  return (
    <div className="flex items-center justify-between py-2">
      <div>
        <p className="font-medium">{label}</p>
        <p className="text-xs text-zinc-500">Get notified when this happens</p>
      </div>
      <Switch checked={checked} onCheckedChange={onChange} />
    </div>
  )
}

// Usage:
<NotificationToggle
  label="Lineup Deadline Alerts"
  checked={preferences.notifications.lineup_deadline}
  onChange={(checked) => updatePreferences({
    notifications: { ...preferences.notifications, lineup_deadline: checked }
  })}
/>
```

**Complexity:** MEDIUM — Requires state management updates  
**Priority:** CRITICAL — Page is non-functional

---

### Issue 1.4: The Z-Score Alienation — "Hot threshold (z-score) 0.5"

**User Impact:** 90% of users don't understand z-scores. Elite managers don't want to do math to configure their dashboard.

**Root Cause Analysis:**

```typescript
// settings/page.tsx lines 159-171
<div className="flex justify-between">
  <span>Hot threshold (z-score)</span>
  <span className="font-medium">{preferences.streak_settings.hot_threshold}</span>
</div>
```

**The Fix:** Hide the math behind a human-friendly slider:

```typescript
// AFTER:
const STREAK_MODES = [
  { 
    value: 0.3, 
    label: "Aggressive", 
    description: "Flags players after just a few good games" 
  },
  { 
    value: 0.5, 
    label: "Balanced", 
    description: "Standard streak detection" 
  },
  { 
    value: 0.8, 
    label: "Conservative", 
    description: "Requires long track record of excellence" 
  },
]

// In JSX:
<div className="space-y-4">
  <Label>Streak Detection Sensitivity</Label>
  <RadioGroup 
    value={String(preferences.streak_settings.hot_threshold)}
    onValueChange={(val) => updateStreakSettings({ hot_threshold: Number(val) })}
  >
    {STREAK_MODES.map(mode => (
      <div key={mode.value} className="flex items-start space-x-3 py-2">
        <RadioGroupItem value={String(mode.value)} />
        <div>
          <p className="font-medium">{mode.label}</p>
          <p className="text-xs text-zinc-500">{mode.description}</p>
        </div>
      </div>
    ))}
  </RadioGroup>
</div>
```

**Complexity:** LOW  
**Priority:** HIGH — User comprehension

---

## Part 2: The Elite Blueprint — Why These Settings Are Pure Gold

### Feature 2.1: God-Tier Notifications (Discord Integration)

**Current State:** `{"channels": ["discord"], "discord_user_id": null}` as JSON text

**Elite Manager Need:** "Ping my Discord 15 minutes before first pitch: 'Juan Soto is sitting today'"

**Implementation:**

```typescript
// Notification section with working Discord OAuth
function DiscordIntegration() {
  const { discord_user_id } = preferences.notifications
  
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <DiscordLogo className="h-5 w-5" />
          Discord Notifications
        </CardTitle>
      </CardHeader>
      <CardContent>
        {discord_user_id ? (
          <div className="space-y-4">
            <div className="flex items-center gap-2 text-emerald-600">
              <CheckCircle2 className="h-4 w-4" />
              <span>Connected to Discord</span>
            </div>
            
            {/* Toggle per notification type */}
            <div className="space-y-2">
              <NotificationToggle
                label="Lineup Deadline Alerts"
                description="15 min before first pitch if starter benched"
                checked={preferences.notifications.lineup_deadline}
                onChange={...}
              />
              <NotificationToggle
                label="Injury Alerts"
                description="When rostered player goes on IL"
                checked={preferences.notifications.injury_alerts}
                onChange={...}
              />
              <NotificationToggle
                label="Trade Offers"
                description="When you receive a trade proposal"
                checked={preferences.notifications.trade_offers}
                onChange={...}
              />
              <NotificationToggle
                label="Waiver Suggestions"
                description="Daily top waiver targets for your roster"
                checked={preferences.notifications.waiver_suggestions}
                onChange={...}
              />
            </div>
            
            <Button variant="outline" onClick={disconnectDiscord}>
              Disconnect
            </Button>
          </div>
        ) : (
          <div className="text-center py-6">
            <p className="text-zinc-500 mb-4">
              Connect Discord for real-time lineup alerts
            </p>
            <Button onClick={initiateDiscordOAuth}>
              <DiscordLogo className="h-4 w-4 mr-2" />
              Connect Discord
            </Button>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
```

**Backend Requirements:**
- Discord OAuth flow endpoint
- Webhook or DM integration
- Notification queue/jobs

**Complexity:** HIGH — Requires OAuth + webhook infrastructure  
**Priority:** HIGH — Premium feature

---

### Feature 2.2: Waiver Preferences — Fix for Page 3 Issues

**Current State:** Min/Max ownership shown as static text (0%, 60%)

**User UAT Feedback:** Hated that max ownership was stuck at 90% on Waiver Wire page

**Implementation:**

```typescript
// Dual-handle slider for ownership range
function OwnershipRangeSlider() {
  const { min_percent_owned, max_percent_owned } = preferences.waiver_preferences
  
  return (
    <Card>
      <CardHeader>
        <CardTitle>Waiver Wire Ownership Filter</CardTitle>
        <CardDescription>
          Show players between {min_percent_owned}% and {max_percent_owned}% owned
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="pt-6 pb-2">
          <DualRangeSlider
            value={[min_percent_owned, max_percent_owned]}
            min={0}
            max={100}
            step={5}
            onValueChange={([min, max]) => {
              updateWaiverPreferences({
                min_percent_owned: min,
                max_percent_owned: max
              })
            }}
          />
        </div>
        <div className="flex justify-between text-xs text-zinc-500">
          <span>0%</span>
          <span>25%</span>
          <span>50%</span>
          <span>75%</span>
          <span>100%</span>
        </div>
        
        {/* Preset buttons */}
        <div className="flex gap-2 mt-4">
          <Button 
            size="sm" 
            variant="outline"
            onClick={() => updateWaiverPreferences({ min_percent_owned: 0, max_percent_owned: 25 })}
          >
            Deep League (0-25%)
          </Button>
          <Button 
            size="sm" 
            variant="outline"
            onClick={() => updateWaiverPreferences({ min_percent_owned: 10, max_percent_owned: 40 })}
          >
            Competitive (10-40%)
          </Button>
          <Button 
            size="sm" 
            variant="outline"
            onClick={() => updateWaiverPreferences({ min_percent_owned: 0, max_percent_owned: 60 })}
          >
            Wide Net (0-60%)
          </Button>
        </div>
      </CardContent>
    </Card>
  )
}
```

**Connection to Waiver Wire:**
These preferences should be **automatically applied** when loading the Waiver Wire page:

```typescript
// waiver/page.tsx — Load user preferences on mount
const { data: prefs } = useQuery({
  queryKey: ['user-preferences'],
  queryFn: endpoints.getUserPreferences
})

// Set default maxOwned from preferences
useEffect(() => {
  if (prefs?.waiver_preferences) {
    setMaxOwned(prefs.waiver_preferences.max_percent_owned)
  }
}, [prefs])
```

**Complexity:** LOW — Slider component + preference wiring  
**Priority:** HIGH — Fixes Waiver Wire issue

---

### Feature 2.3: Dashboard Layout — Drag-and-Drop Widgets

**Current State:** Static list of panels with "Enabled" badge

**Elite Manager Need:** "Rearrange or hide widgets to match my workflow"

**Implementation:**

```typescript
// Drag-and-drop dashboard layout
import { DndContext, useSortable } from '@dnd-kit/sortable'

function DashboardLayoutEditor() {
  const panels = preferences.dashboard_layout.panels
  
  return (
    <Card>
      <CardHeader>
        <CardTitle>Dashboard Layout</CardTitle>
        <CardDescription>
          Drag to reorder, toggle to show/hide widgets
        </CardDescription>
      </CardHeader>
      <CardContent>
        <DndContext onDragEnd={handleDragEnd}>
          <div className="space-y-2">
            {panels.map((panel) => (
              <SortablePanelItem 
                key={panel.id} 
                panel={panel}
                onToggle={(enabled) => updatePanel(panel.id, { enabled })}
              />
            ))}
          </div>
        </DndContext>
      </CardContent>
    </Card>
  )
}

function SortablePanelItem({ panel, onToggle }) {
  const { attributes, listeners, setNodeRef, transform, transition } = useSortable({
    id: panel.id
  })
  
  const panelLabels = {
    lineup_gaps: "Lineup Gaps",
    hot_cold_streaks: "Hot/Cold Streaks",
    waiver_targets: "Waiver Targets",
    injury_flags: "Injury Flags",
    matchup_preview: "Matchup Preview",
    probable_pitchers: "Probable Pitchers",
  }
  
  return (
    <div
      ref={setNodeRef}
      style={{ transform: CSS.Transform.toString(transform), transition }}
      className="flex items-center justify-between p-3 border rounded bg-zinc-900"
    >
      <div className="flex items-center gap-3">
        <button {...attributes} {...listeners} className="cursor-grab">
          <GripVertical className="h-4 w-4 text-zinc-500" />
        </button>
        <span className="font-medium">{panelLabels[panel.id] || panel.id}</span>
      </div>
      <Switch checked={panel.enabled} onCheckedChange={onToggle} />
    </div>
  )
}
```

**Complexity:** MEDIUM — Requires @dnd-kit integration  
**Priority:** MEDIUM — UX enhancement

---

## Part 3: Implementation Roadmap

### Phase 1: Critical Fixes (Immediate)

| Issue | File | Lines | Fix | Complexity |
|-------|------|-------|-----|------------|
| 1.1 JSON.stringify | `settings/page.tsx` | 124 | Replace with human-readable components | LOW |
| 1.2 Refresh interval | `settings/page.tsx` | 137-138 | Add human-readable dropdown | LOW |
| 1.3 Interactivity | `settings/page.tsx` | 120-195 | Add Switch, Select, Slider components | MEDIUM |
| 1.4 Z-score | `settings/page.tsx` | 159-171 | Hide math behind Aggressive/Conservative selector | LOW |

### Phase 2: Elite Features (Next Sprint)

| Feature | Implementation | Complexity |
|---------|----------------|------------|
| 2.1 Discord OAuth | Backend OAuth endpoint + frontend connect button | HIGH |
| 2.2 Ownership Slider | Dual-handle range slider with presets | LOW |
| 2.3 Layout DnD | @dnd-kit sortable widgets | MEDIUM |

---

## Part 4: Component Library Requirements

Add these shadcn/ui components:

```bash
cd frontend
npx shadcn add switch        # For toggles
npx shadcn add slider        # For range inputs
npx shadcn add select        # For dropdowns
npx shadcn add radio-group   # For streak mode selector
npx shadcn add tooltip       # For help text
npm install @dnd-kit/sortable @dnd-kit/core  # For drag-and-drop
```

---

## Summary for Claude Code

**Current State:** Settings page is a **read-only JSON viewer**, not a functional settings interface.

**Critical Fixes Needed:**
1. Replace `JSON.stringify()` with human-readable UI components
2. Add interactive controls (Switch, Select, Slider) that actually update state
3. Convert "300s" to "5 minutes" with dropdown
4. Hide z-score math behind "Aggressive/Balanced/Conservative" selector

**Elite Features:**
1. **Discord Integration:** OAuth + notification toggles
2. **Ownership Slider:** Dual-handle range (0-100%) with league presets
3. **Layout Editor:** Drag-and-drop widget reordering

**Files to Modify:**
- `frontend/app/(dashboard)/settings/page.tsx` (complete rewrite of display logic)
- `frontend/lib/types.ts` (ensure UserPreferences type is complete)
- Add shadcn components: switch, slider, select, radio-group

**Estimated Time:** 
- Phase 1 (Critical): 1 session
- Phase 2 (Elite): 2 sessions

---

*Analysis complete. Settings page has strong backend infrastructure but completely broken frontend UX. Fix = premium differentiation from generic fantasy platforms.*
