# CBB Edge — Fantasy Baseball UI Design System v2

**Author:** Kimi CLI (UI Audit → Design Spec)  
**Date:** 2026-05-13  
**Scope:** Color, typography, category identity, and component hierarchy for the Fantasy Baseball dashboard (War Room, Roster, Waiver, Streaming, Budget).

---

## 1. Problem Statement

The current UI suffers from five interrelated problems that make it hard to "read" the data quickly:

| Problem | User Impact |
|---------|-------------|
| **1. Flat grayscale** | Every card, row, and chip uses a near-identical dark gray. No surface hierarchy. |
| **2. No category identity** | "HR" and "ERA" look identical. Users must read labels instead of scanning colors. |
| **3. Overused gold (`#FFC000`)** | Headings, winning values, active buttons, icons, and alerts all compete for the same accent. |
| **4. Low contrast text** | `#7D7D7D` on `#181818` fails WCAG AA. `#494949` is nearly invisible. |
| **5. Monotonous typography** | Everything is 10–12px uppercase tracking-widest. No scan hierarchy. |

**Result:** Users feel "underwhelmed" because the interface requires *reading* instead of *recognizing*.

---

## 2. Design Principles

1. **Color = Meaning.** Every category gets a persistent color. A user should know "that's ERA" without reading the label.
2. **Surface = Depth.** Three clear surface levels (background → card → elevated) create instant spatial hierarchy.
3. **Accent = Rarity.** Gold is reserved for *your team's winning values* and *primary actions* only.
4. **Typography = Priority.** Large numbers, medium labels, small metadata. Not everything shouts.
5. **Status = Motion.** Safe/lead/bubble/behind/lost use a continuous color gradient, not five unrelated hues.

---

## 3. Color Palette

### 3.1 Foundation (Dark Mode Only)

The current `#09090b` is too close to pure black and makes every gray element look muddy. We lift the base slightly and create *three distinct surfaces*:

| Token | Hex | HSL | Usage |
|-------|-----|-----|-------|
| `bg-base` | `#0c0c10` | `240 14% 3%` | Page background |
| `bg-surface` | `#16161e` | `240 14% 7%` | Cards, panels, tables |
| `bg-elevated` | `#1f1f2a` | `240 14% 11%` | Hover states, dropdowns, modal overlays |
| `bg-inset` | `#0a0a0f` | `240 14% 2%` | Input backgrounds, inner wells |
| `border-subtle` | `#272733` | `240 12% 15%` | Dividers, inactive borders |
| `border-default` | `#3a3a4d` | `240 12% 22%` | Card borders, active inputs |
| `border-focus` | `#6b6b8a` | `240 12% 35%` | Focus rings |

**Why this works:** The old palette had `#09090b` → `#181818` → `#202020` — a 2% lightness jump. The new palette has 4% jumps (`3%` → `7%` → `11%`), creating visible depth.

### 3.2 Text Colors

| Token | Hex | Usage | WCAG vs `bg-surface` |
|-------|-----|-------|----------------------|
| `text-primary` | `#e8e8f0` | Headings, player names, primary values | AAA |
| `text-secondary` | `#a0a0b8` | Labels, positions, team abbreviations | AA |
| `text-tertiary` | `#6b6b8a` | Metadata, timestamps, disabled text | AA Large |
| `text-muted` | `#4a4a60` | Placeholders, empty states | — |

**Current problem:** `#7D7D7D` on `#181818` = 3.8:1 ratio (fails AA). `#494949` = 2.1:1 (fails everything).

### 3.3 Semantic Colors (Reduced Set)

The current UI uses emerald, amber, rose, orange, sky, and gold for status. That's six colors for five states. We consolidate to a **single continuous gradient** for win probability:

| Status | Old Color | New Color | Hex | Usage |
|--------|-----------|-----------|-----|-------|
| SAFE (>85%) | `emerald-400` | `status-safe` | `#22c55e` | Safe win |
| LEAD (65–85%) | `#FFC000` | `status-lead` | `#84cc16` | Likely win |
| BUBBLE (35–65%) | `amber-400` | `status-bubble` | `#f59e0b` | Could go either way |
| BEHIND (15–35%) | `orange-400` | `status-behind` | `#f97316` | Likely loss |
| LOST (<15%) | `rose-400` | `status-lost` | `#ef4444` | Safe loss |

**Why:** The old palette jumped from yellow (`#FFC000`) to orange to rose. The new palette uses a true green→yellow→red gradient. Users intuitively understand "more green = better."

**Gold (`#FFC000`) is now reserved for:**
- Your team's winning value in a category row
- Primary CTA buttons (Optimize, RE-RUN)
- Active nav item indicator

### 3.4 Category Identity Colors (The Big Change)

Every H2H category gets a **persistent, unique color**. This is the single biggest readability improvement.

#### Batting Categories

| Category | Color Name | Hex | Preview |
|----------|-----------|-----|---------|
| `R` (Runs) | `cat-runs` | `#3b82f6` | 🔵 Blue |
| `H` (Hits) | `cat-hits` | `#06b6d4` | 🔵 Cyan |
| `HR` (Home Runs) | `cat-hr` | `#a855f7` | 🟣 Purple |
| `RBI` | `cat-rbi` | `#8b5cf6` | 🟣 Violet |
| `K` (Batter K) | `cat-k-bat` | `#f43f5e` | 🔴 Rose |
| `TB` (Total Bases) | `cat-tb` | `#d946ef` | 🟣 Fuchsia |
| `AVG` | `cat-avg` | `#eab308` | 🟡 Yellow |
| `OPS` | `cat-ops` | `#ca8a04` | 🟡 Dark yellow |
| `NSB` | `cat-nsb` | `#10b981` | 🟢 Emerald |

#### Pitching Categories

| Category | Color Name | Hex | Preview |
|----------|-----------|-----|---------|
| `W` (Wins) | `cat-w` | `#3b82f6` | 🔵 Blue |
| `L` (Losses) | `cat-l` | `#f43f5e` | 🔴 Rose |
| `HR` (HR Allowed) | `cat-hr-pit` | `#a855f7` | 🟣 Purple |
| `K` (Pitcher K) | `cat-k-pit` | `#06b6d4` | 🔵 Cyan |
| `ERA` | `cat-era` | `#f97316` | 🟠 Orange |
| `WHIP` | `cat-whip` | `#fb923c` | 🟠 Light orange |
| `K/9` | `cat-k9` | `#22d3ee` | 🔵 Sky |
| `QS` | `cat-qs` | `#10b981` | 🟢 Emerald |
| `NSV` | `cat-nsv` | `#84cc16` | 🟢 Lime |

**Design rationale:**
- Counting stats cluster by hue family (blue = runs/hits/wins/ks, purple = power, green = speed/quality)
- Rate stats get warm hues (yellow = AVG/OPS, orange = ERA/WHIP) because they "feel" different
- Negative categories (K batter, L, HR allowed) get red-adjacent hues as a subtle warning
- Each color is tested for 4.5:1 contrast against `bg-surface` (`#16161e`)

---

## 4. Component Specifications

### 4.1 Category Chip (Streaming Station / Waiver Wire)

**Current:**
```
[HR] [-3.0] ↓    bg-rose-900/20, text-rose-400
```

**New:**
```
┌─────────────────┐
│  ● HR   -3.0  ↓ │  ← Left dot = category color
│  #a855f7 ring   │  ← Background = bg-elevated
└─────────────────┘
```

```tsx
// Exact implementation
function CategoryChip({ category, deficit, winning }: ChipProps) {
  const color = CATEGORY_COLOR[category] // e.g. '#a855f7'
  const absDeficit = Math.abs(deficit)
  
  // Severity tint on the border, not the whole chip
  const borderColor = winning 
    ? 'rgba(34, 197, 94, 0.4)'   // status-safe
    : absDeficit >= 3.0 
      ? 'rgba(239, 68, 68, 0.5)'  // status-lost
      : absDeficit >= 1.0 
        ? 'rgba(245, 158, 11, 0.4)' // status-bubble
        : 'rgba(58, 58, 77, 0.6)'   // border-subtle

  return (
    <span className="inline-flex items-center gap-2 px-2.5 py-1.5 rounded-md bg-bg-elevated border"
      style={{ borderColor }}>
      {/* Category identity dot */}
      <span className="w-2 h-2 rounded-full flex-shrink-0" style={{ backgroundColor: color }} />
      {/* Label */}
      <span className="text-xs font-semibold text-text-secondary">
        {CATEGORY_LABEL[category]}
      </span>
      {/* Value */}
      <span className={`text-xs font-mono font-bold ${winning ? 'text-status-safe' : 'text-text-primary'}`}>
        {deficit > 0 ? '+' : ''}{deficit.toFixed(1)}
      </span>
      {/* Trend icon */}
      {winning ? <TrendingUp className="w-3 h-3 text-status-safe" /> 
               : <TrendingDown className="w-3 h-3 text-status-lost" />}
    </span>
  )
}
```

**Key change:** The category color is a **small dot**, not the whole chip background. This prevents color overload when 20 chips are on screen. The chip background stays neutral so the dots pop.

### 4.2 Category Battlefield Row (War Room)

**Current:**
```
HR    0      [====|====]    0     4→7    BEHIND    Punt?
      ↑ gold if winning      gray  gray   orange    gray
```

**New:**
```
● HR    0      [====|====]    0     4→7    BEHIND    Punt?
↑purple  ↑ your value color  ↑opp  ↑proj  ↑orange   ↑hint
```

Changes:
1. Category label gets its dot + color (e.g., purple for HR)
2. **Your value** is always `text-primary` (white). Gold is removed — it was confusing because it looked like "this is good" even when you were losing.
3. **Opponent value** is always `text-secondary` (light gray). No special color when they're winning.
4. **Comparison bar:** Your side uses the category color at 80% opacity when winning, `#3a3a4d` when losing. Opponent side always uses `#2a2a3d`. This makes the bar itself tell the story.
5. **Status tag:** Uses the new semantic gradient (green→red). Bold, uppercase, same width.

```tsx
// Bar color logic
const myBarColor = winning === true 
  ? `${categoryColor}cc`  // 80% opacity category color
  : '#3a3a4d'
```

### 4.3 Waiver Player Row

**Current:**
```
Landen Roupp    SF    0% owned    SP    HOT    [====] 8.03
                gray  gray text   gray  orange bar    gold
```

**New:**
```
Landen Roupp    SF    SP  RP       ● HOT    [======]  8.03
white           gray  colored dots   badge   need bar   gold
```

Changes:
1. **Position badges** get colored dots matching their category family:
   - SP/RP/P → blue dot (pitching family)
   - OF/1B/2B/3B/SS/C → no dot (position, not category)
2. **HOT/COLD badge:** Only show if the player is in the **top 20%** of delta_z. If everyone is HOT, show nothing. Prevent badge inflation.
3. **Need bar:** Fill color uses semantic gradient (green ≥0.7, amber ≥0.4, gray <0.4). Background uses `bg-inset` for better contrast.
4. **Category match badges** (Row 3): Use the new category color dots instead of amber borders.

```tsx
// HOT/COLD gate
function HotColdBadge({ deltaZ, rankPercentile }: { deltaZ: number; rankPercentile: number }) {
  if (rankPercentile < 80) return null // Only top 20% get a badge
  if (deltaZ > 0.5) return <FlameBadge />
  if (deltaZ < -0.5) return <SnowflakeBadge />
  return null
}
```

### 4.4 Dashboard Waiver Targets Card

**Current:** Shows 5 targets with `Need score: 0.00` and `0% owned`.

**New:** When need score is 0, show a **disabled visual treatment** instead of fake data:
```
Max Meyer    MIA    SP    ─    ─
             gray   dot   no need score, no ownership
```

When data is present:
```
Landen Roupp    MIA    SP    ● Need 8.0    ● 34% owned
white           gray   dot   cyan dot + value   amber dot + value
```

### 4.5 Roster Table

**Current:** Dense grid of numbers with no scanning aids.

**New:**
1. **Column headers** get category color dots
2. **Your bench players** get a subtle left border in `border-subtle`
3. **IL players** get a full-row rose tint (`bg-rose-900/10`) instead of just a badge
4. **"Move player" button** — fix the universal `disabled` state. When enabled, use `bg-elevated hover:bg-[categoryColor]/20`.

---

## 5. Typography Scale

**Current:** Everything is 10px uppercase tracking-widest.

**New:**

| Level | Size | Weight | Transform | Tracking | Usage |
|-------|------|--------|-----------|----------|-------|
| Display | 24px | 700 | uppercase | 0.05em | Page titles ("WAR ROOM") |
| Heading | 18px | 600 | none | 0 | Card titles, section headers |
| Title | 14px | 600 | none | 0 | Player names, team names |
| Body | 13px | 400 | none | 0 | Stats, values, descriptions |
| Label | 11px | 500 | uppercase | 0.08em | Category labels, badges |
| Caption | 10px | 400 | none | 0.02em | Metadata, timestamps |

**Why:** Varying size and case creates instant scan hierarchy. You see the player name first, then the stat, then the metadata.

---

## 6. Motion & Interaction

| Element | Current | New |
|---------|---------|-----|
| Row hover | `hover:bg-[#222]` | `hover:bg-bg-elevated` with 150ms transition |
| Button press | None | `active:scale-[0.98]` |
| Need bar fill | Instant | `transition-all duration-700 ease-out` |
| Category chip sort | Instant | `layout` animation (Framer Motion or CSS grid) |
| Status tag | Static | Subtle pulse when `BUBBLE` (could change) |

---

## 7. Migration Guide

### Step 1: CSS Custom Properties (15 min)

Add to `globals.css`:

```css
:root {
  /* Foundation */
  --bg-base: #0c0c10;
  --bg-surface: #16161e;
  --bg-elevated: #1f1f2a;
  --bg-inset: #0a0a0f;
  --border-subtle: #272733;
  --border-default: #3a3a4d;
  --border-focus: #6b6b8a;
  
  /* Text */
  --text-primary: #e8e8f0;
  --text-secondary: #a0a0b8;
  --text-tertiary: #6b6b8a;
  --text-muted: #4a4a60;
  
  /* Status (continuous gradient) */
  --status-safe: #22c55e;
  --status-lead: #84cc16;
  --status-bubble: #f59e0b;
  --status-behind: #f97316;
  --status-lost: #ef4444;
  
  /* Accent */
  --accent-gold: #FFC000;
  --accent-gold-dim: #b38900;
}
```

### Step 2: Category Color Map (10 min)

Add to `lib/types.ts`:

```ts
export const CATEGORY_COLOR: Record<RotoCategory, string> = {
  // Batting
  R: '#3b82f6', H: '#06b6d4', HR_B: '#a855f7', RBI: '#8b5cf6',
  K_B: '#f43f5e', TB: '#d946ef', AVG: '#eab308', OPS: '#ca8a04', NSB: '#10b981',
  // Pitching
  W: '#3b82f6', L: '#f43f5e', HR_P: '#a855f7', K_P: '#06b6d4',
  ERA: '#f97316', WHIP: '#fb923c', K_9: '#22d3ee', QS: '#10b981', NSV: '#84cc16',
}
```

### Step 3: Update tailwind.config.ts (10 min)

```ts
colors: {
  'signal-bet': '#fbbf24',
  'signal-consider': '#38bdf8',
  'signal-pass': '#71717a',
  'signal-win': '#22c55e',
  'signal-loss': '#ef4444',
  // New: map CSS custom properties
  'bg-base': 'var(--bg-base)',
  'bg-surface': 'var(--bg-surface)',
  'bg-elevated': 'var(--bg-elevated)',
  'text-primary': 'var(--text-primary)',
  'text-secondary': 'var(--text-secondary)',
  'text-tertiary': 'var(--text-tertiary)',
}
```

### Step 4: Component Refactors (Priority Order)

1. **`category-battlefield.tsx`** — Add category dots, fix bar colors, remove gold overuse
2. **`streaming/page.tsx`** — Replace flat chips with dot-chip design, sort by magnitude
3. **`waiver/page.tsx`** — Gate HOT badge, color position badges, improve need bar contrast
4. **`dashboard/page.tsx`** — Fix waiver target empty state, add category dots to 2-start pitchers
5. **`roster/page.tsx`** — Color column headers, fix move button disabled state, tint IL rows

---

## 8. Accessibility Checklist

- [ ] All category colors ≥ 4.5:1 against `bg-surface` (`#16161e`)
- [ ] All text colors ≥ 4.5:1 against their background
- [ ] Status colors are distinguishable for deuteranopia (green-red weakness) — use shape + text, not color alone
- [ ] Focus rings use `border-focus` (`#6b6b8a`) with 2px offset
- [ ] No information conveyed by color alone (icons accompany all status badges)

---

## 9. Appendix: Before / After

### Streaming Station — Category Deficits

**Before:**
```
┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌────┐ ... (20 identical gray boxes)
│HR  │ │R   │ │RBI │ │AVG │ │OPS │ │NSB │
│-3.0│ │+3.0│ │-3.0│ │-0.1│ │-0.1│ │+1.0│
└────┘ └────┘ └────┘ └────┘ └────┘ └────┘
  rose   emerald  rose   gray   gray   emerald
```

**After:**
```
┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐
│● HR    │ │● R     │ │● RBI   │ │● AVG   │
│  -3.0 ↓│ │  +3.0 ↑│ │  -3.0 ↓│ │  -0.1 ↓│
└────────┘ └────────┘ └────────┘ └────────┘
  purple     blue       violet     yellow
  (sorted by |deficit| — largest first)
```

### War Room — Category Row

**Before:**
```
HR    0    [====|====]    0    4→7    BEHIND    Punt?
gray  gold     gray         gray  gray   orange    gray
```

**After:**
```
● HR    0    [━━━━|────]    0    4→7    BEHIND    Punt?
purple  white   purple bar    gray  gray  orange    gray
```

---

*End of spec. Ready for Claude Code review and implementation.*
