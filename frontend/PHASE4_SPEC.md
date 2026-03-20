# PHASE 4 SPEC: Mobile & PWA Implementation

**Produced by:** Kimi CLI (Deep Intelligence)  
**Date:** March 19, 2026  
**For:** Claude Code (Master Architect)  
**Scope:** Next.js Frontend Mobile Responsiveness + PWA

---

## EXECUTIVE SUMMARY

Current frontend is desktop-only with fixed `w-60` sidebar. Phase 4 makes it fully responsive with mobile drawer navigation, touch-friendly targets, and PWA installability.

**Critical Issues Found:**
1. ❌ Missing viewport meta tag
2. ❌ Fixed sidebar blocks content on mobile
3. ❌ Touch targets too small (< 44px)
4. ❌ Tables overflow without horizontal scroll
5. ❌ No PWA manifest

---

## ISSUE 1: Viewport Meta Tag (CRITICAL)

**File:** `frontend/app/layout.tsx`

**Current:**
```tsx
export const metadata: Metadata = {
  title: 'CBB Edge',
  description: 'College basketball betting analytics',
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark">
      <body className={`${inter.variable} ${jetbrains.variable} font-sans bg-zinc-950 text-zinc-50 antialiased`}>
        <Providers>{children}</Providers>
      </body>
    </html>
  )
}
```

**Fix:**
```tsx
export const metadata: Metadata = {
  title: 'CBB Edge',
  description: 'College basketball betting analytics',
  viewport: 'width=device-width, initial-scale=1, maximum-scale=5',
  themeColor: '#09090b',
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark">
      <head>
        <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=5" />
        <meta name="theme-color" content="#09090b" />
        <link rel="manifest" href="/manifest.json" />
        <link rel="apple-touch-icon" href="/icon-192.png" />
      </head>
      <body className={`${inter.variable} ${jetbrains.variable} font-sans bg-zinc-950 text-zinc-50 antialiased`}>
        <Providers>{children}</Providers>
      </body>
    </html>
  )
}
```

---

## ISSUE 2: Mobile Sidebar Drawer

**Files:** 
- `frontend/components/layout/sidebar.tsx`
- `frontend/components/layout/header.tsx` (needs mobile menu button)
- `frontend/app/(dashboard)/layout.tsx`

### 2a. Create MobileSidebar Component

**New File:** `frontend/components/layout/mobile-sidebar.tsx`

```tsx
'use client'

import { useState } from 'react'
import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { X, Menu } from 'lucide-react'
import { cn } from '@/lib/utils'
import { navSections } from './sidebar'

export function MobileMenuButton({ onClick }: { onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      className="lg:hidden p-2 rounded-md text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800 min-h-[44px] min-w-[44px] flex items-center justify-center"
      aria-label="Open menu"
    >
      <Menu className="h-5 w-5" />
    </button>
  )
}

export function MobileSidebar({ isOpen, onClose }: { isOpen: boolean; onClose: () => void }) {
  const pathname = usePathname()

  if (!isOpen) return null

  return (
    <>
      {/* Backdrop */}
      <div
        className="fixed inset-0 bg-black/50 z-40 lg:hidden"
        onClick={onClose}
        aria-hidden="true"
      />
      
      {/* Drawer */}
      <aside className="fixed left-0 top-0 h-full w-72 bg-zinc-900 border-r border-zinc-800 z-50 lg:hidden flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-4 border-b border-zinc-800">
          <div className="font-bold text-lg text-amber-400 tracking-tight">CBB EDGE</div>
          <button
            onClick={onClose}
            className="p-2 rounded-md text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800 min-h-[44px] min-w-[44px] flex items-center justify-center"
            aria-label="Close menu"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        {/* Nav */}
        <nav className="flex-1 overflow-y-auto py-4 px-3 space-y-6">
          {navSections.map((section) => (
            <div key={section.label}>
              <p className="px-3 mb-2 text-xs font-semibold text-zinc-600 uppercase tracking-wider">
                {section.label}
              </p>
              <ul className="space-y-1">
                {section.items.map((item) => {
                  const Icon = item.icon
                  const isActive = pathname === item.href

                  return (
                    <li key={item.href}>
                      <Link
                        href={item.href}
                        onClick={onClose}
                        className={cn(
                          'flex items-center gap-3 px-3 py-3 rounded-md text-sm min-h-[44px] transition-colors',
                          isActive
                            ? 'text-amber-400 bg-amber-400/10'
                            : 'text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800',
                        )}
                      >
                        <Icon className="h-5 w-5 flex-shrink-0" />
                        <span>{item.label}</span>
                      </Link>
                    </li>
                  )
                })}
              </ul>
            </div>
          ))}
        </nav>
      </aside>
    </>
  )
}
```

### 2b. Export navSections from Sidebar

**File:** `frontend/components/layout/sidebar.tsx`

**Change:** Export the navSections constant

```tsx
// At top of file, change from const to export:
export const navSections = [
  // ... existing content
]
```

### 2c. Update Header with Mobile Menu Button

**File:** `frontend/components/layout/header.tsx`

**Current:** (assumed structure)
```tsx
export default function Header() {
  return (
    <header className="h-14 border-b border-zinc-800 flex items-center px-6 bg-zinc-950">
      {/* content */}
    </header>
  )
}
```

**Fix:**
```tsx
'use client'

import { MobileMenuButton } from './mobile-sidebar'

interface HeaderProps {
  onMenuClick: () => void
}

export default function Header({ onMenuClick }: HeaderProps) {
  return (
    <header className="h-14 border-b border-zinc-800 flex items-center px-4 lg:px-6 bg-zinc-950 gap-4">
      <MobileMenuButton onClick={onMenuClick} />
      
      {/* Logo for mobile (hidden on desktop) */}
      <div className="lg:hidden font-bold text-amber-400 tracking-tight">
        CBB EDGE
      </div>
      
      {/* Existing header content */}
      <div className="flex-1" />
      
      {/* User menu, etc */}
    </header>
  )
}
```

### 2d. Update Dashboard Layout

**File:** `frontend/app/(dashboard)/layout.tsx`

**Current:**
```tsx
import Sidebar from '@/components/layout/sidebar'
import Header from '@/components/layout/header'

export default function DashboardLayout({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex h-screen bg-zinc-950 overflow-hidden">
      <Sidebar />
      <div className="flex flex-col flex-1 ml-60 min-w-0">
        <Header />
        <main className="flex-1 overflow-y-auto p-6">
          {children}
        </main>
      </div>
    </div>
  )
}
```

**Fix:**
```tsx
'use client'

import { useState } from 'react'
import Sidebar from '@/components/layout/sidebar'
import Header from '@/components/layout/header'
import { MobileSidebar } from '@/components/layout/mobile-sidebar'

export default function DashboardLayout({ children }: { children: React.ReactNode }) {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)

  return (
    <div className="flex h-screen bg-zinc-950 overflow-hidden">
      {/* Desktop Sidebar - hidden on mobile */}
      <div className="hidden lg:block">
        <Sidebar />
      </div>
      
      {/* Mobile Sidebar Drawer */}
      <MobileSidebar 
        isOpen={mobileMenuOpen} 
        onClose={() => setMobileMenuOpen(false)} 
      />
      
      {/* Main content */}
      <div className="flex flex-col flex-1 lg:ml-60 min-w-0">
        <Header onMenuClick={() => setMobileMenuOpen(true)} />
        <main className="flex-1 overflow-y-auto p-4 lg:p-6">
          {children}
        </main>
      </div>
    </div>
  )
}
```

---

## ISSUE 3: Touch Target Sizes

**Requirement:** All interactive elements must be at least 44×44px

**Files to update:**

### Sidebar Links (Already OK)
Current: `py-2` → Need: `min-h-[44px]`

**File:** `frontend/components/layout/sidebar.tsx`

```tsx
// Change nav link class:
className={cn(
  'flex items-center gap-3 px-3 py-2 rounded-md text-sm transition-colors min-h-[44px]',
  // ... rest
)}
```

### Header Elements

**File:** `frontend/components/layout/header.tsx`

Ensure all buttons have `min-h-[44px] min-w-[44px]`

### Filter Tabs / Buttons

**Files:** Any page with filter buttons

**Example from live-slate:**
```tsx
// Ensure filter buttons are touch-friendly
<button className="px-3 py-1.5 rounded-md text-xs font-medium min-h-[44px]">
  {label}
</button>
```

### DataTable Row Heights

**File:** `frontend/components/ui/data-table.tsx`

```tsx
// Ensure table rows are tall enough for touch
tr className="min-h-[44px]"
```

---

## ISSUE 4: Responsive Grid Layouts

### KPI Cards Grid

**Files:** All pages using KpiCard grids

**Current:**
```tsx
<div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
```

**Fix:**
```tsx
<div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
```

### Two-Column Layouts

**Example from today page:**
```tsx
// Current
<div className="grid grid-cols-2 gap-4">

// Fix
<div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
```

### Tables Overflow

**File:** `frontend/components/ui/data-table.tsx`

**Wrap in scrollable container:**
```tsx
<div className="overflow-x-auto">
  <table className="w-full min-w-[600px]">
    {/* table content */}
  </table>
</div>
```

---

## ISSUE 5: PWA Manifest

**New File:** `frontend/public/manifest.json`

```json
{
  "name": "CBB Edge Analytics",
  "short_name": "CBBEdge",
  "description": "College basketball betting analytics platform",
  "start_url": "/",
  "display": "standalone",
  "background_color": "#09090b",
  "theme_color": "#09090b",
  "orientation": "portrait-primary",
  "scope": "/",
  "icons": [
    {
      "src": "/icon-72.png",
      "sizes": "72x72",
      "type": "image/png",
      "purpose": "maskable any"
    },
    {
      "src": "/icon-96.png",
      "sizes": "96x96",
      "type": "image/png",
      "purpose": "maskable any"
    },
    {
      "src": "/icon-128.png",
      "sizes": "128x128",
      "type": "image/png",
      "purpose": "maskable any"
    },
    {
      "src": "/icon-144.png",
      "sizes": "144x144",
      "type": "image/png",
      "purpose": "maskable any"
    },
    {
      "src": "/icon-152.png",
      "sizes": "152x152",
      "type": "image/png",
      "purpose": "maskable any"
    },
    {
      "src": "/icon-192.png",
      "sizes": "192x192",
      "type": "image/png",
      "purpose": "maskable any"
    },
    {
      "src": "/icon-384.png",
      "sizes": "384x384",
      "type": "image/png",
      "purpose": "maskable any"
    },
    {
      "src": "/icon-512.png",
      "sizes": "512x512",
      "type": "image/png",
      "purpose": "maskable any"
    }
  ]
}
```

### Icons Required

Create these icons (or use placeholder):
- `/frontend/public/icon-72.png`
- `/frontend/public/icon-96.png`
- `/frontend/public/icon-128.png`
- `/frontend/public/icon-144.png`
- `/frontend/public/icon-152.png`
- `/frontend/public/icon-192.png`
- `/frontend/public/icon-384.png`
- `/frontend/public/icon-512.png`

**Simple placeholder generator:**
```bash
cd frontend/public
# Create simple amber square icons (replace with actual logo later)
for size in 72 96 128 144 152 192 384 512; do
  convert -size ${size}x${size} xc:'#f59e0b' icon-${size}.png
done
```

---

## IMPLEMENTATION CHECKLIST

### Step 1: Viewport & PWA Setup
- [ ] Update `frontend/app/layout.tsx` with viewport meta
- [ ] Create `frontend/public/manifest.json`
- [ ] Generate PWA icons (8 sizes)

### Step 2: Mobile Navigation
- [ ] Create `frontend/components/layout/mobile-sidebar.tsx`
- [ ] Export `navSections` from `sidebar.tsx`
- [ ] Update `header.tsx` with menu button
- [ ] Update `(dashboard)/layout.tsx` with mobile state

### Step 3: Touch Targets
- [ ] Audit all buttons for `min-h-[44px] min-w-[44px]`
- [ ] Update sidebar links
- [ ] Update filter buttons
- [ ] Update DataTable row heights

### Step 4: Responsive Layouts
- [ ] Update all `grid-cols-2` to `grid-cols-1 sm:grid-cols-2`
- [ ] Add `overflow-x-auto` to tables
- [ ] Test on mobile viewport (375px width)

### Step 5: Testing
- [ ] Chrome DevTools mobile emulation
- [ ] iPhone Safari testing
- [ ] Android Chrome testing
- [ ] PWA install prompt appears
- [ ] Offline mode works (if service worker added)

---

## TESTING COMMANDS

```bash
cd frontend

# Build check
npm run build

# TypeScript check
npx tsc --noEmit

# Lighthouse PWA audit (requires build)
npm install -g lighthouse
lighthouse http://localhost:3000 --preset=desktop
```

---

## EXPECTED RESULTS

### Desktop (> 1024px)
- Sidebar visible on left (w-60)
- Full navigation visible
- Grid layouts at full columns

### Tablet (768px - 1024px)
- Sidebar visible but narrower (optional)
- Grid layouts at 2 columns
- Touch targets maintained

### Mobile (< 768px)
- Sidebar hidden, hamburger menu
- Drawer slides in from left
- Single column layouts
- All touch targets ≥ 44px
- Horizontal scroll on tables
- PWA installable

---

**Spec produced by:** Kimi CLI  
**Date:** March 19, 2026  
**Status:** Ready for implementation
