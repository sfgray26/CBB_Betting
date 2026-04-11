# Revolut-Inspired Design System Implementation Plan

> **Date:** April 10, 2026  
> **Author:** Kimi CLI (Deep Intelligence)  
> **Status:** READY FOR IMPLEMENTATION  
> **Estimated Duration:** 2-3 weeks (phased approach)

---

## Executive Summary

This document provides a comprehensive implementation plan for applying the Revolut-inspired design system to the CBB Edge / Fantasy Baseball platform. The design system emphasizes fintech confidence through bold typography, pill-shaped buttons, zero shadows, and a disciplined neutral palette.

### Key Transformation Areas

| Aspect | Current State | Target State (Revolut) |
|--------|---------------|------------------------|
| **Background** | Zinc-950 (`#09090b`) dark | Near-black/white binary (`#191c1f` / `#ffffff`) |
| **Buttons** | Rounded-lg with zinc colors | Full pills (9999px radius) with generous padding |
| **Typography** | Inter + JetBrains Mono | Aeonik Pro (display) + Inter (body) |
| **Depth** | Flat (current) | Zero shadows (matches current) ✓ |
| **Color System** | Signal colors (bet/consider/pass) | Semantic tokens (danger/warning/teal/blue) |

---

## Phase 1: Design Tokens & Configuration (Days 1-2)

### 1.1 Tailwind Configuration Updates

**File:** `frontend/tailwind.config.ts`

```typescript
import type { Config } from 'tailwindcss'
import defaultTheme from 'tailwindcss/defaultTheme'

const config: Config = {
  darkMode: 'class',
  content: [
    './app/**/*.{ts,tsx}',
    './components/**/*.{ts,tsx}',
  ],
  theme: {
    extend: {
      // Typography
      fontFamily: {
        display: ['Aeonik Pro', ...defaultTheme.fontFamily.sans],
        sans: ['Inter', ...defaultTheme.fontFamily.sans],
        mono: ['var(--font-mono)', ...defaultTheme.fontFamily.mono],
      },
      
      // Revolut Color System
      colors: {
        // Primary Surfaces
        'revolut-dark': '#191c1f',
        'revolut-white': '#ffffff',
        'revolut-surface': '#f4f4f4',
        
        // Brand/Interactive
        'revolut-blue': '#494fdf',
        'revolut-action-blue': '#4f55f1',
        'revolut-link-blue': '#376cd5',
        
        // Semantic Colors
        'revolut-danger': '#e23b4a',
        'revolut-deep-pink': '#e61e49',
        'revolut-warning': '#ec7e00',
        'revolut-yellow': '#b09000',
        'revolut-teal': '#00a87e',
        'revolut-light-green': '#428619',
        'revolut-green-text': '#006400',
        'revolut-light-blue': '#007bc2',
        'revolut-brown': '#936d62',
        'revolut-red-text': '#8b0000',
        
        // Neutral Scale
        'revolut-slate': '#505a63',
        'revolut-gray': '#8d969e',
        'revolut-gray-tone': '#c9c9cd',
        
        // Legacy signal colors (map to semantic)
        'signal-bet': '#ec7e00',      // warning → bet signal
        'signal-consider': '#494fdf', // blue → consider signal  
        'signal-pass': '#8d969e',     // gray → pass signal
        'signal-win': '#00a87e',      // teal → win
        'signal-loss': '#e23b4a',     // danger → loss
      },
      
      // Typography Scale (Revolut)
      fontSize: {
        'display-mega': ['8.5rem', { lineHeight: '1.00', letterSpacing: '-0.17em' }],   // 136px
        'display-hero': ['5rem', { lineHeight: '1.00', letterSpacing: '-0.05em' }],      // 80px
        'display-section': ['3rem', { lineHeight: '1.21', letterSpacing: '-0.03em' }],   // 48px
        'display-sub': ['2.5rem', { lineHeight: '1.20', letterSpacing: '-0.025em' }],    // 40px
        'display-card': ['2rem', { lineHeight: '1.19', letterSpacing: '-0.02em' }],      // 32px
        'display-feature': ['1.5rem', { lineHeight: '1.33' }],                           // 24px
        'display-nav': ['1.25rem', { lineHeight: '1.40' }],                              // 20px
        'body-large': ['1.125rem', { lineHeight: '1.56', letterSpacing: '-0.005em' }],   // 18px
        'body': ['1rem', { lineHeight: '1.50', letterSpacing: '0.015em' }],              // 16px
        'body-semibold': ['1rem', { lineHeight: '1.50', letterSpacing: '0.01em' }],      // 16px
      },
      
      // Border Radius (Revolut)
      borderRadius: {
        'pill': '9999px',
        'card': '20px',
        'standard': '12px',
      },
      
      // Spacing (8px base system)
      spacing: {
        '4.5': '1.125rem',  // 18px
        '5.5': '1.375rem',  // 22px
        '7': '1.75rem',     // 28px
        '14': '3.5rem',     // 56px
        '22': '5.5rem',     // 88px
        '30': '7.5rem',     // 120px
      },
      
      // Shadows (Zero shadows - flat design)
      boxShadow: {
        'focus': '0 0 0 0.125rem rgba(73, 79, 223, 0.5)',
      },
    },
  },
  plugins: [],
}

export default config
```

### 1.2 Global CSS Updates

**File:** `frontend/app/globals.css`

```css
@tailwind base;
@tailwind components;
@tailwind utilities;

/* Aeonik Pro - Display Font */
@font-face {
  font-family: 'Aeonik Pro';
  src: url('/fonts/AeonikPro-Regular.woff2') format('woff2');
  font-weight: 400;
  font-display: swap;
}

@font-face {
  font-family: 'Aeonik Pro';
  src: url('/fonts/AeonikPro-Medium.woff2') format('woff2');
  font-weight: 500;
  font-display: swap;
}

/* Inter - Body Font */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
  /* Revolut Color Tokens */
  --revolut-dark: #191c1f;
  --revolut-white: #ffffff;
  --revolut-surface: #f4f4f4;
  --revolut-blue: #494fdf;
  --revolut-danger: #e23b4a;
  --revolut-warning: #ec7e00;
  --revolut-teal: #00a87e;
  --revolut-slate: #505a63;
  --revolut-gray: #8d969e;
}

/* Base Styles */
html,
body {
  height: 100%;
}

body {
  background-color: var(--revolut-dark);
  color: var(--revolut-white);
  font-family: 'Inter', ui-sans-serif, system-ui, sans-serif;
}

/* Typography Utilities */
.font-display {
  font-family: 'Aeonik Pro', ui-sans-serif, system-ui, sans-serif;
}

/* Display Typography (Aeonik Pro Weight 500) */
.text-display-mega {
  font-family: 'Aeonik Pro', sans-serif;
  font-size: 8.5rem;
  font-weight: 500;
  line-height: 1.00;
  letter-spacing: -0.17em;
}

.text-display-hero {
  font-family: 'Aeonik Pro', sans-serif;
  font-size: 5rem;
  font-weight: 500;
  line-height: 1.00;
  letter-spacing: -0.05em;
}

.text-display-section {
  font-family: 'Aeonik Pro', sans-serif;
  font-size: 3rem;
  font-weight: 500;
  line-height: 1.21;
  letter-spacing: -0.03em;
}

/* Body Typography (Inter with positive tracking) */
.text-body-inter {
  font-family: 'Inter', sans-serif;
  letter-spacing: 0.015em;
}

/* Scrollbar Styling (Neutral, minimal) */
::-webkit-scrollbar {
  width: 6px;
  height: 6px;
}

::-webkit-scrollbar-track {
  background: var(--revolut-dark);
}

::-webkit-scrollbar-thumb {
  background: var(--revolut-slate);
  border-radius: 3px;
}

::-webkit-scrollbar-thumb:hover {
  background: var(--revolut-gray);
}

/* Focus States (Revolut Style) */
.focus-ring:focus-visible {
  outline: none;
  box-shadow: 0 0 0 0.125rem rgba(73, 79, 223, 0.5);
}

/* Utility Classes */
@layer utilities {
  .pill {
    border-radius: 9999px;
  }
  
  .no-shadow {
    box-shadow: none;
  }
}
```

### 1.3 Font Loading Strategy

**Option A: Self-Host Aeonik Pro (Recommended)**
- Purchase/download Aeonik Pro font files
- Place in `frontend/public/fonts/`
- Load via @font-face (as shown above)

**Option B: Use Fallback System**
- If Aeonik Pro unavailable, use Inter as display fallback
- Apply tight letter-spacing to mimic Aeonik Pro

```css
/* Fallback Display Font */
.font-display-fallback {
  font-family: 'Inter', sans-serif;
  font-weight: 600;
  letter-spacing: -0.05em;
}
```

---

## Phase 2: Core Component Migration (Days 3-5)

### 2.1 Button Component Migration

**Current:** `frontend/components/ui/button.tsx`
- Rounded-lg corners
- Zinc color scheme
- Small padding (px-3/4)

**Target:** Revolut Pill Buttons
- 9999px radius (full pill)
- Near-black (`#191c1f`) / white (`#ffffff`) / light surface (`#f4f4f4`)
- Generous padding (14px 32px)

**New Implementation:**

```tsx
// frontend/components/ui/button.tsx
import { cn } from '@/lib/utils'
import { cva, type VariantProps } from 'class-variance-authority'
import { Slot } from '@radix-ui/react-slot'
import * as React from 'react'

const buttonVariants = cva(
  'inline-flex items-center justify-center gap-2 font-display font-medium transition-all duration-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-revolut-blue/50 disabled:pointer-events-none disabled:opacity-50',
  {
    variants: {
      variant: {
        // Primary: Dark pill (main CTA)
        primary: 
          'bg-revolut-dark text-revolut-white pill px-8 py-3.5 text-xl hover:opacity-85',
        
        // Secondary: Light pill
        secondary: 
          'bg-revolut-surface text-revolut-dark pill px-8 py-3.5 text-xl hover:opacity-85',
        
        // Outlined: Transparent with border
        outlined: 
          'bg-transparent text-revolut-dark pill px-8 py-3.5 text-xl border-2 border-revolut-dark hover:bg-revolut-surface/10',
        
        // Ghost on Dark: For dark backgrounds
        ghost: 
          'bg-white/10 text-revolut-surface pill px-8 py-3.5 text-xl border-2 border-revolut-surface hover:bg-white/20',
        
        // Semantic: Danger/Warning/Success
        danger: 
          'bg-revolut-danger text-revolut-white pill px-8 py-3.5 text-xl hover:opacity-85',
        success: 
          'bg-revolut-teal text-revolut-white pill px-8 py-3.5 text-xl hover:opacity-85',
        warning: 
          'bg-revolut-warning text-revolut-white pill px-8 py-3.5 text-xl hover:opacity-85',
        
        // Legacy support (mapped to new system)
        default: 
          'bg-revolut-dark text-revolut-white pill px-8 py-3.5 text-xl hover:opacity-85',
      },
      size: {
        sm: 'text-sm px-6 py-2.5',
        md: 'text-base px-8 py-3.5',
        lg: 'text-xl px-10 py-4',
        icon: 'p-3',
      },
    },
    defaultVariants: { variant: 'primary', size: 'md' },
  }
)

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {
  asChild?: boolean
}

export const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, asChild = false, ...props }, ref) => {
    const Comp = asChild ? Slot : 'button'
    return (
      <Comp 
        className={cn(buttonVariants({ variant, size, className }))} 
        ref={ref} 
        {...props} 
      />
    )
  }
)
Button.displayName = 'Button'
```

### 2.2 Card Component Migration

**Current:** Rounded-lg with zinc colors
**Target:** 20px radius, flat (no shadows), dark/light variants

```tsx
// frontend/components/ui/card.tsx
import { cn } from '@/lib/utils'

// Card Container - 20px radius, no shadows
export function Card({ 
  className, 
  children, 
  variant = 'dark',
  ...props 
}: React.HTMLAttributes<HTMLDivElement> & { variant?: 'dark' | 'light' }) {
  return (
    <div
      className={cn(
        'rounded-card border-0', // No shadows, 20px radius
        variant === 'dark' && 'bg-revolut-dark border border-revolut-slate/20',
        variant === 'light' && 'bg-revolut-surface',
        className
      )}
      {...props}
    >
      {children}
    </div>
  )
}

// Card Header
export function CardHeader({
  className,
  children,
  ...props
}: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div className={cn('p-6 pb-4', className)} {...props}>
      {children}
    </div>
  )
}

// Card Title (Aeonik Pro, weight 500)
export function CardTitle({
  className,
  children,
  ...props
}: React.HTMLAttributes<HTMLHeadingElement>) {
  return (
    <h3
      className={cn(
        'font-display text-xl font-medium text-revolut-white',
        className
      )}
      {...props}
    >
      {children}
    </h3>
  )
}

// Card Value (Large, mono for numbers)
export function CardValue({
  className,
  children,
  positive,
  negative,
  ...props
}: React.HTMLAttributes<HTMLDivElement> & { positive?: boolean; negative?: boolean }) {
  return (
    <div
      className={cn(
        'text-3xl font-mono tabular-nums font-semibold',
        positive && 'text-revolut-teal',
        negative && 'text-revolut-danger',
        !positive && !negative && 'text-revolut-white',
        className
      )}
      {...props}
    >
      {children}
    </div>
  )
}

// Card Content
export function CardContent({
  className,
  children,
  ...props
}: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div className={cn('p-6 pt-0', className)} {...props}>
      {children}
    </div>
  )
}

// Card Description
export function CardDescription({
  className,
  children,
  ...props
}: React.HTMLAttributes<HTMLParagraphElement>) {
  return (
    <p className={cn('text-sm text-revolut-gray', className)} {...props}>
      {children}
    </p>
  )
}
```

### 2.3 Alert/Badge Components

```tsx
// frontend/components/ui/alert.tsx
import { cn } from '@/lib/utils'

interface AlertProps extends React.HTMLAttributes<HTMLDivElement> {
  variant?: 'danger' | 'warning' | 'success' | 'info'
}

export function Alert({ className, variant = 'info', children, ...props }: AlertProps) {
  return (
    <div
      className={cn(
        'rounded-standard p-4 border-l-4',
        variant === 'danger' && 'bg-revolut-danger/10 border-revolut-danger text-revolut-danger',
        variant === 'warning' && 'bg-revolut-warning/10 border-revolut-warning text-revolut-warning',
        variant === 'success' && 'bg-revolut-teal/10 border-revolut-teal text-revolut-teal',
        variant === 'info' && 'bg-revolut-blue/10 border-revolut-blue text-revolut-blue',
        className
      )}
      {...props}
    >
      {children}
    </div>
  )
}
```

```tsx
// frontend/components/ui/badge.tsx
import { cn } from '@/lib/utils'

interface BadgeProps extends React.HTMLAttributes<HTMLSpanElement> {
  variant?: 'danger' | 'warning' | 'success' | 'info' | 'neutral'
}

export function Badge({ className, variant = 'neutral', children, ...props }: BadgeProps) {
  return (
    <span
      className={cn(
        'inline-flex items-center px-3 py-1 text-sm font-medium rounded-pill',
        variant === 'danger' && 'bg-revolut-danger text-revolut-white',
        variant === 'warning' && 'bg-revolut-warning text-revolut-white',
        variant === 'success' && 'bg-revolut-teal text-revolut-white',
        variant === 'info' && 'bg-revolut-blue text-revolut-white',
        variant === 'neutral' && 'bg-revolut-surface text-revolut-dark',
        className
      )}
      {...props}
    >
      {children}
    </span>
  )
}
```

---

## Phase 3: Page Layout Migration (Days 6-10)

### 3.1 Layout Structure

**Current:** Dark theme throughout, zinc colors
**Target:** Dark/light section alternation (Revolut style)

```tsx
// frontend/app/(dashboard)/layout.tsx
export default function DashboardLayout({ children }: { children: React.ReactNode }) {
  return (
    <div className="min-h-screen bg-revolut-dark">
      {/* Navigation - Dark */}
      <header className="sticky top-0 z-50 bg-revolut-dark border-b border-revolut-slate/20">
        {/* Nav content */}
      </header>
      
      {/* Main Content */}
      <main className="flex-1">
        {children}
      </main>
    </div>
  )
}
```

### 3.2 Hero Section (Dashboard Home)

```tsx
// Example hero section using Revolut style
<section className="relative bg-revolut-dark py-20 px-6 lg:px-12">
  <div className="max-w-7xl mx-auto">
    {/* Mega Display Typography */}
    <h1 className="font-display text-display-hero text-revolut-white mb-6">
      Fantasy Baseball
      <span className="text-revolut-blue">Edge</span>
    </h1>
    
    {/* Body Text */}
    <p className="text-body-large text-revolut-gray max-w-2xl mb-8">
      Data-driven insights for H2H One Win fantasy baseball. 
      Real-time analytics, scarcity tracking, and matchup optimization.
    </p>
    
    {/* Pill Button CTAs */}
    <div className="flex flex-wrap gap-4">
      <Button variant="primary" size="lg">
        View Dashboard
      </Button>
      <Button variant="outlined" size="lg">
        Explore Features
      </Button>
    </div>
  </div>
</section>

{/* Alternating Light Section */}
<section className="bg-revolut-surface py-20 px-6 lg:px-12">
  <div className="max-w-7xl mx-auto">
    <h2 className="font-display text-display-section text-revolut-dark mb-8">
      This Week&apos;s Matchup
    </h2>
    {/* Content */}
  </div>
</section>
```

### 3.3 Data Tables (Modern Revolut Style)

```tsx
// Modern table with Revolut styling
<div className="rounded-card bg-revolut-dark border border-revolut-slate/20 overflow-hidden">
  <table className="w-full">
    <thead className="bg-revolut-surface/50">
      <tr>
        <th className="px-6 py-4 text-left text-sm font-display font-medium text-revolut-slate">
          Player
        </th>
        <th className="px-6 py-4 text-right text-sm font-display font-medium text-revolut-slate">
          Position
        </th>
        <th className="px-6 py-4 text-right text-sm font-display font-medium text-revolut-slate">
          Scarcity
        </th>
      </tr>
    </thead>
    <tbody className="divide-y divide-revolut-slate/10">
      {players.map((player) => (
        <tr key={player.id} className="hover:bg-revolut-surface/5 transition-colors">
          <td className="px-6 py-4 font-medium text-revolut-white">
            {player.name}
          </td>
          <td className="px-6 py-4 text-right">
            <Badge variant="neutral">{player.position}</Badge>
          </td>
          <td className="px-6 py-4 text-right">
            <span className={cn(
              'font-mono font-semibold',
              player.scarcity > 80 ? 'text-revolut-danger' :
              player.scarcity > 50 ? 'text-revolut-warning' :
              'text-revolut-teal'
            )}>
              {player.scarcity}%
            </span>
          </td>
        </tr>
      ))}
    </tbody>
  </table>
</div>
```

---

## Phase 4: Feature-Specific Components (Days 11-14)

### 4.1 KPI Cards (Revolut Style)

```tsx
// High-impact KPI cards for dashboard
interface KPICardProps {
  title: string
  value: string | number
  change?: number
  changeLabel?: string
  variant?: 'default' | 'positive' | 'negative'
}

export function KPICard({ title, value, change, changeLabel, variant = 'default' }: KPICardProps) {
  return (
    <Card variant="dark" className="p-6">
      <CardTitle className="text-sm text-revolut-gray mb-2">{title}</CardTitle>
      <CardValue 
        positive={variant === 'positive'}
        negative={variant === 'negative'}
        className="text-4xl mb-2"
      >
        {value}
      </CardValue>
      {change !== undefined && (
        <div className="flex items-center gap-2">
          <Badge 
            variant={change > 0 ? 'success' : change < 0 ? 'danger' : 'neutral'}
          >
            {change > 0 ? '+' : ''}{change}%
          </Badge>
          {changeLabel && (
            <span className="text-sm text-revolut-gray">{changeLabel}</span>
          )}
        </div>
      )}
    </Card>
  )
}
```

### 4.2 Player Cards (Scarcity Visualization)

```tsx
// Player cards showing scarcity index
interface PlayerCardProps {
  player: {
    name: string
    team: string
    positions: string[]
    scarcityRank: number
    rosteredPct: number
    isMultiEligible: boolean
  }
}

export function PlayerCard({ player }: PlayerCardProps) {
  return (
    <Card variant="dark" className="p-6 hover:border-revolut-blue/30 transition-colors">
      <div className="flex items-start justify-between mb-4">
        <div>
          <h3 className="font-display text-xl font-medium text-revolut-white mb-1">
            {player.name}
          </h3>
          <p className="text-sm text-revolut-gray">{player.team}</p>
        </div>
        {player.isMultiEligible && (
          <Badge variant="info">Multi</Badge>
        )}
      </div>
      
      <div className="flex flex-wrap gap-2 mb-4">
        {player.positions.map((pos) => (
          <Badge key={pos} variant="neutral">{pos}</Badge>
        ))}
      </div>
      
      <div className="flex items-center justify-between pt-4 border-t border-revolut-slate/20">
        <div>
          <p className="text-xs text-revolut-gray uppercase tracking-wider mb-1">
            Scarcity
          </p>
          <p className={cn(
            'font-mono font-bold text-lg',
            player.scarcityRank <= 10 ? 'text-revolut-danger' :
            player.scarcityRank <= 25 ? 'text-revolut-warning' :
            'text-revolut-teal'
          )}>
            #{player.scarcityRank}
          </p>
        </div>
        <div className="text-right">
          <p className="text-xs text-revolut-gray uppercase tracking-wider mb-1">
            Rostered
          </p>
          <p className="font-mono font-bold text-lg text-revolut-white">
            {player.rosteredPct}%
          </p>
        </div>
      </div>
    </Card>
  )
}
```

### 4.3 Navigation (Revolut Style)

```tsx
// Clean, pill-based navigation
export function Navigation() {
  return (
    <nav className="sticky top-0 z-50 bg-revolut-dark border-b border-revolut-slate/20">
      <div className="max-w-7xl mx-auto px-6 lg:px-12">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <Link href="/" className="font-display text-2xl font-medium text-revolut-white">
            Fantasy<span className="text-revolut-blue">Edge</span>
          </Link>
          
          {/* Desktop Nav - Pills */}
          <div className="hidden md:flex items-center gap-2">
            {navItems.map((item) => (
              <Link
                key={item.href}
                href={item.href}
                className={cn(
                  'px-4 py-2 rounded-pill text-sm font-display font-medium transition-colors',
                  item.active 
                    ? 'bg-revolut-white text-revolut-dark' 
                    : 'text-revolut-gray hover:text-revolut-white hover:bg-revolut-surface/10'
                )}
              >
                {item.label}
              </Link>
            ))}
          </div>
          
          {/* CTA Button */}
          <Button variant="primary" size="sm">
            Sync Yahoo
          </Button>
        </div>
      </div>
    </nav>
  )
}
```

---

## Phase 5: Responsive Design (Days 15-17)

### 5.1 Mobile-First Breakpoints

```css
/* Tailwind config additions */
screens: {
  'xs': '400px',      /* Mobile Small */
  'sm': '720px',      /* Mobile → Tablet */
  'md': '1024px',     /* Tablet → Desktop */
  'lg': '1280px',     /* Desktop */
  'xl': '1920px',     /* Large */
}
```

### 5.2 Typography Scaling

```tsx
// Responsive typography component
export function ResponsiveHeading({ 
  children, 
  as: Component = 'h1',
  size = 'hero'
}: HeadingProps) {
  const sizeClasses = {
    mega: 'text-4xl sm:text-6xl md:text-7xl lg:text-display-hero xl:text-display-mega',
    hero: 'text-3xl sm:text-4xl md:text-5xl lg:text-display-hero',
    section: 'text-2xl sm:text-3xl md:text-display-section',
    card: 'text-xl sm:text-2xl md:text-display-card',
  }
  
  return (
    <Component className={cn(
      'font-display font-medium leading-tight tracking-tight',
      sizeClasses[size]
    )}>
      {children}
    </Component>
  )
}
```

---

## Phase 6: Animation & Interactions (Days 18-19)

### 6.1 Micro-interactions

```css
/* Subtle hover transitions (Revolut style) */
@layer utilities {
  .hover-lift {
    transition: transform 0.2s ease, opacity 0.2s ease;
  }
  .hover-lift:hover {
    transform: translateY(-2px);
  }
  
  .hover-opacity {
    transition: opacity 0.2s ease;
  }
  .hover-opacity:hover {
    opacity: 0.85;
  }
}
```

### 6.2 Page Transitions

```tsx
// Smooth page transitions
import { motion } from 'framer-motion'

export function PageTransition({ children }: { children: React.ReactNode }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -10 }}
      transition={{ duration: 0.3, ease: 'easeOut' }}
    >
      {children}
    </motion.div>
  )
}
```

---

## Phase 7: Quality Assurance (Days 20-21)

### 7.1 Visual Regression Checklist

- [ ] All buttons use pill shape (9999px radius)
- [ ] No shadows anywhere (flat design)
- [ ] Aeonik Pro weight 500 for all headings
- [ ] Inter with positive letter-spacing for body
- [ ] Color contrast meets WCAG AA standards
- [ ] Responsive at all breakpoints (400px → 1920px)
- [ ] Touch targets minimum 44px on mobile

### 7.2 Component Audit

| Component | Status | Notes |
|-----------|--------|-------|
| Button | ⏳ Pending | Migrate to pill shape |
| Card | ⏳ Pending | 20px radius, no shadows |
| Badge | ⏳ Pending | Pill shape, semantic colors |
| Alert | ⏳ Pending | Left border accent style |
| Navigation | ⏳ Pending | Pill nav items |
| Tables | ⏳ Pending | Clean, minimal borders |
| Forms | ⏳ Pending | Revolut input styling |

---

## Implementation Priority Matrix

### P0 (Critical - Week 1)
1. Tailwind configuration with Revolut tokens
2. Button component migration (pill shape)
3. Global CSS updates (fonts, colors)

### P1 (High - Week 2)
4. Card component migration
5. Navigation redesign
6. Dashboard hero section

### P2 (Medium - Week 3)
7. Data tables redesign
8. KPI cards implementation
9. Player scarcity cards
10. Responsive testing

### P3 (Low - Future)
11. Advanced animations
12. Dark/light mode toggle
13. Additional page templates

---

## Design Token Quick Reference

### Colors
```
Primary Dark:     #191c1f  (revolut-dark)
Primary Light:    #ffffff  (revolut-white)
Surface:          #f4f4f4  (revolut-surface)
Brand Blue:       #494fdf  (revolut-blue)
Danger:           #e23b4a  (revolut-danger)
Warning:          #ec7e00  (revolut-warning)
Success:          #00a87e  (revolut-teal)
```

### Typography
```
Display:   Aeonik Pro, weight 500, tight tracking
Body:      Inter, weight 400, positive tracking
Mono:      JetBrains Mono (for numbers)
```

### Border Radius
```
Pill:      9999px (buttons)
Card:      20px (containers)
Standard:  12px (inputs, small elements)
```

### Spacing
```
Base:      8px
Scale:     4, 6, 8, 14, 16, 20, 24, 32, 40, 48, 80, 120px
```

---

## Migration Strategy

### Approach 1: Gradual Migration (Recommended)
- Keep existing components working
- Create new `ui-revolut/` component folder
- Migrate pages one at a time
- Remove old components after full migration

### Approach 2: Big Bang
- Update all components simultaneously
- Higher risk but faster completion
- Requires thorough testing

### Recommended: Approach 1
```
frontend/components/
  ui/              # Existing components (keep during migration)
  ui-revolut/      # New Revolut-style components
  layout/          # Layout components (update gradually)
```

---

## Success Metrics

- **Visual Consistency:** 100% of components follow Revolut design language
- **Performance:** No regression in Lighthouse scores
- **Accessibility:** WCAG AA compliance on all interactive elements
- **Responsive:** Perfect rendering at all breakpoints
- **Developer Experience:** Clear component API, comprehensive Storybook stories

---

## Resources

### Fonts
- **Aeonik Pro:** Purchase from [CoType Foundry](https://co-type.com)
- **Inter:** Google Fonts (free)
- **JetBrains Mono:** Google Fonts (free)

### Reference
- Revolut Marketing Site: https://www.revolut.com
- Design System Document: `DESIGN.md` (root directory)

---

*End of Implementation Plan - Ready for Execution*
