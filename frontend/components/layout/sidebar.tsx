'use client'

import Link from 'next/link'
import { usePathname } from 'next/navigation'
import {
  BarChart2,
  TrendingUp,
  ClipboardList,
  Target,
  Bell,
  Zap,
  Activity,
  Radio,
  Trophy,
  Dumbbell,
  ShieldAlert,
  ListChecks,
  ArrowLeftRight,
  Users,
  Swords,
} from 'lucide-react'
import { useQuery } from '@tanstack/react-query'
import { endpoints } from '@/lib/api'
import { cn } from '@/lib/utils'

const navSections = [
  {
    label: 'Analytics',
    items: [
      { href: '/performance', label: 'Performance', icon: BarChart2 },
      { href: '/clv', label: 'CLV Analysis', icon: TrendingUp },
      { href: '/bet-history', label: 'Bet History', icon: ClipboardList },
      { href: '/calibration', label: 'Calibration', icon: Target },
      { href: '/alerts', label: 'Alerts', icon: Bell },
    ],
    soon: false,
  },
  {
    label: 'Trading',
    items: [
      { href: '/today', label: "Today's Bets", icon: Zap },
      { href: '/live-slate', label: 'Live Slate', icon: Activity },
      { href: '/odds-monitor', label: 'Odds Monitor', icon: Radio },
    ],
    soon: false,
  },
  {
    label: 'Tournament',
    items: [
      { href: '/bracket', label: 'Bracket Simulator', icon: Trophy },
    ],
    soon: false,
  },
  {
    label: 'Fantasy Baseball',
    items: [
      { href: '/fantasy', label: 'Draft Board', icon: Dumbbell },
      { href: '/fantasy/lineup', label: 'Daily Lineup', icon: ListChecks },
      { href: '/fantasy/waiver', label: 'Waiver Wire', icon: ArrowLeftRight },
      { href: '/fantasy/roster', label: 'My Roster', icon: Users },
      { href: '/fantasy/matchup', label: 'Matchup', icon: Swords },
    ],
    soon: false,
  },
  {
    label: 'Admin',
    items: [
      { href: '/admin', label: 'Risk Dashboard', icon: ShieldAlert },
    ],
    soon: false,
  },
]

interface SidebarProps {
  isOpen?: boolean
  onClose?: () => void
}

export default function Sidebar({ isOpen = false, onClose }: SidebarProps) {
  const pathname = usePathname()

  const { data: portfolio } = useQuery({
    queryKey: ['portfolio'],
    queryFn: endpoints.portfolioStatus,
    refetchInterval: 60_000,
  })

  const drawdown = portfolio?.drawdown_pct ?? 0
  const dotColor =
    drawdown < 5
      ? 'bg-emerald-400'
      : drawdown < 10
        ? 'bg-amber-400'
        : 'bg-rose-500'

  return (
    <aside
      className={cn(
        // Base: fixed sidebar, transitions for mobile drawer
        'fixed left-0 top-0 h-full w-60 bg-zinc-900 border-r border-zinc-800 flex flex-col z-30',
        'transition-transform duration-200 ease-in-out',
        // Desktop: always visible
        'md:translate-x-0',
        // Mobile: slide in when open, slide out when closed
        isOpen ? 'translate-x-0' : '-translate-x-full md:translate-x-0',
      )}
    >
      {/* Logo */}
      <div className="px-5 py-5 border-b border-zinc-800">
        <div className="font-bold text-lg text-amber-400 tracking-tight">CBB EDGE</div>
        <div className="text-xs text-zinc-500 mt-0.5">Analytics</div>
      </div>

      {/* Nav */}
      <nav className="flex-1 overflow-y-auto py-4 px-3 space-y-6">
        {navSections.map((section) => (
          <div key={section.label}>
            <p className="px-2 mb-2 text-xs font-semibold text-zinc-600 uppercase tracking-wider">
              {section.label}
            </p>
            <ul className="space-y-0.5">
              {section.items.map((item) => {
                const Icon = item.icon
                const isActive = pathname === item.href

                if (section.soon) {
                  return (
                    <li key={item.href}>
                      <span className="flex items-center gap-3 px-3 py-2.5 rounded-md text-zinc-600 cursor-not-allowed min-h-[44px]">
                        <Icon className="h-4 w-4 flex-shrink-0" />
                        <span className="flex-1 text-sm">{item.label}</span>
                        <span className="text-xs bg-zinc-700/50 text-zinc-500 px-1.5 py-0.5 rounded-full border border-zinc-700">
                          Soon
                        </span>
                      </span>
                    </li>
                  )
                }

                return (
                  <li key={item.href}>
                    <Link
                      href={item.href}
                      onClick={onClose}
                      className={cn(
                        'flex items-center gap-3 px-3 py-2.5 rounded-md text-sm transition-colors min-h-[44px]',
                        isActive
                          ? 'text-amber-400 bg-amber-400/10 border-l-2 border-amber-400 -ml-px pl-[11px]'
                          : 'text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800',
                      )}
                    >
                      <Icon className="h-4 w-4 flex-shrink-0" />
                      <span>{item.label}</span>
                    </Link>
                  </li>
                )
              })}
            </ul>
          </div>
        ))}
      </nav>

      {/* Bottom panel */}
      <div className="border-t border-zinc-800 px-4 py-3 space-y-2">
        {/* Portfolio chip */}
        <div className="flex items-center gap-2 px-2 py-2 bg-zinc-800/50 rounded-md">
          <span className={cn('h-2 w-2 rounded-full flex-shrink-0', dotColor)} />
          <div className="flex-1 min-w-0">
            <div className="text-xs text-zinc-400 leading-none">Portfolio</div>
            <div className="text-xs font-mono text-zinc-300 mt-0.5 tabular-nums">
              {portfolio
                ? `DD: ${drawdown.toFixed(1)}% | Exp: ${portfolio.total_exposure_pct.toFixed(1)}%`
                : 'Loading...'}
            </div>
          </div>
        </div>

        {/* Streamlit link */}
        <a
          href="http://localhost:8501"
          target="_blank"
          rel="noopener noreferrer"
          className="flex items-center gap-2 px-2 py-1.5 text-xs text-zinc-500 hover:text-zinc-300 transition-colors rounded"
        >
          <span>Streamlit Dashboard</span>
          <span className="text-zinc-600">&#8599;</span>
        </a>
      </div>
    </aside>
  )
}
