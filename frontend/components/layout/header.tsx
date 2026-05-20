'use client'

import { useEffect, useState } from 'react'
import { usePathname, useRouter } from 'next/navigation'
import { RefreshCw, LogOut, Menu } from 'lucide-react'
import { clearApiKey } from '@/lib/api'
import { queryClient } from '@/lib/query-client'
import { Button } from '@/components/ui/button'

const PAGE_TITLES: Record<string, string> = {
  // CBB betting pages
  '/dashboard': 'Dashboard',
  '/performance': 'Performance',
  '/clv': 'CLV Analysis',
  '/bet-history': 'Bet History',
  '/calibration': 'Calibration',
  '/alerts': 'Alerts',
  '/today': "Today's Bets",
  '/live-slate': 'Live Slate',
  '/odds-monitor': 'Odds Monitor',
  // Fantasy War Room
  '/decisions': 'Daily Decisions',
  '/war-room': 'War Room',
  '/war-room/streaming': 'Streaming Station',
  '/war-room/roster-lab': 'Roster Lab',
}

interface HeaderProps {
  onMenuClick?: () => void
}

export default function Header({ onMenuClick }: HeaderProps) {
  const pathname = usePathname()
  const router = useRouter()
  const [secondsAgo, setSecondsAgo] = useState(0)
  const [lastRefresh, setLastRefresh] = useState(Date.now())
  const [refreshing, setRefreshing] = useState(false)

  const title = PAGE_TITLES[pathname] ?? 'Dashboard'

  useEffect(() => {
    const interval = setInterval(() => {
      setSecondsAgo(Math.floor((Date.now() - lastRefresh) / 1000))
    }, 1000)
    return () => clearInterval(interval)
  }, [lastRefresh])

  function handleRefresh() {
    setRefreshing(true)
    queryClient.invalidateQueries()
    setLastRefresh(Date.now())
    setSecondsAgo(0)
    setTimeout(() => setRefreshing(false), 800)
  }

  function handleLogout() {
    clearApiKey()
    router.push('/login')
  }

  function formatSecondsAgo(s: number): string {
    if (s < 60) return `${s}s ago`
    const m = Math.floor(s / 60)
    if (m < 60) return `${m}m ago`
    return `${Math.floor(m / 60)}h ago`
  }

  return (
    <header className="h-14 bg-white border-b border-gray-200 flex items-center px-4 gap-3 flex-shrink-0">
      {/* Hamburger — mobile only */}
      <Button
        variant="ghost"
        size="icon"
        onClick={onMenuClick}
        className="md:hidden text-gray-500 hover:text-gray-900 flex-shrink-0"
        aria-label="Open navigation"
      >
        <Menu className="h-5 w-5" />
      </Button>

      <span className="text-base font-semibold text-gray-900 flex-1 truncate">{title}</span>

      <div className="flex items-center gap-3">
        <span className="text-xs text-gray-500">
          Last updated: {formatSecondsAgo(secondsAgo)}
        </span>

        <Button
          variant="ghost"
          size="icon"
          onClick={handleRefresh}
          title="Refresh data"
          className="text-gray-500 hover:text-gray-900"
        >
          <RefreshCw className={`h-4 w-4 ${refreshing ? 'animate-spin' : ''}`} />
        </Button>

        <Button
          variant="ghost"
          size="icon"
          onClick={handleLogout}
          title="Sign out"
          className="text-gray-500 hover:text-red-600"
        >
          <LogOut className="h-4 w-4" />
        </Button>
      </div>
    </header>
  )
}
