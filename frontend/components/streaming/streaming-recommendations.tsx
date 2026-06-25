'use client'

import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { endpoints } from '@/lib/api'
import type { StreamingPitcher } from '@/lib/types'
import { Loader2, AlertCircle, ChevronDown, ChevronUp, Filter, Play, Sparkles } from 'lucide-react'
import { FreshnessBadge } from '@/components/freshness/freshness-badge'
import { ActionModal } from './action-modal'

type RecommendationTier = 'EXCELLENT' | 'GOOD' | 'AVERAGE' | 'AVOID' | 'ALL'

const TIER_COLORS: Record<RecommendationTier, string> = {
  EXCELLENT: 'bg-status-safe/15 text-status-safe border-status-safe/30',
  GOOD: 'bg-blue-400/15 text-blue-400 border-blue-400/30',
  AVERAGE: 'bg-status-bubble/15 text-status-bubble border-status-bubble/30',
  AVOID: 'bg-status-lost/15 text-status-lost border-status-lost/30',
  ALL: 'bg-bg-surface text-text-secondary border-border-subtle',
}

const TIER_ORDER: RecommendationTier[] = ['EXCELLENT', 'GOOD', 'AVERAGE', 'AVOID']

export function StreamingRecommendations({ targetDate }: { targetDate: string }) {
  const [tierFilter, setTierFilter] = useState<RecommendationTier>('ALL')
  const [sortField, setSortField] = useState<'quality' | 'name'>('quality')
  const [sortDesc, setSortDesc] = useState(true)
  const [expandedRows, setExpandedRows] = useState<Set<number>>(new Set())
  const [selectedPitcher, setSelectedPitcher] = useState<StreamingPitcher | null>(null)
  const [isModalOpen, setIsModalOpen] = useState(false)
  const [autoStreamEnabled, setAutoStreamEnabled] = useState(false)
  const [pendingActions, setPendingActions] = useState<number[]>([])

  const { data, isLoading, isError, error, refetch } = useQuery({
    queryKey: ['streaming-recommendations', targetDate],
    queryFn: () => endpoints.getStreamingRecommendations(targetDate, 7),
    staleTime: 5 * 60_000,
    retry: 1,
  })

  const toggleRow = (id: number) => {
    setExpandedRows(prev => {
      const next = new Set(prev)
      if (next.has(id)) {
        next.delete(id)
      } else {
        next.add(id)
      }
      return next
    })
  }

  const openActionModal = (pitcher: StreamingPitcher) => {
    setSelectedPitcher(pitcher)
    setIsModalOpen(true)
  }

  const handleActionSuccess = (transactionId: string) => {
    // In a real implementation, you might show a toast notification
    console.log('Action succeeded with transaction:', transactionId)
    // Refresh the streaming data
    refetch()
  }

  const handleAutoStreamToggle = () => {
    if (!autoStreamEnabled) {
      // Show confirmation before enabling
      if (confirm('Auto-Stream will automatically add EXCELLENT + HIGH confidence pitchers. Make sure your drop priority list is configured. Continue?')) {
        setAutoStreamEnabled(true)
        // Queue up EXCELLENT + HIGH confidence pitchers
        const excellentHighPitchers = data?.two_start_pitchers.filter(
          p => p.recommendation === 'EXCELLENT' && p.transparency.confidence === 'HIGH'
        ) || []
        setPendingActions(excellentHighPitchers.map(p => p.bdl_player_id))
      }
    } else {
      setAutoStreamEnabled(false)
      setPendingActions([])
    }
  }

  const cancelPendingAction = (bdlId: number) => {
    setPendingActions(prev => prev.filter(id => id !== bdlId))
  }

  if (isLoading) {
    return (
      <div className="bg-bg-surface border border-border-subtle rounded-lg p-6">
        <div className="flex items-center gap-2 text-text-secondary">
          <Loader2 className="h-4 w-4 animate-spin" />
          <span className="text-sm">Loading streaming recommendations...</span>
        </div>
      </div>
    )
  }

  if (isError) {
    return (
      <div className="bg-bg-surface border border-border-subtle rounded-lg p-6">
        <div className="flex items-center gap-2 text-status-lost">
          <AlertCircle className="h-4 w-4" />
          <span className="text-sm">{error?.message ?? 'Failed to load recommendations'}</span>
        </div>
      </div>
    )
  }

  if (!data) {
    return (
      <div className="bg-bg-surface border border-border-subtle rounded-lg p-6">
        <p className="text-text-secondary text-sm">No recommendations available.</p>
      </div>
    )
  }

  const filtered = data.two_start_pitchers.filter(p =>
    tierFilter === 'ALL' || p.recommendation === tierFilter
  )

  const sorted = [...filtered].sort((a, b) => {
    if (sortField === 'quality') {
      return sortDesc ? a.overall_quality - b.overall_quality : b.overall_quality - a.overall_quality
    }
    return sortDesc ? a.name.localeCompare(b.name) : b.name.localeCompare(a.name)
  })

  return (
    <div className="space-y-4">
      {/* Header with freshness and Auto-Stream toggle */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <h2 className="text-sm font-semibold tracking-widest uppercase text-text-primary">
            2-Start Pitchers
          </h2>
          <span className="text-xs text-text-muted">
            ({sorted.length} found)
          </span>
          {data.freshness && (
            <FreshnessBadge
              severity={data.freshness.staleness_ms != null && data.freshness.staleness_ms > 3600000 ? 'warning' : 'fresh'}
              minutesAgo={data.freshness.staleness_ms != null ? Math.floor(data.freshness.staleness_ms / 60000) : null}
              warningText={data.freshness.staleness_ms != null && data.freshness.staleness_ms > 3600000 ? 'Data stale' : null}
              isClickable={true}
              onRefresh={() => refetch()}
            />
          )}
        </div>

        {/* Auto-Stream Toggle (Beta) */}
        <div className="flex items-center gap-2">
          <button
            onClick={handleAutoStreamToggle}
            className={`flex items-center gap-2 px-3 py-1.5 rounded-md text-xs font-medium border transition-colors ${
              autoStreamEnabled
                ? 'bg-accent-primary/20 text-accent-primary border-accent-primary/30'
                : 'bg-bg-inset text-text-muted border-border-subtle hover:border-border-default'
            }`}
          >
            {autoStreamEnabled ? (
              <>
                <Sparkles className="h-3.5 w-3.5" />
                Auto-Stream ON
              </>
            ) : (
              <>
                <Play className="h-3.5 w-3.5" />
                Auto-Stream
              </>
            )}
          </button>
          <span className="text-[9px] px-1.5 py-0.5 rounded bg-bg-elevated text-text-muted border border-border-subtle">
            BETA
          </span>
        </div>
      </div>

      {/* Pending Actions Queue */}
      {autoStreamEnabled && pendingActions.length > 0 && (
        <div className="bg-accent-primary/10 border border-accent-primary/30 rounded-lg p-3">
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs font-medium text-accent-primary">
              Pending Actions ({pendingActions.length})
            </span>
            <button
              onClick={() => setAutoStreamEnabled(false)}
              className="text-xs text-text-muted hover:text-text-primary transition-colors"
            >
              Cancel All
            </button>
          </div>
          <div className="space-y-1">
            {pendingActions.map(id => {
              const pitcher = data?.two_start_pitchers.find(p => p.bdl_player_id === id)
              return pitcher ? (
                <div key={id} className="flex items-center justify-between text-xs bg-bg-surface rounded px-2 py-1">
                  <span className="text-text-secondary">{pitcher.name}</span>
                  <button
                    onClick={() => cancelPendingAction(id)}
                    className="text-text-muted hover:text-text-lost transition-colors"
                  >
                    Cancel
                  </button>
                </div>
              ) : null
            })}
          </div>
        </div>
      )}

      {/* Filters */}
      <div className="bg-bg-elevated border border-border-subtle rounded-lg p-3">
        <div className="flex items-center gap-4 flex-wrap">
          <div className="flex items-center gap-2">
            <Filter className="h-3.5 w-3.5 text-text-muted" />
            <span className="text-xs text-text-muted uppercase tracking-wider">Tier:</span>
          </div>
          {(['ALL', ...TIER_ORDER] as RecommendationTier[]).map(tier => (
            <button
              key={tier}
              onClick={() => setTierFilter(tier)}
              className={`px-2.5 py-1 rounded-md text-xs font-medium border transition-colors ${
                tierFilter === tier
                  ? TIER_COLORS[tier]
                  : 'bg-bg-inset text-text-tertiary border-border-subtle hover:border-border-default'
              }`}
            >
              {tier}
            </button>
          ))}
          <div className="flex-1" />
          <div className="flex items-center gap-2">
            <button
              onClick={() => {
                if (sortField === 'quality') setSortDesc(!sortDesc)
                else { setSortField('quality'); setSortDesc(true) }
              }}
              className={`text-xs font-medium transition-colors ${
                sortField === 'quality' ? 'text-accent-gold' : 'text-text-muted hover:text-text-secondary'
              }`}
            >
              Quality {sortField === 'quality' && (sortDesc ? '↓' : '↑')}
            </button>
            <button
              onClick={() => {
                if (sortField === 'name') setSortDesc(!sortDesc)
                else { setSortField('name'); setSortDesc(true) }
              }}
              className={`text-xs font-medium transition-colors ${
                sortField === 'name' ? 'text-accent-gold' : 'text-text-muted hover:text-text-secondary'
              }`}
            >
              Name {sortField === 'name' && (sortDesc ? '↓' : '↑')}
            </button>
          </div>
        </div>
      </div>

      {/* Results */}
      {sorted.length === 0 ? (
        <div className="bg-bg-surface border border-border-subtle rounded-lg p-6 text-center">
          <p className="text-text-secondary text-sm">
            {tierFilter === 'ALL'
              ? 'No 2-start pitchers found for this date range.'
              : `No pitchers with ${tierFilter} recommendation.`}
          </p>
        </div>
      ) : (
        <div className="space-y-2">
          {sorted.map(pitcher => (
            <StreamingPitcherRow
              key={pitcher.bdl_player_id}
              pitcher={pitcher}
              isExpanded={expandedRows.has(pitcher.bdl_player_id)}
              onToggle={() => toggleRow(pitcher.bdl_player_id)}
              onExecuteAdd={() => openActionModal(pitcher)}
              isPendingAction={pendingActions.includes(pitcher.bdl_player_id)}
            />
          ))}
        </div>
      )}

      {/* Data sources footer */}
      <div className="text-[10px] text-text-muted uppercase tracking-wider">
        Data sources: {data.data_sources.join(', ')}
      </div>

      {/* Action Modal */}
      {selectedPitcher && (
        <ActionModal
          pitcher={selectedPitcher}
          isOpen={isModalOpen}
          onClose={() => setIsModalOpen(false)}
          onSuccess={handleActionSuccess}
        />
      )}
    </div>
  )
}

function StreamingPitcherRow({
  pitcher,
  isExpanded,
  onToggle,
  onExecuteAdd,
  isPendingAction,
}: {
  pitcher: StreamingPitcher
  isExpanded: boolean
  onToggle: () => void
  onExecuteAdd: () => void
  isPendingAction: boolean
}) {
  const tierColor = TIER_COLORS[pitcher.recommendation]

  // Check if button should be disabled
  const isButtonDisabled =
    pitcher.recommendation === 'AVOID' || pitcher.transparency.confidence === 'LOW' || isPendingAction

  return (
    <div className={`bg-bg-surface border rounded-lg overflow-hidden transition-all ${
      pitcher.recommendation === 'EXCELLENT' ? 'border-accent-gold/30' : 'border-border-subtle'
    }`}>
      {/* Main row */}
      <div
        onClick={onToggle}
        className="px-4 py-3 cursor-pointer hover:bg-bg-inset transition-colors"
      >
        <div className="flex items-center justify-between gap-4">
          {/* Left: Name, Team, Handedness */}
          <div className="min-w-0 flex items-center gap-3 flex-1">
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2">
                <span className="text-sm font-medium text-text-primary truncate">
                  {pitcher.name}
                </span>
                <span className="text-xs text-text-secondary">{pitcher.team}</span>
                <span className="text-[10px] text-text-muted uppercase">{pitcher.handedness}</span>
              </div>
              <div className="flex items-center gap-2 mt-1">
                <span className="text-[10px] text-text-muted">
                  {pitcher.starts.length} start{pitcher.starts.length !== 1 ? 's' : ''}
                </span>
                {pitcher.transparency.confidence && (
                  <span className={`text-[10px] px-1.5 py-0.5 rounded font-semibold ${
                    pitcher.transparency.confidence === 'HIGH' ? 'bg-status-safe/20 text-status-safe' :
                    pitcher.transparency.confidence === 'MEDIUM' ? 'bg-status-bubble/20 text-status-bubble' :
                    'bg-status-lost/20 text-status-lost'
                  }`}>
                    {pitcher.transparency.confidence}
                  </span>
                )}
              </div>
            </div>
          </div>

          {/* Center: Quality score */}
          <div className="flex-shrink-0 text-center px-3">
            <div className="text-[10px] text-text-muted uppercase tracking-wider">Quality</div>
            <div className={`text-lg font-mono font-bold ${
              pitcher.overall_quality >= 1.0 ? 'text-status-safe' :
              pitcher.overall_quality >= 0.3 ? 'text-blue-400' :
              pitcher.overall_quality >= -0.3 ? 'text-status-bubble' :
              'text-status-lost'
            }`}>
              {pitcher.overall_quality.toFixed(1)}
            </div>
          </div>

          {/* Right: Recommendation badge + Execute Add */}
          <div className="flex items-center gap-2">
            <span className={`px-3 py-1.5 rounded-md text-xs font-bold border ${tierColor}`}>
              {pitcher.recommendation}
            </span>

            {/* Execute Add button */}
            <button
              onClick={(e) => {
                e.stopPropagation()
                onExecuteAdd()
              }}
              disabled={isButtonDisabled}
              className={`px-3 py-1.5 rounded-md text-xs font-medium border transition-all ${
                isButtonDisabled
                  ? 'bg-bg-inset text-text-muted border-border-subtle cursor-not-allowed opacity-50'
                  : 'bg-accent-primary/20 text-accent-primary border-accent-primary/30 hover:bg-accent-primary/30 cursor-pointer'
              }`}
              title={
                isButtonDisabled
                  ? pitcher.recommendation === 'AVOID'
                    ? 'Cannot add AVOID recommendations'
                    : pitcher.transparency.confidence === 'LOW'
                    ? 'Cannot add LOW confidence recommendations'
                    : 'Action pending'
                  : 'Add this player to your roster'
              }
            >
              {isPendingAction ? (
                <Loader2 className="h-3 w-3 animate-spin" />
              ) : (
                'Execute Add'
              )}
            </button>
          </div>

          {/* Expand icon */}
          <div className="flex-shrink-0 pl-2">
            {isExpanded ? (
              <ChevronUp className="h-4 w-4 text-text-muted" />
            ) : (
              <ChevronDown className="h-4 w-4 text-text-muted" />
            )}
          </div>
        </div>

        {/* Risk note */}
        <div className="mt-2 text-[10px] text-text-muted">
          {pitcher.risk_note}
        </div>
      </div>

      {/* Expanded start details */}
      {isExpanded && (
        <div className="border-t border-border-subtle bg-bg-inset">
          <div className="p-4 space-y-2">
            {pitcher.starts.map((start, idx) => (
              <div
                key={idx}
                className="flex items-center justify-between text-xs bg-bg-surface rounded px-3 py-2"
              >
                <div className="flex items-center gap-3">
                  <span className="text-text-muted w-16">{start.date}</span>
                  <span className="text-text-primary font-medium">{start.pitcher_name}</span>
                  <span className="text-text-secondary">@ {start.opponent}</span>
                  <span className={`text-[10px] px-1.5 py-0.5 rounded ${
                    start.is_home ? 'bg-accent-primary/20 text-accent-primary' : 'bg-bg-inset text-text-muted'
                  }`}>
                    {start.is_home ? 'HOME' : 'AWAY'}
                  </span>
                </div>
                <div className="flex items-center gap-3">
                  <span className="text-text-muted">{start.game_time_et}</span>
                  <div className="flex items-center gap-1">
                    <span className="text-text-muted">Q:</span>
                    <span className={`font-mono font-bold ${
                      start.quality_score >= 1.0 ? 'text-status-safe' :
                      start.quality_score >= 0.3 ? 'text-blue-400' :
                      start.quality_score >= -0.3 ? 'text-status-bubble' :
                      'text-status-lost'
                    }`}>
                      {start.quality_score.toFixed(1)}
                    </span>
                  </div>
                  {start.is_confirmed && (
                    <span className="text-[9px] px-1.5 py-0.5 rounded bg-status-safe/20 text-status-safe font-semibold">
                      CONFIRMED
                    </span>
                  )}
                </div>
              </div>
            ))}
          </div>

          {/* Transparency factors */}
          {pitcher.transparency.factors.length > 0 && (
            <div className="px-4 pb-3">
              <div className="flex flex-wrap gap-1.5">
                {pitcher.transparency.factors.map((factor, idx) => (
                  <span
                    key={idx}
                    className="text-[9px] px-2 py-1 rounded bg-bg-elevated text-text-tertiary border border-border-subtle"
                  >
                    {factor}
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
