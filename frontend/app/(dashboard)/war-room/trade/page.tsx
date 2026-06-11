'use client'

import { useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import { endpoints } from '@/lib/api'
import type { TradeAnalysisResponse, TradeCategoryDelta, TradePlayer } from '@/lib/types'
import { ArrowLeftRight, Plus, X, Loader2, AlertCircle, TrendingUp, TrendingDown, Minus } from 'lucide-react'
import { cn } from '@/lib/utils'

// Display-friendly label for backend cat_score keys (lowercase board keys)
const CAT_DISPLAY: Record<string, string> = {
  r: 'R', h: 'H', hr_b: 'HR', rbi: 'RBI', k_b: 'K', tb: 'TB', avg: 'AVG', ops: 'OPS', nsb: 'NSB',
  w: 'W', l: 'L', hr_p: 'HRA', k_p: 'Ks', era: 'ERA', whip: 'WHIP', k_9: 'K/9', qs: 'QS', nsv: 'SV',
}
function catLabel(key: string): string {
  return CAT_DISPLAY[key.toLowerCase()] ?? key.toUpperCase()
}

const RECOMMENDATION_CONFIG: Record<string, { label: string; color: string; bg: string; border: string }> = {
  strong_accept: { label: 'STRONG ACCEPT', color: 'text-status-safe', bg: 'bg-status-safe/10', border: 'border-status-safe/30' },
  accept:        { label: 'ACCEPT',        color: 'text-status-safe', bg: 'bg-status-safe/5',  border: 'border-status-safe/20' },
  neutral:       { label: 'NEUTRAL',       color: 'text-status-bubble', bg: 'bg-status-bubble/10', border: 'border-status-bubble/30' },
  reject:        { label: 'REJECT',        color: 'text-status-lost', bg: 'bg-status-lost/5',  border: 'border-status-lost/20' },
  strong_reject: { label: 'STRONG REJECT', color: 'text-status-lost', bg: 'bg-status-lost/10', border: 'border-status-lost/30' },
}

function PlayerChip({ name, onRemove }: { name: string; onRemove: () => void }) {
  return (
    <div className="inline-flex items-center gap-1.5 bg-bg-elevated border border-border-subtle rounded-full px-3 py-1.5">
      <span className="text-xs font-semibold text-text-primary">{name}</span>
      <button onClick={onRemove} className="text-text-muted hover:text-status-lost transition-colors">
        <X className="h-3 w-3" />
      </button>
    </div>
  )
}

function PlayerInputRow({ label, players, onAdd, onRemove }: {
  label: string
  players: string[]
  onAdd: (name: string) => void
  onRemove: (index: number) => void
}) {
  const [input, setInput] = useState('')

  function handleAdd() {
    const trimmed = input.trim()
    if (trimmed) {
      onAdd(trimmed)
      setInput('')
    }
  }

  return (
    <div className="space-y-2">
      <p className="text-xs font-bold tracking-widest uppercase text-text-muted">{label}</p>
      <div className="flex gap-2">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && handleAdd()}
          placeholder="Player name…"
          className="flex-1 bg-bg-elevated border border-border-subtle rounded-lg px-3 py-2 text-sm text-text-primary placeholder:text-text-muted focus:outline-none focus:border-accent-gold/50 transition-colors"
        />
        <button
          onClick={handleAdd}
          disabled={!input.trim()}
          className="flex items-center gap-1 px-3 py-2 bg-bg-elevated border border-border-subtle rounded-lg text-xs font-semibold text-text-secondary hover:text-accent-gold hover:border-accent-gold/30 transition-colors disabled:opacity-40"
        >
          <Plus className="h-3.5 w-3.5" />
          Add
        </button>
      </div>
      {players.length > 0 && (
        <div className="flex flex-wrap gap-2 pt-1">
          {players.map((name, i) => (
            <PlayerChip key={i} name={name} onRemove={() => onRemove(i)} />
          ))}
        </div>
      )}
    </div>
  )
}

function CategoryDeltaRow({ delta }: { delta: TradeCategoryDelta }) {
  const isGain = delta.direction === 'gain'
  const isLoss = delta.direction === 'loss'
  const label = catLabel(delta.category)

  return (
    <div className="flex items-center gap-3 py-2 border-b border-border-subtle last:border-0">
      <span className="w-12 text-xs font-semibold text-text-secondary uppercase tracking-wider flex-shrink-0">{label}</span>
      <div className="flex-1 grid grid-cols-3 gap-2 text-xs tabular-nums">
        <span className="text-text-muted text-right">
          {delta.give_z >= 0 ? '+' : ''}{delta.give_z.toFixed(2)}z
        </span>
        <span className="text-center text-text-muted">→</span>
        <span className={cn('text-left font-semibold', isGain ? 'text-status-safe' : isLoss ? 'text-status-lost' : 'text-text-secondary')}>
          {delta.receive_z >= 0 ? '+' : ''}{delta.receive_z.toFixed(2)}z
        </span>
      </div>
      <div className="w-16 flex items-center justify-end gap-1 flex-shrink-0">
        {isGain && <TrendingUp className="h-3 w-3 text-status-safe" />}
        {isLoss && <TrendingDown className="h-3 w-3 text-status-lost" />}
        {!isGain && !isLoss && <Minus className="h-3 w-3 text-text-muted" />}
        <span className={cn('text-xs font-bold tabular-nums', isGain ? 'text-status-safe' : isLoss ? 'text-status-lost' : 'text-text-muted')}>
          {delta.delta >= 0 ? '+' : ''}{delta.delta.toFixed(2)}
        </span>
      </div>
    </div>
  )
}

function TradeResult({ result }: { result: TradeAnalysisResponse }) {
  const rec = RECOMMENDATION_CONFIG[result.recommendation] ?? RECOMMENDATION_CONFIG['neutral']
  const sorted = [...result.category_deltas].sort((a, b) => Math.abs(b.delta) - Math.abs(a.delta))
  const gains = result.category_deltas.filter(d => d.direction === 'gain')
  const losses = result.category_deltas.filter(d => d.direction === 'loss')

  return (
    <div className="space-y-4">
      {/* Verdict */}
      <div className={cn('rounded-lg p-4 border flex items-center justify-between', rec.bg, rec.border)}>
        <div>
          <p className="text-[10px] font-semibold tracking-widest uppercase text-text-muted mb-1">Trade Verdict</p>
          <p className={cn('text-lg font-bold tracking-wider', rec.color)}>{rec.label}</p>
        </div>
        <div className="text-right">
          <p className="text-[10px] text-text-muted uppercase tracking-wider mb-1">Net z-score</p>
          <p className={cn('text-2xl font-bold font-mono tabular-nums', result.total_z_delta >= 0 ? 'text-status-safe' : 'text-status-lost')}>
            {result.total_z_delta >= 0 ? '+' : ''}{result.total_z_delta.toFixed(2)}
          </p>
        </div>
      </div>

      {/* Summary */}
      <p className="text-sm text-text-secondary leading-relaxed">{result.summary}</p>

      {/* Player summaries */}
      <div className="grid grid-cols-2 gap-3">
        <div className="bg-bg-elevated border border-border-subtle rounded-lg p-3 space-y-1">
          <p className="text-[10px] font-bold tracking-widest uppercase text-status-lost">YOU GIVE</p>
          {result.give_players.map((p, i) => (
            <div key={i} className="text-xs text-text-secondary">
              <span className="font-semibold text-text-primary">{p.name}</span>
              {p.team && <span className="text-text-muted ml-1">{p.team}</span>}
              <span className="ml-1 text-text-muted tabular-nums">
                ({p.z_score >= 0 ? '+' : ''}{p.z_score.toFixed(1)}z)
              </span>
            </div>
          ))}
          {result.give_players.length === 0 && <p className="text-[10px] text-text-muted">No players</p>}
        </div>
        <div className="bg-bg-elevated border border-border-subtle rounded-lg p-3 space-y-1">
          <p className="text-[10px] font-bold tracking-widest uppercase text-status-safe">YOU RECEIVE</p>
          {result.receive_players.map((p, i) => (
            <div key={i} className="text-xs text-text-secondary">
              <span className="font-semibold text-text-primary">{p.name}</span>
              {p.team && <span className="text-text-muted ml-1">{p.team}</span>}
              <span className="ml-1 text-text-muted tabular-nums">
                ({p.z_score >= 0 ? '+' : ''}{p.z_score.toFixed(1)}z)
              </span>
            </div>
          ))}
          {result.receive_players.length === 0 && <p className="text-[10px] text-text-muted">No players</p>}
        </div>
      </div>

      {/* Category impact quick-scan */}
      {(gains.length > 0 || losses.length > 0) && (
        <div className="grid grid-cols-2 gap-3">
          {gains.length > 0 && (
            <div>
              <p className="text-[10px] font-semibold tracking-widest uppercase text-status-safe mb-1">Cats Gained</p>
              <div className="flex flex-wrap gap-1">
                {gains.map(d => (
                  <span key={d.category} className="text-[10px] px-1.5 py-0.5 bg-status-safe/10 text-status-safe border border-status-safe/20 rounded font-semibold">
                    {catLabel(d.category)} +{d.delta.toFixed(1)}z
                  </span>
                ))}
              </div>
            </div>
          )}
          {losses.length > 0 && (
            <div>
              <p className="text-[10px] font-semibold tracking-widest uppercase text-status-lost mb-1">Cats Lost</p>
              <div className="flex flex-wrap gap-1">
                {losses.map(d => (
                  <span key={d.category} className="text-[10px] px-1.5 py-0.5 bg-status-lost/10 text-status-lost border border-status-lost/20 rounded font-semibold">
                    {catLabel(d.category)} {d.delta.toFixed(1)}z
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Full category breakdown */}
      {sorted.length > 0 && (
        <div className="bg-bg-surface border border-border-subtle rounded-lg p-4">
          <div className="flex items-center gap-3 px-0 pb-2 mb-1 border-b border-border-subtle">
            <span className="w-12 text-[10px] font-semibold uppercase tracking-wider text-text-muted">CAT</span>
            <div className="flex-1 grid grid-cols-3 gap-2 text-[10px] uppercase tracking-wider text-text-muted">
              <span className="text-right">Giving</span>
              <span className="text-center" />
              <span>Receiving</span>
            </div>
            <span className="w-16 text-right text-[10px] uppercase tracking-wider text-text-muted">Delta</span>
          </div>
          {sorted.map((d) => <CategoryDeltaRow key={d.category} delta={d} />)}
        </div>
      )}

      <p className="text-[10px] text-text-muted">
        z-scores based on rest-of-season projections (Steamer/Statcast). Higher = better, except ERA/WHIP/L/HRA where lower value is preferred.
      </p>
    </div>
  )
}

export default function TradePage() {
  const [givePlayers, setGivePlayers] = useState<string[]>([])
  const [receivePlayers, setReceivePlayers] = useState<string[]>([])

  const mutation = useMutation({
    mutationFn: () => endpoints.analyzeTrade(givePlayers, receivePlayers),
  })

  function handleAnalyze() {
    if (givePlayers.length === 0 || receivePlayers.length === 0) return
    mutation.mutate()
  }

  function handleReset() {
    setGivePlayers([])
    setReceivePlayers([])
    mutation.reset()
  }

  return (
    <div className="min-h-screen bg-bg-base p-6 space-y-6 max-w-3xl mx-auto">
      {/* Header */}
      <div className="flex items-center gap-3">
        <ArrowLeftRight className="h-6 w-6 text-accent-gold" />
        <span className="text-lg font-bold tracking-widest uppercase text-accent-gold">Trade Analyzer</span>
      </div>
      <p className="text-sm text-text-secondary">
        Enter players on each side of a proposed trade. Get category-level impact and a verdict based on rest-of-season projections.
      </p>

      {/* Trade Input */}
      <div className="bg-bg-surface border border-border-subtle rounded-lg p-5 space-y-5">
        <PlayerInputRow
          label="You Give"
          players={givePlayers}
          onAdd={(n) => setGivePlayers(v => [...v, n])}
          onRemove={(i) => setGivePlayers(v => v.filter((_, idx) => idx !== i))}
        />
        <div className="border-t border-border-subtle" />
        <PlayerInputRow
          label="You Receive"
          players={receivePlayers}
          onAdd={(n) => setReceivePlayers(v => [...v, n])}
          onRemove={(i) => setReceivePlayers(v => v.filter((_, idx) => idx !== i))}
        />
      </div>

      {/* Actions */}
      <div className="flex items-center gap-3">
        <button
          onClick={handleAnalyze}
          disabled={mutation.isPending || givePlayers.length === 0 || receivePlayers.length === 0}
          className="flex items-center gap-2 px-5 py-2.5 bg-accent-gold text-black rounded-lg font-semibold text-sm hover:bg-amber-400 transition-colors disabled:opacity-40"
        >
          {mutation.isPending && <Loader2 className="h-4 w-4 animate-spin" />}
          Analyze Trade
        </button>
        {(mutation.data || mutation.isError) && (
          <button
            onClick={handleReset}
            className="text-xs text-text-muted hover:text-text-secondary font-semibold transition-colors"
          >
            Reset
          </button>
        )}
      </div>

      {/* Error */}
      {mutation.isError && (
        <div className="bg-status-lost/10 border border-status-lost/30 rounded-lg p-4 flex items-start gap-2">
          <AlertCircle className="h-4 w-4 text-status-lost flex-shrink-0 mt-0.5" />
          <div>
            <p className="text-sm font-semibold text-status-lost">Analysis failed</p>
            <p className="text-xs text-text-secondary mt-0.5">
              {mutation.error instanceof Error ? mutation.error.message : 'Unknown error'}
            </p>
            <p className="text-xs text-text-muted mt-1">
              Tip: Use exact player names as they appear on the roster (e.g., "Juan Soto", "Gerrit Cole").
            </p>
          </div>
        </div>
      )}

      {/* Result */}
      {mutation.data && <TradeResult result={mutation.data} />}
    </div>
  )
}
