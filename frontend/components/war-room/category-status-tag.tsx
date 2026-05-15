import { cn } from '@/lib/utils'

type TagState = 'locked-win' | 'leaning-win' | 'bubble' | 'leaning-loss' | 'locked-loss' | 'pending'

function getState(winProb: number | null): TagState {
  if (winProb === null) return 'pending'
  if (winProb > 0.95) return 'locked-win'
  if (winProb > 0.65) return 'leaning-win'
  if (winProb >= 0.35) return 'bubble'
  if (winProb >= 0.05) return 'leaning-loss'
  return 'locked-loss'
}

const STATE_CONFIG: Record<TagState, { label: string; cls: string }> = {
  'locked-win':   { label: 'LOCK W',   cls: 'bg-accent-gold text-black font-bold' },
  'leaning-win':  { label: 'LEAN W',   cls: 'border border-accent-gold text-accent-gold' },
  'bubble':       { label: 'BUBBLE',   cls: 'border border-status-bubble text-status-bubble' },
  'leaning-loss': { label: 'LEAN L',   cls: 'border border-text-secondary text-text-secondary' },
  'locked-loss':  { label: 'LOCK L',   cls: 'bg-text-muted text-text-secondary font-bold' },
  'pending':      { label: '—',        cls: 'text-text-muted' },
}

interface Props {
  winProb: number | null
  className?: string
}

export function CategoryStatusTag({ winProb, className }: Props) {
  const state = getState(winProb)
  const { label, cls } = STATE_CONFIG[state]

  return (
    <span
      className={cn(
        'inline-flex items-center justify-center px-1.5 py-0.5 text-[10px] tracking-widest uppercase whitespace-nowrap',
        cls,
        className,
      )}
    >
      {label}
    </span>
  )
}
