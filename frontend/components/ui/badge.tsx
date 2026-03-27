import { cn } from '@/lib/utils'
import { cva, type VariantProps } from 'class-variance-authority'

const badgeVariants = cva(
  'inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium font-mono',
  {
    variants: {
      variant: {
        bet: 'bg-amber-400/15 text-amber-400 border border-amber-400/30',
        consider: 'bg-sky-400/15 text-sky-400 border border-sky-400/30',
        pass: 'bg-zinc-700/50 text-zinc-400 border border-zinc-700',
        win: 'bg-emerald-400/15 text-emerald-400 border border-emerald-400/30',
        loss: 'bg-rose-500/15 text-rose-400 border border-rose-500/30',
        push: 'bg-zinc-700/50 text-zinc-400 border border-zinc-700',
        pending: 'bg-sky-400/10 text-sky-400 border border-sky-400/20',
        confirmed: 'bg-emerald-400/15 text-emerald-400',
        caution: 'bg-amber-400/15 text-amber-400',
        volatile: 'bg-rose-500/15 text-rose-400',
        default: 'bg-zinc-800 text-zinc-300 border border-zinc-700',
        secondary: 'bg-zinc-700/50 text-zinc-400 border border-zinc-600',
      },
    },
    defaultVariants: { variant: 'default' },
  },
)

export interface BadgeProps
  extends React.HTMLAttributes<HTMLSpanElement>,
    VariantProps<typeof badgeVariants> {}

export function Badge({ className, variant, ...props }: BadgeProps) {
  return <span className={cn(badgeVariants({ variant }), className)} {...props} />
}
