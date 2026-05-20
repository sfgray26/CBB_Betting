import { cn } from '@/lib/utils'
import { cva, type VariantProps } from 'class-variance-authority'

const badgeVariants = cva(
  'inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium font-mono',
  {
    variants: {
      variant: {
        bet: 'bg-amber-100 text-amber-700 border border-amber-200',
        consider: 'bg-sky-100 text-sky-700 border border-sky-200',
        pass: 'bg-gray-100 text-gray-600 border border-gray-200',
        win: 'bg-green-100 text-green-700 border border-green-200',
        loss: 'bg-red-100 text-red-700 border border-red-200',
        push: 'bg-gray-100 text-gray-600 border border-gray-200',
        pending: 'bg-blue-50 text-blue-600 border border-blue-200',
        confirmed: 'bg-green-100 text-green-700',
        caution: 'bg-amber-100 text-amber-700',
        volatile: 'bg-red-100 text-red-700',
        default: 'bg-gray-100 text-gray-700 border border-gray-200',
        secondary: 'bg-gray-50 text-gray-500 border border-gray-200',
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
