import { cn } from '@/lib/utils'

export function Card({ className, children, ...props }: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn('rounded-lg border border-zinc-800 bg-zinc-900 p-6', className)}
      {...props}
    >
      {children}
    </div>
  )
}

export function CardHeader({
  className,
  children,
  ...props
}: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div className={cn('mb-4', className)} {...props}>
      {children}
    </div>
  )
}

export function CardTitle({
  className,
  children,
  ...props
}: React.HTMLAttributes<HTMLHeadingElement>) {
  return (
    <h3
      className={cn('text-sm font-medium text-zinc-400 uppercase tracking-wider', className)}
      {...props}
    >
      {children}
    </h3>
  )
}

export function CardValue({
  className,
  children,
  ...props
}: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn('text-2xl font-semibold font-mono tabular-nums text-zinc-50', className)}
      {...props}
    >
      {children}
    </div>
  )
}
