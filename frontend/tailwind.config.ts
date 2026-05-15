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
      fontFamily: {
        sans: ['var(--font-inter)', ...defaultTheme.fontFamily.sans],
        mono: ['var(--font-mono)', ...defaultTheme.fontFamily.mono],
      },
      colors: {
        'signal-bet': '#fbbf24',
        'signal-consider': '#38bdf8',
        'signal-pass': '#71717a',
        'signal-win': '#22c55e',
        'signal-loss': '#ef4444',
        /* Design System v2 — semantic tokens */
        'bg-base': '#0c0c10',
        'bg-surface': '#16161e',
        'bg-elevated': '#1f1f2a',
        'bg-inset': '#0a0a0f',
        'text-primary': '#e8e8f0',
        'text-secondary': '#a0a0b8',
        'text-tertiary': '#6b6b8a',
        'text-muted': '#4a4a60',
        'border-subtle': '#272733',
        'border-default': '#3a3a4d',
        'status-safe': '#22c55e',
        'status-lead': '#84cc16',
        'status-bubble': '#f59e0b',
        'status-behind': '#f97316',
        'status-lost': '#ef4444',
        'accent-gold': '#FFC000',
      },
    },
  },
  plugins: [],
}

export default config
