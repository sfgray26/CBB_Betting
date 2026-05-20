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
        /* Signal Colors */
        'signal-bet': '#d97706',
        'signal-consider': '#0891b2',
        'signal-pass': '#6b7280',
        'signal-win': '#16a34a',
        'signal-loss': '#dc2626',

        /* Design System v3 — Light Theme Semantic Tokens */
        'bg-base': '#ffffff',
        'bg-surface': '#f8f9fa',
        'bg-elevated': '#ffffff',
        'bg-inset': '#f1f3f5',
        'bg-hover': '#e9ecef',
        'bg-active': '#dee2e6',

        'text-primary': '#212529',
        'text-secondary': '#495057',
        'text-tertiary': '#6c757d',
        'text-muted': '#adb5bd',

        'border-subtle': '#e9ecef',
        'border-default': '#dee2e6',
        'border-focus': '#adb5bd',

        'status-safe': '#16a34a',
        'status-lead': '#65a30d',
        'status-bubble': '#d97706',
        'status-behind': '#ea580c',
        'status-lost': '#dc2626',

        'accent-primary': '#2563eb',
        'accent-primary-hover': '#1d4ed8',
        'accent-gold': '#d97706',
      },
    },
  },
  plugins: [],
}

export default config
