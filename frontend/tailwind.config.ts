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
        'signal-win': '#34d399',
        'signal-loss': '#f43f5e',
      },
    },
  },
  plugins: [],
}

export default config
