'use client'

import { Component, type ReactNode } from 'react'
import { AlertCircle } from 'lucide-react'

interface Props {
  children: ReactNode
  fallback?: ReactNode
}

interface State {
  hasError: boolean
  error?: Error
}

export class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props)
    this.state = { hasError: false }
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error }
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    // eslint-disable-next-line no-console
    console.error('ErrorBoundary caught error:', error, errorInfo)
  }

  render() {
    if (this.state.hasError) {
      if (this.props.fallback) {
        return this.props.fallback
      }
      return (
        <div className="min-h-[40vh] flex items-center justify-center p-6">
          <div className="bg-bg-surface border border-status-lost/30 rounded-lg p-6 max-w-md w-full">
            <div className="flex items-center gap-2 text-status-lost mb-3">
              <AlertCircle className="h-5 w-5" />
              <span className="text-sm font-semibold">Something went wrong</span>
            </div>
            <p className="text-text-secondary text-sm">
              {this.state.error?.message ?? 'An unexpected error occurred.'}
            </p>
            <button
              onClick={() => this.setState({ hasError: false })}
              className="mt-4 text-xs text-accent-gold hover:text-amber-300 font-semibold"
            >
              Try again
            </button>
          </div>
        </div>
      )
    }

    return this.props.children
  }
}
