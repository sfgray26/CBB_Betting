'use client'

import { useState } from 'react'
import { useRouter } from 'next/navigation'
import { setApiKey } from '@/lib/api'

const BASE_URL = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000'

export default function LoginPage() {
  const router = useRouter()
  const [apiKey, setApiKeyState] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    if (!apiKey.trim()) {
      setError('Please enter your API key.')
      return
    }
    setLoading(true)
    setError('')
    try {
      const res = await fetch(`${BASE_URL}/health`, {
        headers: { 'X-API-Key': apiKey.trim() },
      })
      if (res.ok) {
        setApiKey(apiKey.trim())
        router.push('/performance')
      } else if (res.status === 401) {
        setError('Invalid API key. Please check and try again.')
      } else {
        setError(`Server returned an error (${res.status}). Please try again.`)
      }
    } catch {
      setError(`Could not reach the backend at ${BASE_URL}. Check that NEXT_PUBLIC_API_URL is set correctly and the server is running.`)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-zinc-950 flex items-center justify-center p-4">
      <div className="w-full max-w-md">
        {/* Logo / Title */}
        <div className="text-center mb-8">
          <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl bg-amber-400/10 border border-amber-400/20 mb-4">
            <span className="text-3xl">🏀</span>
          </div>
          <h1 className="text-3xl font-bold text-zinc-50 tracking-tight">
            CBB <span className="text-amber-400">Edge</span>
          </h1>
          <p className="mt-2 text-zinc-400 text-sm">
            College Basketball Betting Analytics
          </p>
        </div>

        {/* Card */}
        <div className="bg-zinc-900 border border-zinc-800 rounded-xl p-8 shadow-2xl">
          <p className="text-zinc-300 text-sm text-center mb-4">
            Enter your API key to continue
          </p>
          <p className="text-zinc-600 text-xs text-center mb-6 font-mono truncate" title={BASE_URL}>
            ↗ {BASE_URL}
          </p>

          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label htmlFor="apikey" className="block text-xs font-medium text-zinc-400 uppercase tracking-wider mb-2">
                API Key
              </label>
              <input
                id="apikey"
                type="password"
                autoComplete="current-password"
                placeholder="Enter your API key..."
                value={apiKey}
                onChange={(e) => setApiKeyState(e.target.value)}
                className="w-full bg-zinc-800 border border-zinc-700 rounded-lg px-4 py-3 text-zinc-50 placeholder-zinc-500 font-mono text-sm focus:outline-none focus:ring-2 focus:ring-amber-400/50 focus:border-amber-400/50 transition-colors"
              />
            </div>

            {error && (
              <div className="bg-rose-500/10 border border-rose-500/30 rounded-lg px-4 py-3 text-rose-400 text-sm">
                {error}
              </div>
            )}

            <button
              type="submit"
              disabled={loading}
              className="w-full bg-amber-400 hover:bg-amber-300 disabled:bg-amber-400/50 disabled:cursor-not-allowed text-zinc-950 font-semibold rounded-lg px-4 py-3 text-sm transition-colors focus:outline-none focus:ring-2 focus:ring-amber-400/50 focus:ring-offset-2 focus:ring-offset-zinc-900"
            >
              {loading ? (
                <span className="flex items-center justify-center gap-2">
                  <svg className="animate-spin h-4 w-4" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                  </svg>
                  Signing in...
                </span>
              ) : (
                'Sign In'
              )}
            </button>
          </form>
        </div>

        <p className="text-center text-zinc-600 text-xs mt-6">
          CBB Edge Analytics &mdash; v9.1
        </p>
      </div>
    </div>
  )
}
