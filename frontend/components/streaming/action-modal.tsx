'use client'

import { useState } from 'react'
import { endpoints } from '@/lib/api'
import type { StreamingPitcher, RosterActionResponse } from '@/lib/types'
import {
  X,
  Loader2,
  CheckCircle,
  AlertCircle,
  Info,
  AlertTriangle,
} from 'lucide-react'

interface ActionModalProps {
  pitcher: StreamingPitcher
  isOpen: boolean
  onClose: () => void
  onSuccess: (transactionId: string) => void
}

type ModalStep = 'confirm' | 'executing' | 'success' | 'error'

export function ActionModal({
  pitcher,
  isOpen,
  onClose,
  onSuccess,
}: ActionModalProps) {
  const [step, setStep] = useState<ModalStep>('confirm')
  const [selectedDrop, setSelectedDrop] = useState<string>('')
  const [response, setResponse] = useState<RosterActionResponse | null>(null)

  // Determine if waiver claim might be needed (heuristic)
  const needsWaiverClaim = pitcher.overall_quality > 0.5

  const handleExecute = async () => {
    setStep('executing')

    try {
      const result = await endpoints.rosterAction({
        action: 'ADD',
        add_player_id: `bdl.${pitcher.bdl_player_id}`,
        position: 'P',
        drop_player_id: selectedDrop || undefined,
      })

      setResponse(result)

      if (result.success) {
        setStep('success')
        onSuccess(result.transaction_id ?? '')
      } else {
        setStep('error')
      }
    } catch (error) {
      setResponse({
        success: false,
        errors: [{ code: 'NETWORK_ERROR', message: error instanceof Error ? error.message : 'Failed to execute action' }],
      })
      setStep('error')
    }
  }

  const handleRetry = () => {
    setStep('confirm')
    setResponse(null)
  }

  const handleClose = () => {
    setStep('confirm')
    setSelectedDrop('')
    setResponse(null)
    onClose()
  }

  if (!isOpen) return null

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm">
      <div className="bg-bg-surface border border-border-subtle rounded-lg shadow-xl max-w-md w-full mx-4">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-border-subtle">
          <h3 className="text-lg font-semibold text-text-primary">
            {step === 'confirm' ? 'Confirm Add' : step === 'executing' ? 'Executing...' : step === 'success' ? 'Success!' : 'Action Failed'}
          </h3>
          <button
            onClick={handleClose}
            className="text-text-muted hover:text-text-primary transition-colors"
            disabled={step === 'executing'}
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        {/* Content */}
        <div className="p-6">
          {step === 'confirm' && (
            <div className="space-y-4">
              {/* Player Info */}
              <div className="bg-bg-elevated rounded-lg p-4 border border-border-subtle">
                <div className="flex items-center justify-between mb-3">
                  <div>
                    <h4 className="font-semibold text-text-primary">{pitcher.name}</h4>
                    <p className="text-sm text-text-secondary">{pitcher.team} • {pitcher.handedness}</p>
                  </div>
                  <div className="text-right">
                    <div className={`text-lg font-bold ${
                      pitcher.overall_quality >= 1.0 ? 'text-status-safe' :
                      pitcher.overall_quality >= 0.3 ? 'text-blue-400' :
                      pitcher.overall_quality >= -0.3 ? 'text-status-bubble' :
                      'text-status-lost'
                    }`}>
                      {pitcher.overall_quality.toFixed(1)}
                    </div>
                    <div className="text-[10px] text-text-muted uppercase">Quality</div>
                  </div>
                </div>

                {/* Starts */}
                <div className="space-y-2">
                  {pitcher.starts.map((start, idx) => (
                    <div key={idx} className="flex items-center justify-between text-xs bg-bg-surface rounded px-3 py-2">
                      <div className="flex items-center gap-2">
                        <span className="text-text-muted">{start.date}</span>
                        <span className="text-text-secondary">@ {start.opponent}</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <span className={`font-mono font-bold ${
                          start.quality_score >= 1.0 ? 'text-status-safe' :
                          start.quality_score >= 0.3 ? 'text-blue-400' :
                          'text-status-bubble'
                        }`}>
                          {start.quality_score.toFixed(1)}
                        </span>
                        {start.is_confirmed && (
                          <span className="text-[9px] px-1.5 py-0.5 rounded bg-status-safe/20 text-status-safe font-semibold">
                            CONFIRMED
                          </span>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Warnings */}
              {needsWaiverClaim && (
                <div className="flex items-start gap-2 p-3 bg-status-bubble/10 border border-status-bubble/30 rounded-lg">
                  <Info className="h-4 w-4 text-status-bubble mt-0.5 flex-shrink-0" />
                  <p className="text-sm text-status-bubble">
                    This player may require a waiver claim (not immediate add). Claim priority will be determined by league waiver rules.
                  </p>
                </div>
              )}

              {pitcher.transparency.confidence === 'MEDIUM' && (
                <div className="flex items-start gap-2 p-3 bg-accent-gold/10 border border-accent-gold/30 rounded-lg">
                  <AlertTriangle className="h-4 w-4 text-accent-gold mt-0.5 flex-shrink-0" />
                  <p className="text-sm text-accent-gold">
                    Medium confidence recommendation. Start quality may vary based on opponent and park factors.
                  </p>
                </div>
              )}

              {/* Drop Candidate Selection */}
              <div className="space-y-2">
                <label className="text-sm font-medium text-text-primary">
                  Drop candidate (if roster is full)
                </label>
                <select
                  value={selectedDrop}
                  onChange={(e) => setSelectedDrop(e.target.value)}
                  className="w-full bg-bg-inset border border-border-subtle rounded-md px-3 py-2 text-sm text-text-primary focus:outline-none focus:border-accent-primary"
                >
                  <option value="">-- Select player to drop (optional) --</option>
                  <option value="auto">Let system choose lowest-value player</option>
                  {/* In a real implementation, you'd fetch actual roster players here */}
                  <option value="example.1">Example Player 1 (0.2 z-score)</option>
                  <option value="example.2">Example Player 2 (-0.5 z-score)</option>
                </select>
                <p className="text-xs text-text-muted">
                  Only required if your roster is at maximum capacity. Leave empty to let the system handle it.
                </p>
              </div>
            </div>
          )}

          {step === 'executing' && (
            <div className="flex flex-col items-center justify-center py-8">
              <Loader2 className="h-8 w-8 animate-spin text-accent-primary mb-4" />
              <p className="text-sm text-text-secondary">Executing roster action...</p>
              <p className="text-xs text-text-muted mt-2">This may take a few seconds</p>
            </div>
          )}

          {step === 'success' && (
            <div className="space-y-4">
              <div className="flex flex-col items-center justify-center py-4">
                <CheckCircle className="h-12 w-12 text-status-safe mb-3" />
                <p className="text-lg font-semibold text-text-primary">Player Added Successfully!</p>
                {response?.transaction_id && (
                  <p className="text-sm text-text-muted mt-1">
                    Transaction ID: <code className="bg-bg-inset px-2 py-1 rounded text-xs">{response.transaction_id}</code>
                  </p>
                )}
              </div>

              {response?.warnings && response.warnings.length > 0 && (
                <div className="space-y-2">
                  <p className="text-sm font-medium text-text-primary">Warnings:</p>
                  {response.warnings.map((warning, idx) => (
                    <div key={idx} className="flex items-start gap-2 p-2 bg-accent-gold/10 border border-accent-gold/30 rounded">
                      <AlertTriangle className="h-4 w-4 text-accent-gold mt-0.5 flex-shrink-0" />
                      <div>
                        <p className="text-xs font-mono text-accent-gold">{warning.code}</p>
                        <p className="text-sm text-text-secondary">{warning.message}</p>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {step === 'error' && (
            <div className="space-y-4">
              <div className="flex flex-col items-center justify-center py-4">
                <AlertCircle className="h-12 w-12 text-status-lost mb-3" />
                <p className="text-lg font-semibold text-text-primary">Action Failed</p>
                <p className="text-sm text-text-secondary mt-1">Please review the errors below and try again</p>
              </div>

              {response?.errors && response.errors.length > 0 && (
                <div className="space-y-2">
                  {response.errors.map((error, idx) => (
                    <div key={idx} className="flex items-start gap-2 p-3 bg-status-lost/10 border border-status-lost/30 rounded">
                      <AlertCircle className="h-4 w-4 text-status-lost mt-0.5 flex-shrink-0" />
                      <div>
                        <p className="text-xs font-mono text-status-lost">{error.code}</p>
                        <p className="text-sm text-text-secondary">{error.message}</p>
                      </div>
                    </div>
                  ))}
                </div>
              )}

              {response?.manual_action_required && (
                <div className="flex items-start gap-2 p-3 bg-accent-primary/10 border border-accent-primary/30 rounded">
                  <Info className="h-4 w-4 text-accent-primary mt-0.5 flex-shrink-0" />
                  <p className="text-sm text-accent-primary">
                    This action requires manual intervention. Please check your Yahoo Fantasy team and complete the action manually.
                  </p>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-end gap-3 px-6 py-4 border-t border-border-subtle">
          {step === 'confirm' && (
            <>
              <button
                onClick={handleClose}
                className="px-4 py-2 text-sm font-medium text-text-secondary hover:text-text-primary transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={handleExecute}
                className="px-4 py-2 bg-accent-primary text-white rounded-md text-sm font-medium hover:bg-accent-primary/90 transition-colors"
              >
                Add Player
              </button>
            </>
          )}

          {step === 'error' && (
            <>
              <button
                onClick={handleClose}
                className="px-4 py-2 text-sm font-medium text-text-secondary hover:text-text-primary transition-colors"
              >
                Close
              </button>
              <button
                onClick={handleRetry}
                className="px-4 py-2 bg-accent-primary text-white rounded-md text-sm font-medium hover:bg-accent-primary/90 transition-colors"
              >
                Try Again
              </button>
            </>
          )}

          {step === 'success' && (
            <button
              onClick={handleClose}
              className="px-4 py-2 bg-accent-primary text-white rounded-md text-sm font-medium hover:bg-accent-primary/90 transition-colors"
            >
              Done
            </button>
          )}
        </div>
      </div>
    </div>
  )
}
