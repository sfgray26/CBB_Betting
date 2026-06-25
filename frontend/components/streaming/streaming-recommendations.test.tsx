/**
 * Basic rendering test for StreamingRecommendations component.
 *
 * This test validates that the component renders without crashing
 * and displays the correct structure for 2-start pitcher recommendations.
 */

import { render, screen, fireEvent } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { endpoints } from '@/lib/api'
import { StreamingRecommendations } from './streaming-recommendations'
import { ActionModal } from './action-modal'
import type { StreamingPitcher } from '@/lib/types'

// Mock the API module
jest.mock('@/lib/api', () => ({
  endpoints: {
    getStreamingRecommendations: jest.fn(),
    rosterAction: jest.fn(),
  },
}))

// Mock FreshnessBadge component
jest.mock('@/components/freshness/freshness-badge', () => ({
  FreshnessBadge: ({ severity }: { severity: string }) => (
    <div data-testid="freshness-badge">{severity}</div>
  ),
}))

function createWrapper() {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
      },
    },
  })
  function QueryWrapper({ children }: { children: React.ReactNode }) {
    return (
      <QueryClientProvider client={queryClient}>
        {children}
      </QueryClientProvider>
    )
  }
  QueryWrapper.displayName = 'StreamingRecommendationsQueryWrapper'
  return QueryWrapper
}

describe('StreamingRecommendations', () => {
  it('renders loading state', () => {
    jest.mocked(endpoints.getStreamingRecommendations).mockImplementation(
      () => new Promise(() => {})
    )

    render(<StreamingRecommendations targetDate="2026-06-24" />, { wrapper: createWrapper() })

    expect(screen.getByText(/Loading streaming recommendations/i)).toBeInTheDocument()
  })

  it('renders error state on API failure', async () => {
    jest.mocked(endpoints.getStreamingRecommendations).mockRejectedValue(new Error('API error'))

    render(<StreamingRecommendations targetDate="2026-06-24" />, { wrapper: createWrapper() })

    await expect(screen.findByText(/Failed to load recommendations/i)).toBeInTheDocument()
  })

  it('renders 2-start pitchers when data is loaded', async () => {
    jest.mocked(endpoints.getStreamingRecommendations).mockResolvedValue({
      target_date: '2026-06-24',
      analysis_window_days: 7,
      two_start_pitchers: [
        {
          bdl_player_id: 12345,
          name: 'Gerrit Cole',
          team: 'NYY',
          handedness: 'R',
          starts: [
            {
              pitcher_name: 'Gerrit Cole',
              team: 'NYY',
              handedness: 'R',
              date: '2026-06-24',
              opponent: 'BOS',
              is_home: true,
              quality_score: 1.2,
              is_confirmed: true,
              game_time_et: '7:05 PM',
            },
            {
              pitcher_name: 'Gerrit Cole',
              team: 'NYY',
              handedness: 'R',
              date: '2026-06-29',
              opponent: 'BAL',
              is_home: false,
              quality_score: 0.8,
              is_confirmed: true,
              game_time_et: '1:05 PM',
            },
          ],
          overall_quality: 1.0,
          recommendation: 'EXCELLENT',
          risk_note: 'Both starts confirmed — safe stream',
          transparency: {
            quality_score: 1.0,
            factors: ['starts_count: 2', 'avg_quality: 1.00'],
            confidence: 'HIGH',
          },
        },
      ],
      freshness: {
        last_refresh_at: '2026-06-23T12:30:18Z',
        staleness_ms: 0,
        query_time_et: '2026-06-23T08:30:18-04:00',
      },
      data_sources: ['ProbablePitcherSnapshot', 'StatcastPerformances'],
    })

    render(<StreamingRecommendations targetDate="2026-06-24" />, { wrapper: createWrapper() })

    await expect(screen.findByText('Gerrit Cole')).toBeInTheDocument()
    await expect(screen.findByText('NYY')).toBeInTheDocument()
    await expect(screen.findByText(/EXCELLENT/i)).toBeInTheDocument()
    await expect(screen.findByText(/Both starts confirmed/i)).toBeInTheDocument()
  })

  it('renders empty state when no pitchers found', async () => {
    jest.mocked(endpoints.getStreamingRecommendations).mockResolvedValue({
      target_date: '2026-06-24',
      analysis_window_days: 7,
      two_start_pitchers: [],
      freshness: {
        last_refresh_at: '2026-06-23T12:30:18Z',
        staleness_ms: 0,
        query_time_et: '2026-06-23T08:30:18-04:00',
      },
      data_sources: ['ProbablePitcherSnapshot'],
    })

    render(<StreamingRecommendations targetDate="2026-06-24" />, { wrapper: createWrapper() })

    await expect(screen.findByText(/No 2-start pitchers found/i)).toBeInTheDocument()
  })

  it('is forward-compatible with one_start_pitchers field', async () => {
    jest.mocked(endpoints.getStreamingRecommendations).mockResolvedValue({
      target_date: '2026-06-24',
      analysis_window_days: 7,
      two_start_pitchers: [],
      // Forward-compatible field - may be added in future iterations
      one_start_pitchers: [
        {
          bdl_player_id: 67890,
          name: 'One Starter',
          team: 'LAD',
          handedness: 'L',
          starts: [
            {
              pitcher_name: 'One Starter',
              team: 'LAD',
              handedness: 'L',
              date: '2026-06-24',
              opponent: 'SF',
              is_home: true,
              quality_score: 0.5,
              is_confirmed: true,
              game_time_et: '10:10 PM',
            },
          ],
          overall_quality: 0.5,
          recommendation: 'AVERAGE',
          risk_note: 'One start confirmed — monitor for scratches',
          transparency: {
            quality_score: 0.5,
            factors: ['starts_count: 1'],
            confidence: 'HIGH',
          },
        },
      ],
      freshness: {
        last_refresh_at: '2026-06-23T12:30:18Z',
        staleness_ms: 0,
        query_time_et: '2026-06-23T08:30:18-04:00',
      },
      data_sources: ['ProbablePitcherSnapshot'],
    })

    render(<StreamingRecommendations targetDate="2026-06-24" />, { wrapper: createWrapper() })

    // Should not crash and should handle the response correctly
    // (one_start_pitchers is ignored for now, but component should be forward-compatible)
    await expect(screen.findByText(/No 2-start pitchers found/i)).toBeInTheDocument()
  })
})

// ═════════════════════════════════════════════════════════════════════════════
// Loop Iteration 12 Tests: Action Modal and Button States
// ═════════════════════════════════════════════════════════════════════════════

describe('ActionModal (Loop Iteration 12)', () => {
  const createMockPitcher = (overrides: Partial<StreamingPitcher> = {}): StreamingPitcher => ({
    bdl_player_id: 12345,
    name: 'Test Pitcher',
    team: 'NYY',
    handedness: 'RHP',
    starts: [
      {
        pitcher_name: 'Test Pitcher',
        team: 'NYY',
        handedness: 'RHP',
        date: '2026-06-25',
        opponent: 'BOS',
        is_home: true,
        quality_score: 1.2,
        is_confirmed: true,
        game_time_et: '7:05 PM ET',
      },
    ],
    overall_quality: 1.0,
    recommendation: 'GOOD',
    risk_note: 'Solid matchup',
    transparency: {
      quality_score: 1.0,
      factors: ['Favorable matchup'],
      confidence: 'HIGH',
    },
    ...overrides,
  })

  it('renders modal when open', () => {
    const pitcher = createMockPitcher()
    const handleClose = jest.fn()
    const handleSuccess = jest.fn()

    render(
      <QueryClientProvider client={new QueryClient()}>
        <ActionModal
          pitcher={pitcher}
          isOpen={true}
          onClose={handleClose}
          onSuccess={handleSuccess}
        />
      </QueryClientProvider>
    )

    expect(screen.getByText('Confirm Add')).toBeInTheDocument()
    expect(screen.getByText(pitcher.name)).toBeInTheDocument()
  })

  it('does not render when closed', () => {
    const pitcher = createMockPitcher()
    const handleClose = jest.fn()
    const handleSuccess = jest.fn()

    render(
      <QueryClientProvider client={new QueryClient()}>
        <ActionModal
          pitcher={pitcher}
          isOpen={false}
          onClose={handleClose}
          onSuccess={handleSuccess}
        />
      </QueryClientProvider>
    )

    expect(screen.queryByText('Confirm Add')).not.toBeInTheDocument()
  })

  it('calls onClose when cancel button clicked', () => {
    const pitcher = createMockPitcher()
    const handleClose = jest.fn()
    const handleSuccess = jest.fn()

    render(
      <QueryClientProvider client={new QueryClient()}>
        <ActionModal
          pitcher={pitcher}
          isOpen={true}
          onClose={handleClose}
          onSuccess={handleSuccess}
        />
      </QueryClientProvider>
    )

    fireEvent.click(screen.getByText('Cancel'))
    expect(handleClose).toHaveBeenCalled()
  })

  it('displays pitcher information correctly', () => {
    const pitcher = createMockPitcher()
    const handleClose = jest.fn()
    const handleSuccess = jest.fn()

    render(
      <QueryClientProvider client={new QueryClient()}>
        <ActionModal
          pitcher={pitcher}
          isOpen={true}
          onClose={handleClose}
          onSuccess={handleSuccess}
        />
      </QueryClientProvider>
    )

    expect(screen.getByText(pitcher.name)).toBeInTheDocument()
    expect(screen.getByText(pitcher.team)).toBeInTheDocument()
    expect(screen.getByText('1.0')).toBeInTheDocument() // quality score
  })

  it('shows waiver claim warning for high-quality pitchers', () => {
    const pitcher = createMockPitcher({ overall_quality: 1.5 })
    const handleClose = jest.fn()
    const handleSuccess = jest.fn()

    render(
      <QueryClientProvider client={new QueryClient()}>
        <ActionModal
          pitcher={pitcher}
          isOpen={true}
          onClose={handleClose}
          onSuccess={handleSuccess}
        />
      </QueryClientProvider>
    )

    expect(screen.getByText(/waiver claim/i)).toBeInTheDocument()
  })

  it('shows medium confidence warning for MEDIUM confidence pitchers', () => {
    const pitcher = createMockPitcher({
      transparency: {
        quality_score: 0.8,
        factors: [],
        confidence: 'MEDIUM',
      },
    })
    const handleClose = jest.fn()
    const handleSuccess = jest.fn()

    render(
      <QueryClientProvider client={new QueryClient()}>
        <ActionModal
          pitcher={pitcher}
          isOpen={true}
          onClose={handleClose}
          onSuccess={handleSuccess}
        />
      </QueryClientProvider>
    )

    expect(screen.getByText(/Medium confidence/i)).toBeInTheDocument()
  })
})

describe('StreamingRecommendations - Execute Add Button States', () => {
  it('renders Execute Add button for valid recommendations', async () => {
    jest.mocked(endpoints.getStreamingRecommendations).mockResolvedValue({
      target_date: '2026-06-24',
      analysis_window_days: 7,
      two_start_pitchers: [
        {
          bdl_player_id: 12345,
          name: 'Gerrit Cole',
          team: 'NYY',
          handedness: 'R',
          starts: [
            {
              pitcher_name: 'Gerrit Cole',
              team: 'NYY',
              handedness: 'R',
              date: '2026-06-24',
              opponent: 'BOS',
              is_home: true,
              quality_score: 1.2,
              is_confirmed: true,
              game_time_et: '7:05 PM',
            },
            {
              pitcher_name: 'Gerrit Cole',
              team: 'NYY',
              handedness: 'R',
              date: '2026-06-29',
              opponent: 'BAL',
              is_home: false,
              quality_score: 0.8,
              is_confirmed: true,
              game_time_et: '1:05 PM',
            },
          ],
          overall_quality: 1.0,
          recommendation: 'EXCELLENT',
          risk_note: 'Both starts confirmed',
          transparency: {
            quality_score: 1.0,
            factors: ['starts_count: 2'],
            confidence: 'HIGH',
          },
        },
      ],
      freshness: {
        last_refresh_at: '2026-06-23T12:30:18Z',
        staleness_ms: 0,
        query_time_et: '2026-06-23T08:30:18-04:00',
      },
      data_sources: ['ProbablePitcherSnapshot'],
    })

    render(<StreamingRecommendations targetDate="2026-06-24" />, { wrapper: createWrapper() })

    await screen.findByText('Gerrit Cole')
    expect(screen.getByText('Execute Add')).toBeInTheDocument()
  })

  it('disables button for AVOID recommendations', async () => {
    jest.mocked(endpoints.getStreamingRecommendations).mockResolvedValue({
      target_date: '2026-06-24',
      analysis_window_days: 7,
      two_start_pitchers: [
        {
          bdl_player_id: 99999,
          name: 'Bad Pitcher',
          team: 'BAD',
          handedness: 'L',
          starts: [
            {
              pitcher_name: 'Bad Pitcher',
              team: 'BAD',
              handedness: 'L',
              date: '2026-06-24',
              opponent: 'GOOD',
              is_home: false,
              quality_score: -1.5,
              is_confirmed: true,
              game_time_et: '7:05 PM',
            },
            {
              pitcher_name: 'Bad Pitcher',
              team: 'BAD',
              handedness: 'L',
              date: '2026-06-29',
              opponent: 'GOOD',
              is_home: true,
              quality_score: -1.0,
              is_confirmed: false,
              game_time_et: '1:05 PM',
            },
          ],
          overall_quality: -1.5,
          recommendation: 'AVOID',
          risk_note: 'Poor matchups - avoid',
          transparency: {
            quality_score: -1.5,
            factors: ['Unfavorable matchups'],
            confidence: 'HIGH',
          },
        },
      ],
      freshness: {
        last_refresh_at: '2026-06-23T12:30:18Z',
        staleness_ms: 0,
        query_time_et: '2026-06-23T08:30:18-04:00',
      },
      data_sources: ['ProbablePitcherSnapshot'],
    })

    render(<StreamingRecommendations targetDate="2026-06-24" />, { wrapper: createWrapper() })

    await screen.findByText('Bad Pitcher')
    const addButton = screen.getAllByText('Execute Add')[0]
    expect(addButton).toBeDisabled()
  })

  it('disables button for LOW confidence recommendations', async () => {
    jest.mocked(endpoints.getStreamingRecommendations).mockResolvedValue({
      target_date: '2026-06-24',
      analysis_window_days: 7,
      two_start_pitchers: [
        {
          bdl_player_id: 88888,
          name: 'Risky Pitcher',
          team: 'RISK',
          handedness: 'R',
          starts: [
            {
              pitcher_name: 'Risky Pitcher',
              team: 'RISK',
              handedness: 'R',
              date: '2026-06-24',
              opponent: 'UNK',
              is_home: true,
              quality_score: 0.5,
              is_confirmed: false,
              game_time_et: 'TBA',
            },
            {
              pitcher_name: 'Risky Pitcher',
              team: 'RISK',
              handedness: 'R',
              date: '2026-06-29',
              opponent: 'UNK',
              is_home: false,
              quality_score: 0.5,
              is_confirmed: false,
              game_time_et: 'TBA',
            },
          ],
          overall_quality: 0.5,
          recommendation: 'AVERAGE',
          risk_note: 'Unconfirmed starts - risky',
          transparency: {
            quality_score: 0.5,
            factors: ['Unconfirmed starts'],
            confidence: 'LOW',
          },
        },
      ],
      freshness: {
        last_refresh_at: '2026-06-23T12:30:18Z',
        staleness_ms: 0,
        query_time_et: '2026-06-23T08:30:18-04:00',
      },
      data_sources: ['ProbablePitcherSnapshot'],
    })

    render(<StreamingRecommendations targetDate="2026-06-24" />, { wrapper: createWrapper() })

    await screen.findByText('Risky Pitcher')
    const addButton = screen.getAllByText('Execute Add')[0]
    expect(addButton).toBeDisabled()
  })
})
