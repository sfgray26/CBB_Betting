/**
 * Basic rendering test for StreamingRecommendations component.
 *
 * This test validates that the component renders without crashing
 * and displays the correct structure for 2-start pitcher recommendations.
 */

import { render, screen } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { endpoints } from '@/lib/api'
import { StreamingRecommendations } from './streaming-recommendations'

// Mock the API module
jest.mock('@/lib/api', () => ({
  endpoints: {
    getStreamingRecommendations: jest.fn(),
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
