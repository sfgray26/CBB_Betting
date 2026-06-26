import { QueryClient } from '@tanstack/react-query'

/**
 * Unified React Query configuration for all fantasy modules.
 *
 * Strategy:
 * - 5-minute stale-while-revalidate for in-season data
 * - Refetch on window focus (user returns to tab)
 * - Graceful retry with 1 attempt (avoid hammering backend on errors)
 * - 30-second cache time to balance freshness with performance
 */
export const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      // 5 minutes: data is fresh enough to display without revalidating
      staleTime: 5 * 60_000,
      // Refetch on window focus (user returns to tab)
      refetchOnWindowFocus: true,
      // Single retry for transient errors (network blips, 500s)
      retry: 1,
      // Keep garbage collection short to avoid memory buildup
      gcTime: 30 * 60_000,
    },
  },
})

/**
 * Shared query keys for all fantasy modules.
 * Used for targeted cache invalidation (e.g., global refresh button).
 */
export const FANTASY_QUERY_KEYS = {
  // Core data
  matchup: ['matchup'] as const,
  scoreboard: ['scoreboard'] as const,
  roster: ['roster'] as const,
  budget: ['budget'] as const,
  streaks: ['dashboard-streaks'] as const,
  globalFreshness: ['global-freshness'] as const,

  // Recommendations
  waiverRecommendations: ['waiver-recommendations'] as const,
  streamingRecommendations: ['streaming-recommendations'] as const,

  // Projections
  projectionStatus: ['projection-status'] as const,

  // All fantasy keys (for bulk invalidation)
  all: [...[
    'matchup',
    'scoreboard',
    'roster',
    'budget',
    'dashboard-streaks',
    'global-freshness',
    'waiver-recommendations',
    'streaming-recommendations',
    'projection-status',
  ]] as const,
} as const

/**
 * Invalidate all fantasy module caches.
 * Called by the global "Refresh Data" button.
 */
export function invalidateAllFantasyCaches(): void {
  // Invalidate each query key individually
  queryClient.invalidateQueries({ queryKey: FANTASY_QUERY_KEYS.matchup })
  queryClient.invalidateQueries({ queryKey: FANTASY_QUERY_KEYS.scoreboard })
  queryClient.invalidateQueries({ queryKey: FANTASY_QUERY_KEYS.roster })
  queryClient.invalidateQueries({ queryKey: FANTASY_QUERY_KEYS.budget })
  queryClient.invalidateQueries({ queryKey: FANTASY_QUERY_KEYS.streaks })
  queryClient.invalidateQueries({ queryKey: FANTASY_QUERY_KEYS.globalFreshness })
  queryClient.invalidateQueries({ queryKey: FANTASY_QUERY_KEYS.waiverRecommendations })
  queryClient.invalidateQueries({ queryKey: FANTASY_QUERY_KEYS.streamingRecommendations })
  queryClient.invalidateQueries({ queryKey: FANTASY_QUERY_KEYS.projectionStatus })
}
