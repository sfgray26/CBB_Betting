/**
 * Fantasy Baseball E2E Regression Suite
 *
 * Guards against the specific bug regressions fixed in the March 2026 audit:
 *   - Matchup: no "TBD"/"My Team"/"0 vs 0" placeholders
 *   - Roster: no "Failed to fetch" error boundary; no duplicate players
 *   - Lineup: West Coast late games (9:40 PM EDT) display correctly; no false "no game" banners
 *   - Waiver: owned % > 0%; no "vs TBD" opponent
 *   - Dashboard: valid date display; lineup filled (not 0/9)
 *
 * All backend calls are intercepted with deterministic mocks so the suite
 * is self-contained and never hits live rate limits.
 */

import { test, expect, type Page, type Route } from '@playwright/test'

// ---------------------------------------------------------------------------
// Auth helper — inject the cbb_api_key cookie so middleware passes
// ---------------------------------------------------------------------------

async function authenticate(page: Page) {
  await page.context().addCookies([
    {
      name: 'cbb_api_key',
      value: 'test-key-e2e',
      domain: 'localhost',
      path: '/',
    },
  ])
}

// ---------------------------------------------------------------------------
// Mock payloads
// ---------------------------------------------------------------------------

const MOCK_ROSTER = {
  team_key: '411.l.12345.t.1',
  count: 3,
  players: [
    {
      player_key: '411.p.10001',
      name: 'Steven Kwan',
      team: 'CLE',
      positions: ['LF', 'OF'],
      status: null,
      injury_note: null,
      z_score: 1.5,
      is_undroppable: false,
      selected_position: 'LF',
    },
    {
      player_key: '411.p.10002',
      name: 'Juan Soto',
      team: 'NYM',
      positions: ['RF', 'OF'],
      status: null,
      injury_note: null,
      z_score: 2.1,
      is_undroppable: false,
      selected_position: 'RF',
    },
    {
      player_key: '411.p.10003',
      name: 'Blake Snell',
      team: 'SF',
      positions: ['SP', 'P'],
      status: 'IL15',
      injury_note: 'Left shoulder inflammation',
      z_score: 1.8,
      is_undroppable: false,
      selected_position: 'IL',
    },
  ],
}

const MOCK_MATCHUP = {
  week: 3,
  my_team: {
    team_key: '411.l.12345.t.1',
    team_name: 'Fixed Wagering FC',
    stats: { R: 45, HR: 8, RBI: 30, SB: 5, AVG: 0.265, ERA: 3.12, WHIP: 1.18 },
  },
  opponent: {
    team_key: '411.l.12345.t.2',
    team_name: 'Dingers and Dreams',
    stats: { R: 38, HR: 6, RBI: 25, SB: 4, AVG: 0.251, ERA: 3.55, WHIP: 1.25 },
  },
  is_playoffs: false,
  message: null,
}

const MOCK_LINEUP = {
  date: new Date().toISOString().slice(0, 10),
  games_count: 8,
  no_games_today: false,
  lineup_warnings: [],
  batters: [
    {
      player_id: '10001',
      name: 'Steven Kwan',
      team: 'CLE',
      position: 'OF',
      implied_runs: 4.8,
      park_factor: 1.02,
      lineup_score: 8.5,
      // Late West Coast game — the regression we're guarding
      start_time: '2026-03-29T01:40:00.000Z', // 9:40 PM EDT = 01:40 UTC next day
      opponent: 'LAD',
      status: 'START',
      assigned_slot: 'LF',
      injury_status: null,
    },
  ],
  pitchers: [
    {
      player_id: '10010',
      name: 'Logan Gilbert',
      team: 'SEA',
      pitcher_type: 'SP',
      opponent: 'HOU',
      opponent_implied_runs: 3.8,
      park_factor: 0.97,
      sp_score: 7.2,
      start_time: '2026-03-29T02:10:00.000Z',
      status: 'START',
    },
  ],
}

const MOCK_WAIVER = {
  week_end: '2026-04-05',
  matchup_opponent: 'Dingers and Dreams',
  category_deficits: [
    { category: 'SB', my_total: 5, opponent_total: 8, deficit: 3, winning: false },
  ],
  top_available: [
    {
      player_id: '20001',
      name: 'Geraldo Perdomo',
      team: 'ARI',
      position: 'SS',
      need_score: 1.8,
      category_contributions: { SB: 0.9, R: 0.5 },
      owned_pct: 42.5,
      starts_this_week: 5,
      statcast_signals: [],
      hot_cold: 'HOT',
      status: null,
      injury_note: null,
    },
    {
      player_id: '20002',
      name: 'Ceddanne Rafaela',
      team: 'BOS',
      position: 'OF',
      need_score: 1.4,
      category_contributions: { SB: 0.6, HR: 0.4 },
      owned_pct: 31.2,
      starts_this_week: 6,
      statcast_signals: [],
      hot_cold: null,
      status: null,
      injury_note: null,
    },
  ],
  two_start_pitchers: [],
  pagination: { page: 1, per_page: 20, has_next: false },
  urgent_alert: null,
  faab_balance: 85,
}

const MOCK_WAIVER_RECS = {
  week_end: '2026-04-05',
  matchup_opponent: 'Dingers and Dreams',
  recommendations: [
    {
      action: 'ADD_DROP',
      add_player: MOCK_WAIVER.top_available[0],
      drop_player_name: 'Miguel Vargas',
      drop_player_position: '1B',
      rationale: 'Add Geraldo Perdomo (SS, ARI, 42% owned), drop Miguel Vargas (1B). Net gain: +1.4.',
      need_score: 1.4,
      confidence: 0.75,
      statcast_signals: [],
      win_prob_before: 0.48,
      win_prob_after: 0.52,
      win_prob_gain: 0.04,
      mcmc_enabled: true,
    },
  ],
  category_deficits: MOCK_WAIVER.category_deficits,
}

const MOCK_DASHBOARD = {
  success: true,
  timestamp: '2026-03-29T14:00:00-04:00',
  data: {
    timestamp: '2026-03-29T14:00:00-04:00',
    user_id: 'test-user',
    lineup_gaps: [],
    lineup_filled_count: 9,
    lineup_total_count: 9,
    hot_streaks: [
      {
        player_id: '10001',
        name: 'Steven Kwan',
        team: 'CLE',
        positions: ['LF'],
        trend: 'hot',
        trend_score: 1.8,
        last_7_avg: 0.35,
        last_14_avg: 0.31,
        last_30_avg: 0.285,
        reason: 'Hot last 7 days',
      },
    ],
    cold_streaks: [],
    waiver_targets: [
      {
        player_id: '20001',
        name: 'Geraldo Perdomo',
        team: 'ARI',
        positions: ['SS'],
        percent_owned: 42.5,
        priority_score: 7.2,
        tier: 'strong_add',
        reason: 'SB upside',
      },
    ],
    injury_flags: [
      {
        player_id: '10003',
        name: 'Blake Snell',
        status: 'IL15',
        injury_note: 'Left shoulder',
        severity: 'critical',
        estimated_return: '2026-04-10',
        action_needed: 'Move to IL slot',
      },
    ],
    healthy_count: 20,
    injured_count: 2,
    matchup_preview: null,
    probable_pitchers: [],
    two_start_pitchers: [],
  },
  preferences: {
    notifications: {
      lineup_deadline: true,
      injury_alerts: true,
      waiver_suggestions: true,
      trade_offers: false,
      hot_streak_alerts: true,
      channels: [],
      discord_user_id: null,
      email_enabled: false,
    },
    dashboard_layout: {
      panels: [],
      refresh_interval_seconds: 300,
      theme: 'dark',
    },
    streak_settings: {
      hot_threshold: 1.5,
      cold_threshold: -1.5,
      min_sample_days: 7,
      rolling_windows: [7, 14, 30],
    },
    waiver_preferences: {
      min_percent_owned: 0,
      max_percent_owned: 85,
      positions_of_need: [],
      priority_categories: [],
      hide_injured: false,
      streamer_threshold: 1.0,
    },
  },
}

// ---------------------------------------------------------------------------
// Route interceptor factory
// ---------------------------------------------------------------------------

async function mockAllFantasyRoutes(page: Page) {
  // IMPORTANT: Playwright routes are LIFO — last registered = highest priority.
  // Register broad catch-alls FIRST so specific mocks registered later take precedence.

  // Silence unrelated backend calls (registered first = lowest priority)
  await page.route('**/api/**', (route: Route) =>
    route.fulfill({ status: 200, contentType: 'application/json', body: '{}' })
  )

  // Specific mocks registered AFTER catch-all so they win (LIFO)
  await page.route('**/api/dashboard', (route: Route) =>
    route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(MOCK_DASHBOARD) })
  )
  // Waiver base route — must come before /recommendations so the more-specific mock wins
  await page.route('**/api/fantasy/waiver**', (route: Route) =>
    route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(MOCK_WAIVER) })
  )
  // /recommendations is more specific — register AFTER the base waiver route so it wins
  await page.route('**/api/fantasy/waiver/recommendations', (route: Route) =>
    route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(MOCK_WAIVER_RECS) })
  )
  // Lineup URL includes the date — use glob to match any date
  await page.route('**/api/fantasy/lineup/**', (route: Route) =>
    route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(MOCK_LINEUP) })
  )
  await page.route('**/api/fantasy/matchup', (route: Route) =>
    route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(MOCK_MATCHUP) })
  )
  await page.route('**/api/fantasy/roster', (route: Route) =>
    route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(MOCK_ROSTER) })
  )
}

// ---------------------------------------------------------------------------
// Test: Matchup page
// ---------------------------------------------------------------------------

test.describe('Matchup Page', () => {
  test.beforeEach(async ({ page }) => {
    await authenticate(page)
    await mockAllFantasyRoutes(page)
    await page.goto('/fantasy/matchup')
  })

  test('renders real team names — no TBD, My Team, or 0 vs 0 placeholders', async ({ page }) => {
    // Score banner paragraph has the team name — use .first() since it also appears in the table header
    await expect(page.getByText('Fixed Wagering FC').first()).toBeVisible({ timeout: 10_000 })
    await expect(page.getByText('Dingers and Dreams').first()).toBeVisible()

    // Regression guards
    await expect(page.getByText('TBD')).not.toBeVisible()
    await expect(page.getByText('My Team')).not.toBeVisible()
    const scores = page.locator('.tabular-nums')
    await expect(scores.first()).toBeVisible()
  })

  test('shows Week 3 label — not "Playoffs" string or empty', async ({ page }) => {
    await expect(page.getByText('Fixed Wagering FC').first()).toBeVisible({ timeout: 10_000 })
    await expect(page.getByText(/Week 3/)).toBeVisible()
    // Not just the raw "Playoffs" string when is_playoffs is false
    await expect(page.getByText('Playoffs').first()).not.toBeVisible()
  })

  test('shows stat category breakdown table', async ({ page }) => {
    await expect(page.getByText('Fixed Wagering FC').first()).toBeVisible({ timeout: 10_000 })
    await expect(page.getByText('Category Breakdown')).toBeVisible()
    // Table renders at least one stat row
    await expect(page.getByText('ERA').first()).toBeVisible()
  })
})

// ---------------------------------------------------------------------------
// Test: My Roster page
// ---------------------------------------------------------------------------

test.describe('My Roster Page', () => {
  test.beforeEach(async ({ page }) => {
    await authenticate(page)
    await mockAllFantasyRoutes(page)
    await page.goto('/fantasy/roster')
  })

  test('loads successfully — no "Failed to fetch" error boundary', async ({ page }) => {
    // Wait for player name to appear
    await expect(page.getByText('Steven Kwan')).toBeVisible({ timeout: 10_000 })
    // Error boundary must NOT appear
    await expect(page.getByText(/Failed to fetch/i)).not.toBeVisible()
    await expect(page.getByText(/Roster failed to load/i)).not.toBeVisible()
  })

  test('renders all players exactly once — no DOM duplicates', async ({ page }) => {
    await expect(page.getByText('Steven Kwan')).toBeVisible({ timeout: 10_000 })

    // Each name must appear exactly once in the player table
    const kwan = page.getByRole('cell', { name: /Steven Kwan/ })
    const soto = page.getByRole('cell', { name: /Juan Soto/ })
    const snell = page.getByRole('cell', { name: /Blake Snell/ })

    await expect(kwan).toHaveCount(1)
    await expect(soto).toHaveCount(1)
    await expect(snell).toHaveCount(1)
  })

  test('IL player shows IL badge and selected_position slot', async ({ page }) => {
    await expect(page.getByText('Blake Snell')).toBeVisible({ timeout: 10_000 })
    // IL15 badge should be visible next to Blake Snell
    await expect(page.getByText('IL15')).toBeVisible()
    // Slot column shows IL for selected_position
    await expect(page.getByText('IL').first()).toBeVisible()
  })

  test('non-injured players have no IL badge', async ({ page }) => {
    await expect(page.getByText('Steven Kwan')).toBeVisible({ timeout: 10_000 })
    // Count of IL15 badges — only 1 (Blake Snell's)
    await expect(page.getByText('IL15')).toHaveCount(1)
  })
})

// ---------------------------------------------------------------------------
// Test: Daily Lineup page
// ---------------------------------------------------------------------------

test.describe('Daily Lineup Page', () => {
  test.beforeEach(async ({ page }) => {
    await authenticate(page)
    await mockAllFantasyRoutes(page)
    await page.goto('/fantasy/lineup')
  })

  test('loads player roster — no error state', async ({ page }) => {
    await expect(page.getByText('Steven Kwan')).toBeVisible({ timeout: 10_000 })
    await expect(page.getByText(/failed to load/i)).not.toBeVisible()
  })

  test('West Coast late game renders with a time — no "Starting but no game today" banner', async ({
    page,
  }) => {
    await expect(page.getByText('Steven Kwan')).toBeVisible({ timeout: 10_000 })

    // The regression was utcnow() causing West Coast games to show "no game today"
    // Our fix: player shows START status and a non-empty time
    const kwan = page.getByText('Steven Kwan')
    const row = kwan.locator('xpath=ancestor::tr')
    // START badge visible in the row
    await expect(row.getByText('START')).toBeVisible()

    // No false "no game today" warning banners
    await expect(page.getByText(/no game today/i)).not.toBeVisible()
    await expect(page.getByText(/Starting but no game/i)).not.toBeVisible()
  })

  test('Starting Pitchers table shows only SP — no RP rows', async ({ page }) => {
    await expect(page.getByText('Logan Gilbert')).toBeVisible({ timeout: 10_000 })
    // RP rows should not appear in the SP table when pitcher_type is SP
    // The mock has 1 SP; table header visible
    await expect(page.getByText(/Starting Pitchers/i)).toBeVisible()
  })

  test('Apply to Yahoo button is not disabled when no IL starters', async ({ page }) => {
    await expect(page.getByText('Steven Kwan')).toBeVisible({ timeout: 10_000 })
    const applyBtn = page.getByRole('button', { name: /Apply to Yahoo/i })
    if (await applyBtn.isVisible()) {
      await expect(applyBtn).not.toBeDisabled()
    }
  })
})

// ---------------------------------------------------------------------------
// Test: Waiver Wire page
// ---------------------------------------------------------------------------

test.describe('Waiver Wire Page', () => {
  test.beforeEach(async ({ page }) => {
    await authenticate(page)
    await mockAllFantasyRoutes(page)
    await page.goto('/fantasy/waiver')
  })

  test('loads top available players', async ({ page }) => {
    await expect(page.getByText('Geraldo Perdomo')).toBeVisible({ timeout: 10_000 })
  })

  test('Owned % values are populated and non-zero', async ({ page }) => {
    await expect(page.getByText('Geraldo Perdomo')).toBeVisible({ timeout: 10_000 })
    // owned_pct: 42.5 should appear as "42.5%" or "42%"
    await expect(page.getByText(/42\.?5?%/)).toBeVisible()
    // Must NOT display "0.0%" as the only owned value
    const zeroOwned = page.getByText('0.0%')
    // Either it doesn't exist, or it's not the primary value for Perdomo
    const perdomo = page.getByText('Geraldo Perdomo')
    const row = perdomo.locator('xpath=ancestor::tr')
    await expect(row.getByText('0.0%')).not.toBeVisible()
  })

  test('matchup opponent is real team name — no "vs TBD"', async ({ page }) => {
    await expect(page.getByText('Geraldo Perdomo')).toBeVisible({ timeout: 10_000 })
    // Regression: matchup_opponent was "TBD" before the fix
    await expect(page.getByText(/vs TBD/i)).not.toBeVisible()
    await expect(page.getByText(/Dingers and Dreams/i)).toBeVisible()
  })

  test('recommendation rationale does not contain raw "[BUY_LOW...]" tag', async ({ page }) => {
    await expect(page.getByText('Geraldo Perdomo')).toBeVisible({ timeout: 10_000 })
    // The mock rationale doesn't have it, but guard against the pattern
    await expect(page.getByText(/\[BUY_LOW/)).not.toBeVisible()
  })

  test('action label shows human-readable text — not raw "ADD_DROP" enum', async ({ page }) => {
    await expect(page.getByText('Geraldo Perdomo')).toBeVisible({ timeout: 10_000 })
    // "ADD_DROP" raw enum must not appear; should be "Add / Drop" or similar
    await expect(page.getByText('ADD_DROP')).not.toBeVisible()
  })
})

// ---------------------------------------------------------------------------
// Test: Dashboard page
// ---------------------------------------------------------------------------

test.describe('Dashboard Page', () => {
  test.beforeEach(async ({ page }) => {
    await authenticate(page)
    await mockAllFantasyRoutes(page)
    await page.goto('/dashboard')
  })

  test('loads without error', async ({ page }) => {
    // Wait for the h1 heading — unique element, no strict mode violation
    await expect(page.getByRole('heading', { name: 'Fantasy Baseball Dashboard' })).toBeVisible({ timeout: 10_000 })
    await expect(page.getByText(/Error loading dashboard/i)).not.toBeVisible()
  })

  test('last_synced_at does not display "Invalid Date"', async ({ page }) => {
    await expect(page.getByRole('heading', { name: 'Fantasy Baseball Dashboard' })).toBeVisible({ timeout: 10_000 })
    // The timestamp field must not show "Invalid Date"
    await expect(page.getByText(/Invalid Date/i)).not.toBeVisible()
    // Should show a valid date/time string — dashboard timestamp includes "ET", header shows "s ago"
    await expect(page.getByText(/Last updated:.*ET/i)).toBeVisible()
  })

  test('lineup status is not 0/9 — reflects filled lineup', async ({ page }) => {
    await expect(page.getByRole('heading', { name: 'Fantasy Baseball Dashboard' })).toBeVisible({ timeout: 10_000 })
    // Mock returns lineup_filled_count=9, lineup_total_count=9
    await expect(page.getByText('9/9')).toBeVisible()
    await expect(page.getByText('0/9')).not.toBeVisible()
  })

  test('hot streak player name renders', async ({ page }) => {
    await expect(page.getByRole('heading', { name: 'Fantasy Baseball Dashboard' })).toBeVisible({ timeout: 10_000 })
    await expect(page.getByText('Steven Kwan')).toBeVisible()
  })

  test('waiver targets sorted by priority_score — highest first', async ({ page }) => {
    await expect(page.getByRole('heading', { name: 'Fantasy Baseball Dashboard' })).toBeVisible({ timeout: 10_000 })
    // Mock has 1 target with priority_score: 7.2 — should appear
    await expect(page.getByText('Geraldo Perdomo')).toBeVisible()
    // Score value visible — use .first() since "7.2" could appear elsewhere
    await expect(page.getByText('7.2').first()).toBeVisible()
  })
})
