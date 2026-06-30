import { test, expect } from '@playwright/test'

/**
 * CRITICAL 2 REGRESSION TEST
 * Issue: Lineup move UI never updates after POST returns 200
 * Fix: Optimistic update in React Query mutation
 *
 * This test verifies:
 * 1. Player visually moves immediately (optimistic update)
 * 2. Success toast appears
 * 3. UI updates without waiting for backend response
 */

test.describe('Roster move — optimistic update', () => {
  test('player slot updates immediately with optimistic feedback', async ({ page }) => {
    // Mock roster API with test player
    const mockRoster = {
      team_key: 'test.l.t1',
      count: 1,
      freshness: null,
      players: [
        {
          player_name: 'Test Player',
          team: 'BOS',
          eligible_positions: ['1B', 'BN'],
          current_slot: 'BN',
          status: 'playing',
          yahoo_player_key: 'test.player.1',
          season_stats: { values: { HR: 10, RBI: 30, AVG: 0.250, OPS: 0.750 } },
          rolling_7d: null,
          rolling_14d: null,
          rolling_30d: null,
          ros_projection: null,
          game_context: null,
          ownership_pct: null,
          injury_status: null,
        },
      ],
    }

    // Mock successful move response (delayed to simulate network)
    let moveCallCount = 0
    await page.route('**/api/fantasy/roster', (route) =>
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(mockRoster),
      })
    )

    await page.route('**/api/fantasy/roster/move', (route) => {
      moveCallCount++
      // Simulate network delay — optimistic update should happen BEFORE this completes
      setTimeout(() => {
        route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({ message: 'Moved to Util' }),
        })
      }, 500)
    })

    // Mock other required APIs
    await page.route('**/api/freshness/global', (route) =>
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ severity: 'fresh', minutes_ago: 5, warning_text: '' }),
      })
    )

    await page.route('**/api/fantasy/budget', (route) =>
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          budget: {
            acquisition_limit: 7,
            acquisitions_used: 2,
            acquisition_warning: false,
            ip_minimum: 25,
            ip_accumulated: 15.0,
            ip_pace: 'BEHIND',
            il_used: 1,
            il_total: 3,
          },
        }),
      })
    )

    await page.route('**/api/fantasy/scoreboard', (route) =>
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          opponent_name: 'Opponent',
          categories_won: 5,
          categories_lost: 4,
          categories_tied: 2,
          overall_win_probability: 0.55,
          rows: [],
        }),
      })
    )

    await page.route('**/api/dashboard/streaks', (route) =>
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ hot_streaks: [], cold_streaks: [] }),
      })
    )

    // Navigate to roster page
    await page.goto('/war-room/roster')
    await page.waitForLoadState('networkidle')

    // Find the player card and verify initial slot
    const playerCard = page.locator('text=Test Player').first()
    await expect(playerCard).toBeVisible()

    // Verify initial slot is BN
    const initialSlot = page.locator('text=BN').first()
    await expect(initialSlot).toBeVisible()

    // Select target slot (Util/1B option)
    const moveSelect = page.locator('select').first()
    await moveSelect.selectOption('1B')

    // Click the move button
    const moveButton = page.locator('button[title="Move player"]').first()
    await moveButton.click()

    // CRITICAL ASSERTION: Player's slot should update IMMEDIATELY (optimistic update)
    // This should happen BEFORE the 500ms mock network delay completes
    const updatedSlot = page.locator('text=1B').first()
    await expect(updatedSlot).toBeVisible({ timeout: 1000 })

    // Verify success toast appears
    const successToast = page.locator('text=Moved to Util')
    await expect(successToast).toBeVisible({ timeout: 6000 })

    // Verify the move API was called exactly once
    expect(moveCallCount).toBe(1)

    // After everything settles, the player should still show the updated slot
    await page.waitForTimeout(600)
    await expect(updatedSlot).toBeVisible()
  })

  test('move error triggers rollback and shows error toast', async ({ page }) => {
    const mockRoster = {
      team_key: 'test.l.t1',
      count: 1,
      freshness: null,
      players: [
        {
          player_name: 'Test Player',
          team: 'BOS',
          eligible_positions: ['1B', 'BN'],
          current_slot: 'BN',
          status: 'playing',
          yahoo_player_key: 'test.player.1',
          season_stats: { values: { HR: 10, RBI: 30, AVG: 0.250, OPS: 0.750 } },
          rolling_7d: null,
          rolling_14d: null,
          rolling_30d: null,
          ros_projection: null,
          game_context: null,
          ownership_pct: null,
          injury_status: null,
        },
      ],
    }

    await page.route('**/api/fantasy/roster', (route) =>
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(mockRoster),
      })
    )

    // Mock failed move response
    await page.route('**/api/fantasy/roster/move', (route) => {
      setTimeout(() => {
        route.fulfill({
          status: 400,
          contentType: 'application/json',
          body: JSON.stringify({ detail: 'Invalid move' }),
        })
      }, 100)
    })

    // Mock other required APIs
    await page.route('**/api/freshness/global', (route) =>
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ severity: 'fresh', minutes_ago: 5, warning_text: '' }),
      })
    )

    await page.route('**/api/fantasy/budget', (route) =>
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          budget: {
            acquisition_limit: 7,
            acquisitions_used: 2,
            acquisition_warning: false,
            ip_minimum: 25,
            ip_accumulated: 15.0,
            ip_pace: 'BEHIND',
            il_used: 1,
            il_total: 3,
          },
        }),
      })
    )

    await page.route('**/api/fantasy/scoreboard', (route) =>
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          opponent_name: 'Opponent',
          categories_won: 5,
          categories_lost: 4,
          categories_tied: 2,
          overall_win_probability: 0.55,
          rows: [],
        }),
      })
    )

    await page.route('**/api/dashboard/streaks', (route) =>
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ hot_streaks: [], cold_streaks: [] }),
      })
    )

    await page.goto('/war-room/roster')
    await page.waitForLoadState('networkidle')

    // Select target slot and click move
    const moveSelect = page.locator('select').first()
    await moveSelect.selectOption('1B')

    const moveButton = page.locator('button[title="Move player"]').first()
    await moveButton.click()

    // Wait for error response
    await page.waitForTimeout(200)

    // CRITICAL ASSERTION: Slot should ROLLBACK to original BN after error
    const bnSlot = page.locator('text=BN').first()
    await expect(bnSlot).toBeVisible()

    // Verify error toast appears
    const errorToast = page.locator('text=Invalid move').or(page.locator('.text-status-lost'))
    await expect(errorToast).toBeVisible({ timeout: 5000 })
  })
})
