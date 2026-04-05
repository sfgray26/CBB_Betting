import { test, expect } from '@playwright/test'

/**
 * Smoke tests — verify the app shell loads without crashing.
 * These tests use page.route() mocks; no real backend is needed.
 */

test.describe('App shell', () => {
  test('dashboard page loads without JS errors', async ({ page }) => {
    // Mock the dashboard API so the page doesn't hang waiting for a real backend
    await page.route('**/api/dashboard', (route) =>
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ success: true, data: { user_id: 'test', timestamp: new Date().toISOString() } }),
      })
    )

    const errors: string[] = []
    page.on('pageerror', (err) => errors.push(err.message))

    await page.goto('/dashboard')
    await page.waitForLoadState('networkidle')

    expect(errors).toHaveLength(0)
  })

  test('root page loads', async ({ page }) => {
    const errors: string[] = []
    page.on('pageerror', (err) => errors.push(err.message))

    await page.goto('/')
    await page.waitForLoadState('networkidle')

    expect(errors).toHaveLength(0)
  })
})
