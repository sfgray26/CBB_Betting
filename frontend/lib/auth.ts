/**
 * Authentication utilities for API key persistence.
 *
 * Uses dual storage strategy:
 * - localStorage: Primary, persistent until logout
 * - cookie: Fallback for initial auth, 7-day expiry
 */

const API_KEY_COOKIE = 'cbb_api_key'
const API_KEY_LOCALSTORAGE = 'cbb_api_key'
const REDIRECT_PARAM = 'redirect'

/**
 * Get the API key from localStorage (primary) or cookie (fallback).
 */
export function getApiKey(): string {
	if (typeof window === 'undefined') return ''

	// Try localStorage first (primary)
	const localKey = localStorage.getItem(API_KEY_LOCALSTORAGE)
	if (localKey) return localKey

	// Fallback to cookie
	const cookieKey = getCookie(API_KEY_COOKIE)
	if (cookieKey) {
		// Sync to localStorage for future use
		localStorage.setItem(API_KEY_LOCALSTORAGE, cookieKey)
		return cookieKey
	}

	return ''
}

/**
 * Set the API key in both localStorage and cookie.
 */
export function setApiKey(key: string): void {
	if (typeof window === 'undefined') return

	// Store in localStorage (primary)
	localStorage.setItem(API_KEY_LOCALSTORAGE, key)

	// Store in cookie as fallback (7-day expiry)
	const maxAge = 7 * 24 * 60 * 60 // 7 days in seconds
	document.cookie = `${API_KEY_COOKIE}=${encodeURIComponent(key)}; path=/; max-age=${maxAge}; samesite=strict`
}

/**
 * Clear the API key from both localStorage and cookie.
 */
export function clearApiKey(): void {
	if (typeof window === 'undefined') return

	// Clear localStorage
	localStorage.removeItem(API_KEY_LOCALSTORAGE)

	// Clear cookie
	document.cookie = `${API_KEY_COOKIE}=; path=/; max-age=0`
}

/**
 * Check if user is authenticated (has valid API key).
 */
export function isAuthenticated(): boolean {
	return !!getApiKey()
}

/**
 * Get the redirect URL from query params or default to /war-room.
 */
export function getRedirectUrl(): string {
	if (typeof window === 'undefined') return '/war-room'

	const params = new URLSearchParams(window.location.search)
	return params.get(REDIRECT_PARAM) || '/war-room'
}

/**
 * Helper to read a cookie value by name.
 */
function getCookie(name: string): string | null {
	const value = `; ${document.cookie}`
	const parts = value.split(`; ${name}=`)
	if (parts.length === 2) return parts.pop()?.split(';')?.[0] || null
	return null
}
