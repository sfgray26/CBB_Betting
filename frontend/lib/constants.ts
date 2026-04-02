/**
 * Shared constants and helpers used across fantasy baseball pages.
 * Centralised here to prevent drift between matchup, lineup, waiver, and API client.
 */

/**
 * Return today's date as YYYY-MM-DD anchored to US Eastern Time.
 * Uses en-CA locale which produces the ISO-style format natively.
 * Always use this instead of new Date().toISOString().slice(0,10) which is UTC.
 */
export function etTodayStr(): string {
  return new Date().toLocaleDateString('en-CA', { timeZone: 'America/New_York' })
}

// Known stat display names for the 18-category H2H league.
// Includes both Yahoo string keys and numeric category IDs.
export const STAT_LABELS: Record<string, string> = {
  // Batting — string keys
  R: 'Runs',          H: 'Hits',           HR: 'Home Runs',
  RBI: 'RBI',         SB: 'Stolen Bases',  OBP: 'On-Base %',
  AVG: 'Batting Avg', OPS: 'OPS',
  // Batting — numeric IDs
  '3': 'Batting Avg', '7': 'Runs',         '8': 'Hits',
  '12': 'Home Runs',  '13': 'RBI',         '16': 'Stolen Bases',
  '55': 'OPS',        '60': 'Net SB',      '85': 'On-Base %',
  // Batting extra — string keys
  TB: 'Total Bases',  NSB: 'Net SB',       'H/AB': 'H/AB',
  L: 'Losses',
  // Pitching — string keys
  W: 'Wins',          K: 'Strikeouts',     SV: 'Saves',   HLD: 'Holds',
  ERA: 'ERA',         WHIP: 'WHIP',        QS: 'Quality Starts',
  BB: 'Walks (P)',    IP: 'Innings Pitched', K9: 'K/9',   'K/9': 'K/9',
  NSV: 'Net Saves',   GS: 'Games Started', 'K/BB': 'K/BB',
  // Pitching — numeric IDs (57=BB confirmed per K-14 research; 85=OBP already mapped above)
  '57': 'Walks',
  '21': 'Innings Pitched', '23': 'Wins',      '26': 'ERA',
  '27': 'WHIP',            '28': 'Strikeouts', '29': 'Quality Starts',
  '32': 'Saves',           '38': 'K/BB',       '42': 'Strikeouts',
  '48': 'Holds',           '50': 'Innings Pitched', '53': 'K/BB',
  '62': 'Games Started',   '83': 'Net Saves',
}

// Ratio stats get 3 decimal places; counting stats get integer display.
export const RATIO_STATS = new Set([
  'AVG', 'OBP', 'OPS', 'ERA', 'WHIP', 'K9', 'K/9', '3', '26', '27', '55',
])

// Stats where LOWER value wins the category.
// K-26 fix: added L (Losses), K/BB (lower K/BB is worse — actually higher K/BB is better —
// but L (Losses) is definitely lower-is-better).
export const LOWER_IS_BETTER = new Set([
  'ERA', 'WHIP', 'L',
  '26',  // ERA by numeric ID
  '27',  // WHIP by numeric ID
  '29',  // L (Losses) by numeric ID — some leagues use QS here; backend overrides via settings
])

// K-26 fix: display-only reference stats that must NOT count toward the matchup score.
// H/AB and IP are informational; GS is a pitching reference, not a scoring category.
export const MATCHUP_DISPLAY_ONLY = new Set([
  'IP', 'H/AB', 'GS',
  '21', '50',  // IP by numeric ID
  '62',        // GS by numeric ID
])

// Canonical display order for matchup categories.
// Batting section first, then pitching. Stats not in this list are appended at the end.
export const MATCHUP_STAT_ORDER: string[] = [
  // Batting
  'H/AB', 'R', 'H', 'HR', 'RBI', 'SB', 'NSB', 'TB', 'AVG', 'OBP', 'OPS',
  // Pitching
  'IP', 'W', 'L', 'K', 'SV', 'NSV', 'HLD', 'ERA', 'WHIP', 'K/9', 'K9', 'QS', 'K/BB', 'GS', 'BB',
]
