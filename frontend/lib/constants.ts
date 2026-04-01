/**
 * Shared constants used across fantasy baseball pages.
 * Centralised here to prevent drift between matchup, lineup, and waiver pages.
 */

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
  '55': 'OPS',        '60': 'Hits',        '85': 'On-Base %',
  // Pitching — string keys
  W: 'Wins',          K: 'Strikeouts',     SV: 'Saves',   HLD: 'Holds',
  ERA: 'ERA',         WHIP: 'WHIP',        QS: 'Quality Starts',
  BB: 'Walks (P)',    IP: 'Innings Pitched', K9: 'K/9',
  NSV: 'Net Saves',
  // Pitching — numeric IDs (57=BB and 85=OBP excluded: unconfirmed for this league)
  '21': 'Innings Pitched', '23': 'Wins',      '26': 'ERA',
  '27': 'WHIP',            '28': 'Strikeouts', '29': 'Quality Starts',
  '32': 'Saves',           '38': 'K/BB',       '42': 'Strikeouts',
  '50': 'Innings Pitched',                     '62': 'Games Started',
  '83': 'Net Saves',
}

// Ratio stats get 3 decimal places; counting stats get integer display.
export const RATIO_STATS = new Set([
  'AVG', 'OBP', 'OPS', 'ERA', 'WHIP', 'K9', '3', '26', '27', '55',
])

// ERA / WHIP: lower is better.
export const LOWER_IS_BETTER = new Set(['ERA', 'WHIP', '26', '27'])
