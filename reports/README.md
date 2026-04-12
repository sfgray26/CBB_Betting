# Reports Directory

> **Research output from Kimi CLI and other agents.**  
> **Naming Convention:** `YYYY-MM-DD-descriptive-name.md`

---

## Naming Convention

All reports must follow this format:

```
YYYY-MM-DD-descriptive-name.md
```

**Examples:**
- ✅ `2026-04-10-bdl-api-capabilities.md`
- ✅ `2026-03-13-clv-attribution.md`
- ❌ `K11_CLV_ATTRIBUTION_MARCH_2026.md` (old format)
- ❌ `balldontlie-api-research.md` (missing date)

**Why:** Dates enable chronological sorting and make it easy to identify stale documentation.

---

## Report Categories

### Research Bundle: K-34 to K-38 (April 2026)

Foundation research that unblocks implementation tasks:

| Report | Task | Size | Status |
|--------|------|------|--------|
| [2026-04-10-bdl-api-capabilities.md](2026-04-10-bdl-api-capabilities.md) | K-34 | 24KB | ✅ Complete |
| [2026-04-10-zscore-best-practices.md](2026-04-10-zscore-best-practices.md) | K-35 | 22KB | ✅ Complete |
| [2026-04-10-h2h-scoring-systems.md](2026-04-10-h2h-scoring-systems.md) | K-36 | 18KB | ✅ Complete |
| [2026-04-10-mlb-api-comparison.md](2026-04-10-mlb-api-comparison.md) | K-37 | 22KB | ✅ Complete |
| [2026-04-09-vorp-implementation-guide.md](2026-04-09-vorp-implementation-guide.md) | K-38 | 19KB | ✅ Complete |

### API Research & Specifications

| Report | Date | Topic |
|--------|------|-------|
| [2026-03-06-balldontlie-api-research.md](2026-03-06-balldontlie-api-research.md) | Mar 6 | BDL API capabilities (initial) |
| [2026-04-10-bdl-api-capabilities.md](2026-04-10-bdl-api-capabilities.md) | Apr 10 | BDL API comprehensive (K-34) |
| [2026-03-12-api-ground-truth.md](2026-03-12-api-ground-truth.md) | Mar 12 | API type definitions |
| [2026-04-10-mlb-api-comparison.md](2026-04-10-mlb-api-comparison.md) | Apr 10 | MLB vs BDL vs Statcast (K-37) |

### Data Quality & Validation

| Report | Date | Topic |
|--------|------|-------|
| [2026-04-09-data-pipeline-audit.md](2026-04-09-data-pipeline-audit.md) | Apr 9 | Pipeline analysis |
| [2026-04-01-ingestion-audit.md](2026-04-01-ingestion-audit.md) | Apr 1 | Ingestion flow (K-16) |
| [2026-04-05-raw-ingestion-audit.md](2026-04-05-raw-ingestion-audit.md) | Apr 5 | Raw data audit (K-27) |
| [2026-04-11-orphaned-players-audit.md](2026-04-11-orphaned-players-audit.md) | Apr 11 | Orphan analysis |

### Fantasy Baseball Specifications

| Report | Date | Topic |
|--------|------|-------|
| [2026-04-10-h2h-scoring-systems.md](2026-04-10-h2h-scoring-systems.md) | Apr 10 | H2H One Win format (K-36) |
| [2026-04-09-vorp-implementation-guide.md](2026-04-09-vorp-implementation-guide.md) | Apr 9 | VORP formula (K-38) |
| [2026-04-10-zscore-best-practices.md](2026-04-10-zscore-best-practices.md) | Apr 10 | Z-score methodology (K-35) |
| [2026-04-01-yahoo-player-stats-spec.md](2026-04-01-yahoo-player-stats-spec.md) | Apr 1 | Yahoo stats mapping (K-24) |
| [2026-04-01-fangraphs-column-map.md](2026-04-01-fangraphs-column-map.md) | Apr 1 | FanGraphs mapping (K-25) |

### UAT & Testing Reports

| Report | Date | Topic |
|--------|------|-------|
| [2026-04-01-waiver-wire-uat.md](2026-04-01-waiver-wire-uat.md) | Apr 1 | Waiver wire testing (K-20) |
| [2026-04-01-daily-lineup-uat.md](2026-04-01-daily-lineup-uat.md) | Apr 1 | Lineup optimizer testing (K-21) |
| [2026-04-01-matchup-uat.md](2026-04-01-matchup-uat.md) | Apr 1 | Matchup page testing (K-22) |
| [2026-04-01-settings-uat.md](2026-04-01-settings-uat.md) | Apr 1 | Settings page testing (K-23) |

### Architecture & Infrastructure

| Report | Date | Topic |
|--------|------|-------|
| [2026-04-08-redis-railway-architecture-deep-dive.md](2026-04-08-redis-railway-architecture-deep-dive.md) | Apr 8 | Redis optimization |
| [2026-03-31-architecture-analysis-api-worker-pattern.md](2026-03-31-architecture-analysis-api-worker-pattern.md) | Mar 31 | API worker patterns |
| [2026-04-04-fantasy-edge-decoupling-structural.md](2026-04-04-fantasy-edge-decoupling-structural.md) | Apr 4 | Edge decoupling plan |

### UI/UX Research

| Report | Date | Topic |
|--------|------|-------|
| [2026-04-08-fantasy-baseball-ui-ux-research.md](2026-04-08-fantasy-baseball-ui-ux-research.md) | Apr 8 | UI/UX comprehensive |
| [2026-04-10-revolut-design-implementation-plan.md](2026-04-10-revolut-design-implementation-plan.md) | Apr 10 | Design system plan |
| [2026-04-08-fantasy-baseball-ui-roadmap.md](2026-04-08-fantasy-baseball-ui-roadmap.md) | Apr 8 | UI roadmap |

### Historical/Archive

Older reports with dated naming convention are in the main directory. Undated legacy reports may exist - see git history for dates.

---

## Maintenance

**Adding new reports:**
1. Use format: `YYYY-MM-DD-brief-description.md`
2. Include date in report header
3. Add to appropriate category table above
4. Keep descriptions under 50 characters

**Archive policy:**
- Reports older than 6 months: Keep but mark as historical
- Superseded reports: Move to `archive/` subdirectory with header

---

*Last updated: April 11, 2026*
