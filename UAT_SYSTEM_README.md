# CBB Edge - Automated UAT System

## Overview

A comprehensive, repeatable User Acceptance Testing (UAT) system that evaluates CBB Edge from two expert perspectives:

1. **Elite Fantasy Manager** - Domain expertise in fantasy baseball strategy and roster management
2. **Quant Trading & Sabermetrics Expert** - Deep knowledge of statistical modeling, edge detection, and risk management

The system uses Playwright for browser automation and runs on a scheduled basis to ensure continuous quality monitoring.

## System Architecture

```
├─────────────────────────────────────────────────────────┐
│  Documentation                                                  │
├─────────────────────────────────────────────────────────┤
│  • docs/UAT_WORKFLOW.md              - Overall workflow design    │
│  • docs/UAT_ELITE_FM_CRITERIA.md    - Fantasy manager rubric     │
│  • docs/UAT_QUANT_SABERMETRICS_CRITERIA.md - Quant trading rubric │
│  • docs/UAT_SETUP_GUIDE.md          - Installation & setup       │
└─────────────────────────────────────────────────────────┘
                                │
                                ▼
├─────────────────────────────────────────────────────────┐
│  Automation Scripts                                              │
├─────────────────────────────────────────────────────────┤
│  • scripts/uat_automation.py       - Main test orchestrator     │
│  • scripts/run_uat.sh              - Bash wrapper script        │
└─────────────────────────────────────────────────────────┘
                                │
                                ▼
├─────────────────────────────────────────────────────────┐
│  Generated Reports                                               │
├─────────────────────────────────────────────────────────┤
│  • reports/uat/uat_report_YYYYMMDD_HHMMSS.md   - Full report     │
│  • reports/uat/uat_report_YYYYMMDD_HHMMSS.json - Structured data │
│  • reports/uat/latest_summary.md               - Quick summary   │
└─────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Install Dependencies

```bash
# Install Playwright
pip install playwright
playwright install chromium

# Make scripts executable
chmod +x scripts/uat_automation.py scripts/run_uat.sh
```

### 2. Configure Environment

```bash
export UAT_BASE_URL="https://your-app.railway.app"
export UAT_API_KEY="your_api_key_here"
export DISCORD_WEBHOOK="optional_discord_webhook"
```

### 3. Run UAT Manually

```bash
./scripts/run_uat.sh
```

### 4. Schedule Automated Runs

The system is already scheduled to run **daily at 6:00 AM** (cron job ID: `911d869c89aa`).

To verify:
```bash
hermes cron list
```

## Evaluation Perspectives

### Elite Fantasy Manager

Evaluates from the perspective of a championship-caliber fantasy manager:

| Dimension | Weight | Key Tests |
|-----------|--------|-----------|
| Roster Management | 25% | Lineup optimization, position eligibility |
| Waiver Wire Intelligence | 20% | FAAB recommendations, streaming picks |
| Matchup Analysis | 20% | Weekly projections, win probability |
| Data Quality | 15% | Freshness, accuracy, completeness |
| User Experience | 10% | Navigation, information density |
| Competitive Advantage | 10% | Unique insights, early warnings |

**Pass Criteria**: Score ≥ 8.0/10

### Quant Trading & Sabermetrics

Evaluates from the perspective of a quantitative analyst:

| Dimension | Weight | Key Tests |
|-----------|--------|-----------|
| Edge Calculation | 25% | CLV accuracy, EV calculations |
| Statistical Model Quality | 25% | Backtests, calibration, Sharpe ratio |
| Data Quality | 20% | Source verification, staleness |
| Risk Management | 15% | Kelly criterion, position sizing |
| Model Transparency | 10% | Explainability, audit trail |
| Market Efficiency | 5% | Line movement, arbitrage detection |

**Pass Criteria**: Score ≥ 8.0/10

## Test Coverage

### Features Tested

- ✅ Authentication and login flow
- ✅ Roster management and lineup optimization
- ✅ Waiver wire recommendations
- ✅ Matchup analysis and projections
- ✅ CLV (Closing Line Value) calculations
- ✅ EV (Expected Value) metrics
- ✅ Statistical model outputs
- ✅ Data freshness indicators
- ✅ Navigation and UX

### Scoring Methodology

Each dimension is scored 0-10 based on:
- **Automated checks** (70%): Playwright browser tests
- **Statistical validation** (20%): Backtests, correlations
- **Heuristic evaluation** (10%): Best practice compliance

## Sample Output

```
========================================
CBB EDGE - AUTOMATED UAT EXECUTION
========================================

🔐 Authenticating...
✅ Authentication successful

🎽 ELITE FANTASY MANAGER EVALUATION
------------------------------------------------------------
📋 Testing Roster Management...
✅ Lineup optimization recommendations displayed
✅ Found 15 position badges
✅ 45 stat elements found

🎯 Testing Waiver Wire...
✅ 12 waiver recommendations found
✅ 8 FAAB/bid elements found
✅ 5 category filters available

📊 Testing Matchup Analysis...
✅ 3 win probability indicators found
✅ 6 category projections visible

📊 QUANT TRADING & SABERMETRICS EVALUATION
------------------------------------------------------------
📈 Testing CLV & Edge Calculations...
✅ 15 CLV references found
✅ 4 beat line indicators
⚠️ 0 EV calculations visible

🔬 Testing Statistical Models...
✅ 23 projection elements found
✅ 12 edge indicators found
⚠️ 0 confidence indicators found

🔍 Testing Data Quality...
✅ 3 freshness indicators found
✅ 0 staleness warnings (good)

========================================
UAT SUMMARY
========================================
UAT completed with overall status: ⚠️ ACCEPTABLE

Elite Fantasy Manager Score: 8.4/10
- Roster Management: 9.0
- Waiver Wire: 8.5
- Matchup Analysis: 7.5

Quant Trading Score: 7.2/10
- CLV & Edge: 7.0
- Model Quality: 8.0
- Data Quality: 6.5

Critical Issues: 0

✅ UAT report saved to: reports/uat/uat_report_20260517_060000.md
✅ JSON report saved to: reports/uat/uat_report_20260517_060000.json
```

## Report Structure

### Markdown Report

```markdown
# CBB Edge UAT Report
**Date:** 2026-05-17T06:00:00
**Base URL:** https://your-app.railway.app
**Overall Status:** ⚠️ ACCEPTABLE

## Executive Summary
...

## Scores
| Perspective | Score | Status |
|-------------|-------|--------|
| Elite Fantasy Manager | 8.4/10 | ✅ |
| Quant Trading | 7.2/10 | ⚠️ |

## Critical Issues
...

## Detailed Results
...
```

### JSON Report

```json
{
  "timestamp": "2026-05-17T06:00:00",
  "base_url": "https://your-app.railway.app",
  "elite_fm_score": 8.4,
  "quant_score": 7.2,
  "critical_issues": [],
  "results": [...]
}
```

## Scheduling Options

### Option 1: Cron (Already Configured)

Daily at 6:00 AM:
```
0 6 * * * cd /mnt/c/Users/sfgra/repos/Fixed/cbb-edge && ./scripts/run_uat.sh
```

### Option 2: GitHub Actions

See `docs/UAT_SETUP_GUIDE.md` for CI/CD integration.

### Option 3: Manual Trigger

```bash
# Run immediately
python3 scripts/uat_automation.py \
  --base-url $UAT_BASE_URL \
  --api-key $UAT_API_KEY \
  --verbose
```

## Customization

### Adding New Tests

Edit `scripts/uat_automation.py` and add test methods:

```python
async def test_custom_feature(self) -> UATResult:
    findings = []
    await self.page.goto(f"{self.base_url}/custom-feature")
    # Add your tests
    return UATResult(...)
```

### Adjusting Scoring Weights

Modify weights in the `run_full_uat()` method to match your priorities.

## Troubleshooting

### Common Issues

**Playwright not found**
```bash
pip install playwright
playwright install chromium
```

**Authentication fails**
- Verify API key is correct
- Check application is running
- Ensure `/login` page is accessible

**Tests timeout**
- Increase timeout in `setup()` method
- Check network connectivity
- Run with `--verbose` to see browser

## Maintenance

### Monthly Tasks
- Review UAT reports for trends
- Update test selectors if UI changes
- Adjust scoring based on user feedback

### Quarterly Tasks
- Validate backtests against actual results
- Recalibrate scoring rubrics
- Review and update evaluation personas

## Benefits

1. **Continuous Quality Monitoring** - Catch regressions immediately
2. **Dual Perspective Validation** - Ensure both usability and rigor
3. **Automated Documentation** - Generate reports for stakeholders
4. **Statistical Validation** - Verify claims with data
5. **Scheduled Execution** - Set it and forget it

## Next Steps

1. ✅ Review evaluation criteria in docs/
2. ✅ Configure environment variables
3. ✅ Run manual test to verify setup
4. ✅ Schedule is already active (daily at 6 AM)
5. ✅ Review first automated report tomorrow
6. → Customize tests for your specific features
7. → Set up Discord/Slack notifications
8. → Integrate with CI/CD pipeline

## Support

- **Documentation**: See `docs/UAT_SETUP_GUIDE.md`
- **Evaluation Criteria**: See `docs/UAT_ELITE_FM_CRITERIA.md` and `docs/UAT_QUANT_SABERMETRICS_CRITERIA.md`
- **Playwright Docs**: https://playwright.dev/python/

## License

Part of CBB Edge - Internal Use Only
