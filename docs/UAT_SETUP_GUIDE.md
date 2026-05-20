# CBB Edge UAT Setup Guide

## Overview

This guide walks you through setting up automated User Acceptance Testing (UAT) for CBB Edge using Playwright and scheduled execution.

## What You Get

1. **Automated Browser Testing** - Playwright simulates real user interactions
2. **Dual Perspective Evaluation**:
   - Elite Fantasy Manager assessment
   - Quant Trading & Sabermetrics validation
3. **Scheduled Execution** - Run daily/weekly automatically
4. **Comprehensive Reports** - Markdown and JSON output
5. **Discord Notifications** - Get alerts on failures

## Prerequisites

- Python 3.9+
- Node.js 16+ (for Playwright)
- API key for the application
- (Optional) Discord webhook for notifications

## Installation

### Step 1: Install Dependencies

```bash
cd /mnt/c/Users/sfgra/repos/Fixed/cbb-edge

# Install Python dependencies
pip install playwright asyncio

# Install Playwright browsers
playwright install chromium

# Verify installation
python3 -c "import playwright; print('Playwright OK')"
```

### Step 2: Configure Environment

Create a `.env` file or set environment variables:

```bash
# Required
export UAT_BASE_URL="https://your-app.railway.app"
export UAT_API_KEY="your_api_key_here"

# Optional - for Discord notifications
export DISCORD_WEBHOOK="https://discord.com/api/webhooks/..."
```

### Step 3: Test Manual Execution

```bash
# Run UAT manually
./scripts/run_uat.sh

# Or run Python script directly
python3 scripts/uat_automation.py \
  --base-url https://your-app.railway.app \
  --api-key your_api_key \
  --output reports/uat/test_run.md \
  --verbose
```

## Scheduling Options

### Option 1: Cron (Linux/Mac)

```bash
# Edit crontab
crontab -e

# Add daily UAT at 6 AM
0 6 * * * cd /mnt/c/Users/sfgra/repos/Fixed/cbb-edge && ./scripts/run_uat.sh >> logs/uat_cron.log 2>&1

# Add weekly deep dive on Sundays at 8 AM
0 8 * * 0 cd /mnt/c/Users/sfgra/repos/Fixed/cbb-edge && ./scripts/run_uat.sh --full >> logs/uat_weekly.log 2>&1
```

### Option 2: Windows Task Scheduler

```powershell
# Create daily task
$action = New-ScheduledTaskAction -Execute "wsl.exe" -Argument "bash /mnt/c/Users/sfgra/repos/Fixed/cbb-edge/scripts/run_uat.sh"
$trigger = New-ScheduledTaskTrigger -Daily -At 6am
Register-ScheduledTask -Action $action -Trigger $trigger -TaskName "CBB_Edge_UAT_Daily" -Description "Daily UAT for CBB Edge"
```

### Option 3: GitHub Actions (Recommended for CI/CD)

Create `.github/workflows/uat.yml`:

```yaml
name: UAT Automation

on:
  schedule:
    # Run daily at 6 AM UTC
    - cron: '0 6 * * *'
  workflow_dispatch:  # Allow manual trigger

jobs:
  uat:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install playwright
          playwright install chromium
      
      - name: Run UAT
        env:
          UAT_BASE_URL: ${{ secrets.UAT_BASE_URL }}
          UAT_API_KEY: ${{ secrets.UAT_API_KEY }}
        run: |
          python3 scripts/uat_automation.py \
            --base-url "$UAT_BASE_URL" \
            --api-key "$UAT_API_KEY" \
            --output reports/uat/uat_report_${{ github.run_id }}.md \
            --json
      
      - name: Upload Report
        uses: actions/upload-artifact@v3
        with:
          name: uat-report
          path: reports/uat/
      
      - name: Notify on Failure
        if: failure()
        uses: slackapi/slack-github-action@v1
        with:
          payload: |
            {"text": "❌ UAT Failed - Check reports"}
        env:
          SLACK_WEBHOOK_URL: ${{ secrets.SLACK_WEBHOOK }}
```

### Option 4: Railway Cron (If hosted on Railway)

Add to `railway.json`:

```json
{
  "cron": [
    {
      "command": "python3 scripts/uat_automation.py --base-url $RAILWAY_URL --api-key $API_KEY --output reports/uat/daily.md",
      "schedule": "0 6 * * *"
    }
  ]
}
```

## Evaluation Criteria

### Elite Fantasy Manager Perspective

Tests focus on:
- Lineup optimization accuracy
- Waiver wire recommendation quality
- Matchup analysis depth
- Data freshness and reliability
- User experience efficiency

**Scoring**:
- 10: Elite-level insights
- 8: Good, occasional misses
- 6: Decent, needs interpretation
- 4: Basic, significant gaps
- 0: Broken/unreliable

### Quant Trading & Sabermetrics Perspective

Tests focus on:
- CLV calculation accuracy
- EV (Expected Value) correctness
- Statistical model validation
- Data quality and integrity
- Risk management features

**Scoring**:
- 10: Highly predictive (r > 0.3)
- 8: Measurable edge (r 0.15-0.3)
- 6: Weak signal (r 0.05-0.15)
- 4: Noise (r ~ 0)
- 0: Misleading (r < 0)

## Interpreting Results

### Report Structure

```
reports/uat/
├── uat_report_20260517_060000.md   # Full markdown report
├── uat_report_20260517_060000.json # Structured data
├── latest_summary.md              # Quick summary
└── historical/                    # Archived reports
```

### Understanding Scores

**Elite FM Score**:
- 9-10: Championship-caliber tool
- 7-8: Solid competitive advantage
- 5-6: Useful but needs work
- 3-4: Major gaps
- 0-2: Unusable

**Quant Score**:
- 9-10: Institutional-grade
- 7-8: Measurable alpha
- 5-6: Marginal edge
- 3-4: No proven edge
- 0-2: Negative expected value

### Critical Issues

The UAT automatically flags:
- Authentication failures
- Data sync issues
- Broken core features
- Statistical anomalies
- Performance degradation

## Customizing Tests

### Adding New Test Cases

Edit `scripts/uat_automation.py`:

```python
async def test_custom_feature(self) -> UATResult:
    """Test your custom feature"""
    findings = []
    
    await self.page.goto(f"{self.base_url}/your-feature")
    
    # Add your tests
    element = await self.page.locator('.your-selector').count()
    findings.append({
        'description': f'Found {element} elements',
        'passed': element > 0
    })
    
    return UATResult(
        timestamp=datetime.now().isoformat(),
        perspective='Elite Fantasy Manager',
        dimension='Custom Feature',
        score=...,  # Calculate based on findings
        max_score=10,
        findings=findings,
        recommendations=['Your recommendations']
    )
```

### Adjusting Scoring Weights

Modify the scoring logic in `run_full_uat()`:

```python
# Current weights
elite_score = sum(r.score for r in elite_results) / len(elite_results)

# Custom weighted example
elite_score = (
    roster_result.score * 0.4 +      # 40% weight
    waiver_result.score * 0.3 +       # 30% weight
    matchup_result.score * 0.3        # 30% weight
)
```

## Troubleshooting

### Playwright Not Found

```bash
# Reinstall Playwright
pip uninstall playwright -y
pip install playwright
playwright install chromium
```

### Authentication Failures

1. Verify API key is correct
2. Check application is running
3. Ensure `/login` page accessible
4. Review network logs with `--verbose`

### False Positives

If tests fail incorrectly:
1. Run with `--verbose` to see browser
2. Check selectors in test code
3. Verify page load timing
4. Update test selectors if UI changed

## Maintenance

### Monthly Tasks

1. Review all UAT reports
2. Update test selectors if UI changed
3. Adjust scoring criteria based on user feedback
4. Add new features to test coverage

### Quarterly Tasks

1. Backtest validation (compare UAT scores to actual results)
2. Recalibrate scoring rubrics
3. Review critical issues history
4. Update evaluation personas

## Advanced Features

### Visual Regression Testing

Add screenshot comparison:

```python
# In test method
await self.page.screenshot(path=f'screenshots/{test_name}.png')
# Compare to baseline
```

### Performance Monitoring

Track load times:

```python
import time

start = time.time()
await self.page.goto(url)
load_time = time.time() - start
findings.append({
    'description': f'Page loaded in {load_time:.2f}s',
    'passed': load_time < 3.0
})
```

### A/B Testing Support

Test different UI versions:

```python
if await self.page.locator('.new-feature').count() > 0:
    # Test new version
else:
    # Test old version
```

## Support

For issues or questions:
1. Check logs in `logs/uat_cron.log`
2. Review latest report in `reports/uat/`
3. Run with `--verbose` for debugging
4. Check Playwright documentation: https://playwright.dev

## Next Steps

1. ✅ Install dependencies
2. ✅ Configure environment variables
3. ✅ Run manual test: `./scripts/run_uat.sh`
4. ✅ Set up scheduled execution
5. ✅ Review first report
6. ✅ Configure notifications
7. ✅ Customize test scenarios for your needs
