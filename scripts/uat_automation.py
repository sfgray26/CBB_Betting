#!/usr/bin/env python3
"""
CBB Edge - Automated UAT Execution Script

This script orchestrates comprehensive User Acceptance Testing from two perspectives:
1. Elite Fantasy Manager perspective
2. Quant Trading & Sabermetrics perspective

Uses Playwright for browser automation and includes statistical validation.
"""

import asyncio
import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import subprocess

# Try to import playwright
try:
    from playwright.async_api import async_playwright, Page, Browser, BrowserContext
except ImportError:
    print("Playwright not installed. Install with: pip install playwright")
    print("Then run: playwright install")
    sys.exit(1)


@dataclass
class UATResult:
    """Structured result from UAT evaluation"""
    timestamp: str
    perspective: str
    dimension: str
    score: float
    max_score: float
    findings: List[Dict]
    recommendations: List[str]
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class UATReport:
    """Complete UAT report"""
    timestamp: str
    base_url: str
    elite_fm_score: float
    quant_score: float
    critical_issues: List[Dict]
    all_results: List[UATResult]
    executive_summary: str
    
    def generate_markdown(self) -> str:
        """Generate markdown report"""
        md = f"""# CBB Edge UAT Report
**Date:** {self.timestamp}
**Base URL:** {self.base_url}
**Overall Status:** {'✅ PASS' if self.elite_fm_score >= 8.0 and self.quant_score >= 8.0 else '⚠️ ACCEPTABLE' if self.elite_fm_score >= 6.0 and self.quant_score >= 6.0 else '❌ FAIL'}

---

## Executive Summary

{self.executive_summary}

---

## Scores

| Perspective | Score | Status |
|-------------|-------|--------|
| Elite Fantasy Manager | {self.elite_fm_score:.1f}/10 | {'✅' if self.elite_fm_score >= 8.0 else '⚠️' if self.elite_fm_score >= 6.0 else '❌'} |
| Quant Trading & Sabermetrics | {self.quant_score:.1f}/10 | {'✅' if self.quant_score >= 8.0 else '⚠️' if self.quant_score >= 6.0 else '❌'} |

---

## Critical Issues ({len(self.critical_issues)})

"""
        for i, issue in enumerate(self.critical_issues, 1):
            md += f"""### {i}. {issue['title']} ({issue['severity']})
**Dimension:** {issue['dimension']}

{issue['description']}

**Impact:** {issue['impact']}
**Recommended Action:** {issue['action']}

---

"""
        
        md += """## Detailed Results

"""
        for result in self.all_results:
            md += f"""### {result.perspective} - {result.dimension}
**Score:** {result.score:.1f}/{result.max_score}

**Findings:**
"""
            for finding in result.findings:
                status = "✅" if finding.get("passed") else "❌" if finding.get("critical") else "⚠️"
                md += f"- {status} {finding['description']}\n"
            
            if result.recommendations:
                md += "\n**Recommendations:**\n"
                for rec in result.recommendations:
                    md += f"- {rec}\n"
            
            md += "\n---\n\n"
        
        return md


class UATExecutor:
    """Main UAT execution orchestrator"""
    
    def __init__(self, base_url: str, api_key: str, verbose: bool = False):
        self.base_url = base_url
        self.api_key = api_key
        self.verbose = verbose
        self.results: List[UATResult] = []
        self.critical_issues: List[Dict] = []
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None
        
    async def setup(self):
        """Initialize browser and context"""
        playwright = await async_playwright().start()
        self.browser = await playwright.chromium.launch(headless=not self.verbose)
        self.context = await self.browser.new_context(
            viewport={'width': 1440, 'height': 900},
            record_video_dir='./uat_videos/' if self.verbose else None
        )
        self.page = await self.context.new_page()
        
        # Set default timeout
        self.page.set_default_timeout(10000)
        
    async def teardown(self):
        """Clean up browser resources"""
        if self.context:
            await self.context.close()
        if self.browser:
            await self.browser.close()
    
    async def authenticate(self) -> bool:
        """Authenticate with the application"""
        try:
            print("🔐 Authenticating...")
            await self.page.goto(f"{self.base_url}/login")
            
            # Wait for login form
            await self.page.wait_for_selector('input#apikey')
            
            # Enter API key
            await self.page.fill('input#apikey', self.api_key)
            
            # Click sign in
            await self.page.click('button[type="submit"]')
            
            # Wait for navigation to dashboard or performance
            await self.page.wait_for_url(lambda url: '/dashboard' in url or '/performance' in url, timeout=15000)
            
            print("✅ Authentication successful")
            return True
            
        except Exception as e:
            print(f"❌ Authentication failed: {e}")
            return False
    
    async def check_data_freshness(self) -> Dict:
        """Check if data is fresh"""
        try:
            await self.page.goto(f"{self.base_url}/dashboard")
            await self.page.wait_for_load_state('networkidle')
            
            # Look for last updated timestamp
            last_updated_text = await self.page.locator('text=/Last updated/i').first.text_content()
            
            # Check if roster data loads
            roster_link = await self.page.locator('text=My Roster').first
            if await roster_link.is_visible():
                await roster_link.click()
                await self.page.wait_for_load_state('networkidle')
                
                # Check for player data
                player_count = await self.page.locator('.player-row, [data-testid="player"]').count()
                
                return {
                    'last_updated': last_updated_text,
                    'players_loaded': player_count > 0,
                    'player_count': player_count
                }
            
            return {'error': 'Roster link not found'}
            
        except Exception as e:
            return {'error': str(e)}
    
    # Send notification using existing Discord service
    async def send_notification(self, report: UATReport):
        """Send UAT report notification using existing Discord notifier service"""
        import os
        import sys
        
        # Add backend to path for imports
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        
        try:
            from backend.services.discord_notifier import send_to_channel, route_notification
            
            status_emoji = "✅" if report.elite_fm_score >= 8.0 and report.quant_score >= 8.0 else "⚠️" if report.elite_fm_score >= 6.0 and report.quant_score >= 6.0 else "❌"
            
            # Determine color based on scores
            if report.elite_fm_score >= 8.0 and report.quant_score >= 8.0:
                color = 0x2ECC71  # Green
                severity = "normal"
            elif report.elite_fm_score >= 6.0 and report.quant_score >= 6.0:
                color = 0xF1C40F  # Yellow
                severity = "warning"
            else:
                color = 0xE74C3C  # Red
                severity = "critical"
            
            # Build embed
            embed = {
                "title": f"{status_emoji} UAT Report - CBB Edge",
                "description": f"Daily automated testing completed for {self.base_url}",
                "color": color,
                "fields": [
                    {"name": "🎽 Elite FM Score", "value": f"{report.elite_fm_score:.1f}/10", "inline": True},
                    {"name": "📊 Quant Score", "value": f"{report.quant_score:.1f}/10", "inline": True},
                    {"name": "⚠️ Critical Issues", "value": str(len(report.critical_issues)), "inline": True},
                    {"name": "📁 Full Report", "value": f"View at: `{self.base_url}/reports/uat/latest_summary.md`", "inline": False}
                ],
                "timestamp": datetime.now().isoformat(),
                "footer": {"text": "CBB Edge UAT Automation"}
            }
            
            # Try to send to system-logs channel first, fallback to general
            success = send_to_channel("system-logs", embed=embed)
            if not success:
                success = send_to_channel("general", embed=embed)
            
            if success:
                print("📢 Discord notification sent via existing bot")
            else:
                print("⚠️ Discord notification failed - check DISCORD_BOT_TOKEN and channel IDs")
                
        except ImportError as e:
            print(f"⚠️ Could not import discord_notifier: {e}")
            print("   Notifications will not be sent")
        except Exception as e:
            print(f"⚠️ Discord notification error: {e}")

    # ============== Elite Fantasy Manager Tests ==============
    
    async def test_roster_management(self) -> UATResult:
        """Test roster management functionality"""
        print("📋 Testing Roster Management...")
        findings = []
        
        try:
            await self.page.goto(f"{self.base_url}/war-room/roster")
            await self.page.wait_for_load_state('networkidle')
            
            # Test 1: Lineup optimization
            try:
                await self.page.wait_for_selector('[data-testid="optimal-lineup"], .optimal-lineup, .lineup-recommendation', timeout=5000)
                findings.append({
                    'description': 'Lineup optimization recommendations displayed',
                    'passed': True
                })
            except:
                findings.append({
                    'description': 'Lineup optimization not visible or missing',
                    'passed': False,
                    'critical': True
                })
            
            # Test 2: Position eligibility
            position_badges = await self.page.locator('.position-badge, [data-testid="position"]').count()
            findings.append({
                'description': f'Found {position_badges} position badges',
                'passed': position_badges > 0
            })
            
            # Test 3: Player stats visible
            stats_visible = await self.page.locator('.player-stats, .stat-value').count()
            findings.append({
                'description': f'{stats_visible} stat elements found',
                'passed': stats_visible > 0
            })
            
            # Calculate score
            passed = sum(1 for f in findings if f.get('passed'))
            critical_failures = sum(1 for f in findings if not f.get('passed') and f.get('critical'))
            
            score = (passed / len(findings)) * 10 if findings else 0
            if critical_failures > 0:
                score = max(0, score - (critical_failures * 3))
            
            return UATResult(
                timestamp=datetime.now().isoformat(),
                perspective='Elite Fantasy Manager',
                dimension='Roster Management',
                score=score,
                max_score=10,
                findings=findings,
                recommendations=['Add explicit lineup optimization section' if not any('Lineup' in f['description'] and f['passed'] for f in findings) else 'None']
            )
            
        except Exception as e:
            return UATResult(
                timestamp=datetime.now().isoformat(),
                perspective='Elite Fantasy Manager',
                dimension='Roster Management',
                score=0,
                max_score=10,
                findings=[{'description': f'Error: {str(e)}', 'passed': False, 'critical': True}],
                recommendations=['Fix roster page loading issues']
            )
    
    async def test_waiver_wire(self) -> UATResult:
        """Test waiver wire intelligence"""
        print("🎯 Testing Waiver Wire...")
        findings = []
        
        try:
            await self.page.goto(f"{self.base_url}/war-room/waiver")
            await self.page.wait_for_load_state('networkidle')
            
            # Test 1: Player recommendations visible
            players = await self.page.locator('.player-card, .waiver-player, [data-testid="waiver-player"]').count()
            findings.append({
                'description': f'{players} waiver recommendations found',
                'passed': players > 0,
                'critical': True
            })
            
            # Test 2: FAAB or bid suggestions
            faab_elements = await self.page.locator('text=/\\$\\d+/, text=/bid/i, text=/FAAB/i').count()
            findings.append({
                'description': f'{faab_elements} FAAB/bid elements found',
                'passed': faab_elements > 0
            })
            
            # Test 3: Category targeting
            category_filters = await self.page.locator('text=/HR/i, text=/SB/i, text=/RBI/i, text=/OPS/i').count()
            findings.append({
                'description': f'{category_filters} category filters available',
                'passed': category_filters >= 3
            })
            
            passed = sum(1 for f in findings if f.get('passed'))
            critical_failures = sum(1 for f in findings if not f.get('passed') and f.get('critical'))
            
            score = (passed / len(findings)) * 10 if findings else 0
            if critical_failures > 0:
                score = max(0, score - 5)
            
            return UATResult(
                timestamp=datetime.now().isoformat(),
                perspective='Elite Fantasy Manager',
                dimension='Waiver Wire Intelligence',
                score=score,
                max_score=10,
                findings=findings,
                recommendations=['Add FAAB bid recommendations' if faab_elements == 0 else 'None']
            )
            
        except Exception as e:
            return UATResult(
                timestamp=datetime.now().isoformat(),
                perspective='Elite Fantasy Manager',
                dimension='Waiver Wire Intelligence',
                score=0,
                max_score=10,
                findings=[{'description': f'Error: {str(e)}', 'passed': False, 'critical': True}],
                recommendations=['Fix waiver wire page loading']
            )
    
    async def test_matchup_analysis(self) -> UATResult:
        """Test matchup analysis features"""
        print("📊 Testing Matchup Analysis...")
        findings = []
        
        try:
            # Navigate to war room for matchup data
            await self.page.goto(f"{self.base_url}/war-room")
            await self.page.wait_for_load_state('networkidle')
            
            # Test 1: Win probability visible
            win_prob = await self.page.locator('text=/\\d+% win prob/, text=/win probability/i').count()
            findings.append({
                'description': f'{win_prob} win probability indicators found',
                'passed': win_prob > 0
            })
            
            # Test 2: Category projections
            categories = await self.page.locator('text=/HR:/i, text=/R:/i, text=/RBI:/i, text=/SB:/i').count()
            findings.append({
                'description': f'{categories} category projections visible',
                'passed': categories >= 4
            })
            
            passed = sum(1 for f in findings if f.get('passed'))
            score = (passed / len(findings)) * 10 if findings else 0
            
            return UATResult(
                timestamp=datetime.now().isoformat(),
                perspective='Elite Fantasy Manager',
                dimension='Matchup Analysis',
                score=score,
                max_score=10,
                findings=findings,
                recommendations=['Add category-by-category projections' if categories < 4 else 'None']
            )
            
        except Exception as e:
            return UATResult(
                timestamp=datetime.now().isoformat(),
                perspective='Elite Fantasy Manager',
                dimension='Matchup Analysis',
                score=0,
                max_score=10,
                findings=[{'description': f'Error: {str(e)}', 'passed': False}],
                recommendations=['Fix matchup analysis loading']
            )
    
    # ============== Quant Trading Tests ==============
    
    async def test_clv_and_edge(self) -> UATResult:
        """Test Closing Line Value and edge calculations"""
        print("📈 Testing CLV & Edge Calculations...")
        findings = []
        
        try:
            await self.page.goto(f"{self.base_url}/clv")
            await self.page.wait_for_load_state('networkidle')
            
            # Test 1: CLV data visible
            clv_elements = await self.page.locator('text=/CLV/i, text=/line value/i').count()
            findings.append({
                'description': f'{clv_elements} CLV references found',
                'passed': clv_elements > 0,
                'critical': True
            })
            
            # Test 2: Beat line percentage
            beat_line = await self.page.locator('text=/beat/i, text=/closing/i').count()
            findings.append({
                'description': f'{beat_line} beat line indicators',
                'passed': beat_line > 0
            })
            
            # Test 3: EV calculations
            ev_elements = await self.page.locator('text=/EV/i, text=/expected value/i, text=/\\+\\d\\.\\d%$/').count()
            findings.append({
                'description': f'{ev_elements} EV calculations visible',
                'passed': ev_elements > 0
            })
            
            passed = sum(1 for f in findings if f.get('passed'))
            critical_failures = sum(1 for f in findings if not f.get('passed') and f.get('critical'))
            
            score = (passed / len(findings)) * 10 if findings else 0
            if critical_failures > 0:
                score = max(0, score - 5)
            
            return UATResult(
                timestamp=datetime.now().isoformat(),
                perspective='Quant Trading',
                dimension='CLV & Edge Calculation',
                score=score,
                max_score=10,
                findings=findings,
                recommendations=['Implement CLV tracking' if clv_elements == 0 else 'None']
            )
            
        except Exception as e:
            return UATResult(
                timestamp=datetime.now().isoformat(),
                perspective='Quant Trading',
                dimension='CLV & Edge Calculation',
                score=0,
                max_score=10,
                findings=[{'description': f'Error: {str(e)}', 'passed': False, 'critical': True}],
                recommendations=['Fix CLV page loading']
            )
    
    async def test_statistical_models(self) -> UATResult:
        """Test statistical model quality"""
        print("🔬 Testing Statistical Models...")
        findings = []
        
        try:
            # Check for projections
            await self.page.goto(f"{self.base_url}/today")
            await self.page.wait_for_load_state('networkidle')
            
            # Test 1: Projections visible
            projections = await self.page.locator('.projection, [data-testid="projection"], text=/projection/i').count()
            findings.append({
                'description': f'{projections} projection elements found',
                'passed': projections > 0
            })
            
            # Test 2: Edge percentages
            edge_elements = await self.page.locator('text=/\\d+\\.\\d% edge/i, text=/edge:/i').count()
            findings.append({
                'description': f'{edge_elements} edge indicators found',
                'passed': edge_elements > 0
            })
            
            # Test 3: Confidence levels
            confidence = await self.page.locator('text=/high conf/i, text=/medium conf/i, text=/low conf/i').count()
            findings.append({
                'description': f'{confidence} confidence indicators found',
                'passed': confidence > 0
            })
            
            passed = sum(1 for f in findings if f.get('passed'))
            score = (passed / len(findings)) * 10 if findings else 0
            
            return UATResult(
                timestamp=datetime.now().isoformat(),
                perspective='Quant Trading',
                dimension='Statistical Model Quality',
                score=score,
                max_score=10,
                findings=findings,
                recommendations=['Add confidence intervals' if confidence == 0 else 'None']
            )
            
        except Exception as e:
            return UATResult(
                timestamp=datetime.now().isoformat(),
                perspective='Quant Trading',
                dimension='Statistical Model Quality',
                score=0,
                max_score=10,
                findings=[{'description': f'Error: {str(e)}', 'passed': False}],
                recommendations=['Fix model display issues']
            )
    
    async def test_data_quality(self) -> UATResult:
        """Test data quality indicators"""
        print("🔍 Testing Data Quality...")
        findings = []
        
        try:
            # Check for data freshness indicators
            await self.page.goto(f"{self.base_url}/dashboard")
            await self.page.wait_for_load_state('networkidle')
            
            # Test 1: Last updated timestamp
            last_updated = await self.page.locator('text=/Last updated/i, text=/updated:/i').count()
            findings.append({
                'description': f'{last_updated} freshness indicators found',
                'passed': last_updated > 0
            })
            
            # Test 2: Data staleness warnings
            stale_warnings = await self.page.locator('text=/stale/i, text=/outdated/i, text=/delayed/i').count()
            findings.append({
                'description': f'{stale_warnings} staleness warnings (good if 0)',
                'passed': stale_warnings == 0  # No warnings is good
            })
            
            passed = sum(1 for f in findings if f.get('passed'))
            score = (passed / len(findings)) * 10 if findings else 0
            
            return UATResult(
                timestamp=datetime.now().isoformat(),
                perspective='Quant Trading',
                dimension='Data Quality & Integrity',
                score=score,
                max_score=10,
                findings=findings,
                recommendations=['Add data freshness timestamps' if last_updated == 0 else 'None']
            )
            
        except Exception as e:
            return UATResult(
                timestamp=datetime.now().isoformat(),
                perspective='Quant Trading',
                dimension='Data Quality & Integrity',
                score=0,
                max_score=10,
                findings=[{'description': f'Error: {str(e)}', 'passed': False}],
                recommendations=['Fix data quality indicators']
            )
    
    async def run_full_uat(self) -> UATReport:
        """Execute complete UAT suite"""
        print("\n" + "="*60)
        print("CBB EDGE - AUTOMATED UAT EXECUTION")
        print("="*60 + "\n")
        
        # Setup
        await self.setup()
        
        try:
            # Authenticate
            if not await self.authenticate():
                return UATReport(
                    timestamp=datetime.now().isoformat(),
                    base_url=self.base_url,
                    elite_fm_score=0,
                    quant_score=0,
                    critical_issues=[{
                        'title': 'Authentication Failed',
                        'severity': 'P0',
                        'dimension': 'Infrastructure',
                        'description': 'Could not authenticate with provided API key',
                        'impact': 'Cannot perform any UAT tests',
                        'action': 'Verify API key and application availability'
                    }],
                    all_results=[],
                    executive_summary='UAT failed at authentication stage. Application may be down or API key invalid.'
                )
            
            # Check data freshness
            freshness = await self.check_data_freshness()
            if 'error' in freshness:
                print(f"⚠️ Data freshness check warning: {freshness['error']}")
            
            # Run Elite Fantasy Manager tests
            print("\n🎽 ELITE FANTASY MANAGER EVALUATION")
            print("-" * 60)
            
            roster_result = await self.test_roster_management()
            self.results.append(roster_result)
            
            waiver_result = await self.test_waiver_wire()
            self.results.append(waiver_result)
            
            matchup_result = await self.test_matchup_analysis()
            self.results.append(matchup_result)
            
            # Run Quant Trading tests
            print("\n📊 QUANT TRADING & SABERMETRICS EVALUATION")
            print("-" * 60)
            
            clv_result = await self.test_clv_and_edge()
            self.results.append(clv_result)
            
            model_result = await self.test_statistical_models()
            self.results.append(model_result)
            
            data_result = await self.test_data_quality()
            self.results.append(data_result)
            
            # Calculate scores
            elite_results = [r for r in self.results if r.perspective == 'Elite Fantasy Manager']
            quant_results = [r for r in self.results if r.perspective == 'Quant Trading']
            
            elite_score = sum(r.score for r in elite_results) / len(elite_results) if elite_results else 0
            quant_score = sum(r.score for r in quant_results) / len(quant_results) if quant_results else 0
            
            # Collect critical issues
            for result in self.results:
                for finding in result.findings:
                    if not finding.get('passed') and finding.get('critical'):
                        self.critical_issues.append({
                            'title': finding['description'][:50],
                            'severity': 'P0' if result.score < 5 else 'P1',
                            'dimension': result.dimension,
                            'description': finding['description'],
                            'impact': 'Blocks core functionality' if result.score < 5 else 'Degrades user experience',
                            'action': result.recommendations[0] if result.recommendations else 'Investigate and fix'
                        })
            
            # Generate executive summary
            status = '✅ PASS' if elite_score >= 8.0 and quant_score >= 8.0 else '⚠️ ACCEPTABLE' if elite_score >= 6.0 and quant_score >= 6.0 else '❌ FAIL'
            
            summary = f"""UAT completed with overall status: {status}

Elite Fantasy Manager Score: {elite_score:.1f}/10
- Roster Management: {next((r.score for r in elite_results if r.dimension == 'Roster Management'), 0):.1f}
- Waiver Wire: {next((r.score for r in elite_results if r.dimension == 'Waiver Wire Intelligence'), 0):.1f}
- Matchup Analysis: {next((r.score for r in elite_results if r.dimension == 'Matchup Analysis'), 0):.1f}

Quant Trading Score: {quant_score:.1f}/10
- CLV & Edge: {next((r.score for r in quant_results if r.dimension == 'CLV & Edge Calculation'), 0):.1f}
- Model Quality: {next((r.score for r in quant_results if r.dimension == 'Statistical Model Quality'), 0):.1f}
- Data Quality: {next((r.score for r in quant_results if r.dimension == 'Data Quality & Integrity'), 0):.1f}

Critical Issues: {len(self.critical_issues)}
"""
            
            report = UATReport(
                timestamp=datetime.now().isoformat(),
                base_url=self.base_url,
                elite_fm_score=elite_score,
                quant_score=quant_score,
                critical_issues=self.critical_issues,
                all_results=self.results,
                executive_summary=summary
            )
            
            # Send notifications
            await self.send_notification(report)
            
            return report
            
        finally:
            await self.teardown()


def main():
    parser = argparse.ArgumentParser(description='CBB Edge Automated UAT')
    parser.add_argument('--base-url', required=True, help='Base URL of the application')
    parser.add_argument('--api-key', required=True, help='API key for authentication')
    parser.add_argument('--output', default='reports/uat_report.md', help='Output file path')
    parser.add_argument('--verbose', action='store_true', help='Run in verbose mode with visible browser')
    parser.add_argument('--json', action='store_true', help='Also output JSON format')
    
    args = parser.parse_args()
    
    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Run UAT
    executor = UATExecutor(
        base_url=args.base_url,
        api_key=args.api_key,
        verbose=args.verbose
    )
    
    report = asyncio.run(executor.run_full_uat())
    
    # Save markdown report
    markdown = report.generate_markdown()
    with open(output_path, 'w') as f:
        f.write(markdown)
    
    print(f"\n✅ UAT report saved to: {output_path}")
    
    # Save JSON if requested
    if args.json:
        json_path = output_path.with_suffix('.json')
        with open(json_path, 'w') as f:
            json.dump({
                'timestamp': report.timestamp,
                'base_url': report.base_url,
                'elite_fm_score': report.elite_fm_score,
                'quant_score': report.quant_score,
                'critical_issues': report.critical_issues,
                'results': [r.to_dict() for r in report.all_results],
                'executive_summary': report.executive_summary
            }, f, indent=2)
        print(f"✅ JSON report saved to: {json_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("UAT SUMMARY")
    print("="*60)
    print(report.executive_summary)
    
    # Exit code based on results
    if report.elite_fm_score < 6.0 or report.quant_score < 6.0:
        sys.exit(1)  # Fail
    elif report.elite_fm_score < 8.0 or report.quant_score < 8.0:
        sys.exit(2)  # Warning
    else:
        sys.exit(0)  # Success


if __name__ == '__main__':
    main()
