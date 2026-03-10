#!/usr/bin/env python3
"""
Recalibration Audit — P4

Validates that the recalibration service is functioning correctly:
1. Check if we have 30+ settled bets with prediction links
2. Verify recalibration ran and produced valid outputs
3. Alert if parameters are drifting or if recalibration is stuck
"""

import sys
import os
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def audit_recalibration():
    """Run full recalibration audit."""
    try:
        from backend.models import SessionLocal, BetLog, Prediction, ModelParameter
        from sqlalchemy import func
        
        db = SessionLocal()
        try:
            print("=" * 70)
            print("🔍 RECALIBRATION AUDIT")
            print("=" * 70)
            print(f"Audit time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
            print()
            
            # 1. Count settled bets with prediction links
            settled_with_pred = (
                db.query(BetLog)
                .filter(BetLog.outcome.isnot(None))
                .filter(BetLog.prediction_id.isnot(None))
                .count()
            )
            
            print(f"📊 Settled bets with prediction links: {settled_with_pred}")
            
            if settled_with_pred < 30:
                print(f"⚠️  WARNING: Only {settled_with_pred} bets (< 30 threshold)")
                print("   Recalibration needs more data to be reliable")
            else:
                print(f"✅ Sufficient data for recalibration ({settled_with_pred} bets)")
            
            print()
            
            # 2. Check recalibration history
            print("📈 Recent Recalibration Runs:")
            recent_recal = (
                db.query(ModelParameter)
                .filter(ModelParameter.parameter_name.in_([
                    'home_advantage', 'sd_multiplier', 
                    'recalibration_home_advantage', 'recalibration_sd_multiplier'
                ]))
                .order_by(ModelParameter.effective_date.desc())
                .limit(10)
                .all()
            )
            
            if not recent_recal:
                print("   ❌ No recalibration records found")
            else:
                for param in recent_recal:
                    print(f"   • {param.parameter_name}: {param.parameter_value:.4f}")
                    print(f"     Set: {param.effective_date.strftime('%Y-%m-%d %H:%M')}")
            
            print()
            
            # 3. Get current active parameters
            print("⚙️  Current Active Parameters:")
            
            current_ha = (
                db.query(ModelParameter)
                .filter(ModelParameter.parameter_name == 'home_advantage')
                .order_by(ModelParameter.effective_date.desc())
                .first()
            )
            
            current_sd = (
                db.query(ModelParameter)
                .filter(ModelParameter.parameter_name == 'sd_multiplier')
                .order_by(ModelParameter.effective_date.desc())
                .first()
            )
            
            ha_value = current_ha.parameter_value if current_ha else 3.09
            sd_value = current_sd.parameter_value if current_sd else 0.85
            
            print(f"   Home Advantage: {ha_value:.4f}")
            print(f"   SD Multiplier: {sd_value:.4f}")
            
            # 4. Check for drift from baselines
            baseline_ha = 3.09
            baseline_sd = 0.85
            
            ha_drift = abs(ha_value - baseline_ha) / baseline_ha * 100
            sd_drift = abs(sd_value - baseline_sd) / baseline_sd * 100
            
            print()
            print("📉 Drift from Baselines:")
            print(f"   Home Advantage: {ha_drift:.1f}% {'⚠️' if ha_drift > 15 else '✅'}")
            print(f"   SD Multiplier: {sd_drift:.1f}% {'⚠️' if sd_drift > 15 else '✅'}")
            
            if ha_drift > 15 or sd_drift > 15:
                print()
                print("🚨 ALERT: Parameters have drifted >15% from baseline")
                print("   Consider manual review before tournament")
            
            print()
            
            # 5. Check when last recalibration ran
            last_recal = (
                db.query(ModelParameter)
                .filter(ModelParameter.parameter_name == 'home_advantage')
                .order_by(ModelParameter.effective_date.desc())
                .first()
            )
            
            if last_recal:
                days_since = (datetime.utcnow() - last_recal.effective_date).days
                print(f"⏰ Last parameter update: {days_since} days ago")
                
                if days_since > 7:
                    print(f"   ⚠️  Recalibration may be stale (>7 days)")
                else:
                    print(f"   ✅ Recent enough (<7 days)")
            
            print()
            print("=" * 70)
            print("AUDIT COMPLETE")
            print("=" * 70)
            
            return {
                'settled_bets': settled_with_pred,
                'sufficient_data': settled_with_pred >= 30,
                'home_advantage': ha_value,
                'sd_multiplier': sd_value,
                'ha_drift_pct': ha_drift,
                'sd_drift_pct': sd_drift,
                'drift_alert': ha_drift > 15 or sd_drift > 15,
            }
            
        finally:
            db.close()
            
    except Exception as e:
        print(f"❌ Audit failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def generate_audit_report():
    """Generate markdown audit report."""
    result = audit_recalibration()
    
    if result is None:
        return
    
    report_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'reports',
        f'recalibration_audit_{datetime.utcnow().strftime("%Y%m%d")}.md'
    )
    
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    with open(report_path, 'w') as f:
        f.write("# Recalibration Audit Report\n\n")
        f.write(f"**Date:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n\n")
        
        f.write("## Summary\n\n")
        f.write(f"- **Settled Bets with Predictions:** {result['settled_bets']}\n")
        f.write(f"- **Sufficient Data (≥30):** {'✅ Yes' if result['sufficient_data'] else '❌ No'}\n")
        f.write(f"- **Home Advantage:** {result['home_advantage']:.4f}\n")
        f.write(f"- **SD Multiplier:** {result['sd_multiplier']:.4f}\n")
        f.write(f"- **HA Drift:** {result['ha_drift_pct']:.1f}%\n")
        f.write(f"- **SD Drift:** {result['sd_drift_pct']:.1f}%\n")
        f.write(f"- **Drift Alert:** {'🚨 Yes' if result['drift_alert'] else '✅ No'}\n\n")
        
        f.write("## Recommendations\n\n")
        
        if not result['sufficient_data']:
            f.write("1. **Collect more bets:** Need 30+ settled bets for reliable recalibration\n")
        
        if result['drift_alert']:
            f.write("2. **Parameter drift detected:** Review before tournament\n")
        else:
            f.write("2. **Parameters stable:** Within acceptable drift range\n")
        
        f.write("\n---\n\n")
        f.write("_Generated by P4 Recalibration Audit_\n")
    
    print(f"📄 Report saved to: {report_path}")


if __name__ == "__main__":
    audit_recalibration()
    print()
    generate_audit_report()
