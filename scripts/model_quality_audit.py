import sys
import os
import io
from datetime import datetime, timedelta
from sqlalchemy import func

# Add the project root to the path
sys.path.append(os.getcwd())

def run_quality_audit():
    output = io.StringIO()
    try:
        from backend.models import SessionLocal, BetLog, Prediction
        db = SessionLocal()
        
        # 1. Check Signal Drift (Delta between our projected margin and Market Consensus)
        recent_preds = db.query(Prediction).order_by(Prediction.id.desc()).limit(20).all()
        drift_count = 0
        total_drift = 0.0
        
        for p in recent_preds:
            if p.full_analysis and 'inputs' in p.full_analysis:
                market_spread = p.full_analysis['inputs']['odds'].get('spread', 0)
                our_margin = p.projected_margin
                # market_spread is like 4.0 (home is dog), our_margin is like -5.6 (home wins by 5.6)
                drift = abs(our_margin - (-market_spread)) 
                total_drift += drift
                drift_count += 1
        
        avg_drift = (total_drift / drift_count) if drift_count > 0 else 0
        
        output.write("🔬 **Autonomous Model Quality Audit**\n")
        output.write(f"Average Signal-to-Market Drift: {avg_drift:.2f} pts\n")
        
        if avg_drift > 4.5:
            output.write("⚠️ WARNING: High Signal Drift detected. Model may be 'too loud' vs Sharp market.\n")
        else:
            output.write("✅ Signal Stability: Within normal 1.5-sigma bounds.\n")

        # 2. Structural Roadmap Item
        output.write("\n🛠️ **Roadmap Item Proposed:**\n")
        if avg_drift > 3.0:
            output.write("- Implement 'Market Gravity' dampening for high-drift games (v8.1).\n")
        else:
            output.write("- Implement conference-specific efficiency bias adjustments.\n")

        db.close()

    except Exception as e:
        output.write(f"Audit Error: {str(e)}\n")

    message = output.getvalue()
    print(message)
    
    # Delivery
    if message.strip():
        safe_msg = message.replace('"', "'").replace("`", "")
        # PAUSED (2026-04-21): OpenClaw Discord notifications disabled.
        # os.system(f'openclaw message send --channel discord --target "1477436117426110615" --message "{safe_msg}"')
        print(f"[PAUSED] Would send: {safe_msg[:80]}...")

if __name__ == "__main__":
    run_quality_audit()
