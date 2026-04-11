#!/usr/bin/env python
"""Temporary admin endpoint for orphan linking on Railway"""
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from backend.models import get_db
from backend.fantasy_baseball.orphan_linker import link_orphaned_records
from sqlalchemy import text

router = APIRouter()

@router.post("/admin/temp-orphan-link")
async def run_orphan_linking(db: Session = Depends(get_db)):
    """Execute orphan linking and return results"""

    # Count before
    before = db.execute(text('''
        SELECT COUNT(*) FROM position_eligibility pe
        LEFT JOIN player_id_mapping pim ON pe.bdl_player_id = pim.id
        WHERE pe.yahoo_player_key IS NOT NULL AND pe.bdl_player_id IS NULL
    ''')).scalar()

    # Execute linking
    result = link_orphaned_records(db, dry_run=False, verbose=False)

    # Sample linked records
    sample = db.execute(text('''
        SELECT pe.player_name, pim.full_name, pe.bdl_player_id
        FROM position_eligibility pe
        JOIN player_id_mapping pim ON pe.bdl_player_id = pim.id
        WHERE pe.bdl_player_id IS NOT NULL
        ORDER BY pe.id DESC
        LIMIT 5
    ''')).fetchall()

    return {
        "before_count": before,
        "linked_count": result["linked_count"],
        "remaining_count": result["remaining_count"],
        "success_rate": result["success_rate"],
        "sample_records": [
            {"pe_name": row.player_name, "pim_name": row.full_name, "bdl_id": row.bdl_player_id}
            for row in sample
        ]
    }
