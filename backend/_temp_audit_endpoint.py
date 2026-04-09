"""Temporary audit endpoint for database tables - add to backend/main.py temporarily"""
from fastapi import APIRouter, Depends
from sqlalchemy import inspect, text
from backend.models import SessionLocal, engine

router = APIRouter()

@router.get("/admin/audit-tables")
def audit_tables():
    """Quick audit of all database tables - TEMPORARY ENDPOINT"""
    inspector = inspect(engine)
    tables = sorted(inspector.get_table_names())

    results = []
    db = SessionLocal()
    try:
        for table in tables:
            try:
                result = db.execute(text(f'SELECT COUNT(*) FROM "{table}"'))
                count = result.scalar()
                results.append({
                    'table': table,
                    'count': count,
                    'empty': count == 0
                })
            except Exception as e:
                results.append({
                    'table': table,
                    'count': 0,
                    'empty': False,
                    'error': str(e)[:100]
                })
    finally:
        db.close()

    # Categorize results
    empty = [r for r in results if r.get('empty') and not r.get('error')]
    populated = [r for r in results if not r.get('empty')]
    errors = [r for r in results if r.get('error')]

    return {
        'summary': {
            'total_tables': len(tables),
            'empty_tables': len(empty),
            'populated_tables': len(populated),
            'errors': len(errors)
        },
        'empty_tables': empty,
        'populated_tables': populated,
        'errors': errors
    }