from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from backend.models import DataIngestionLog

CRITICAL_CHAIN = {
    "mlb_game_log", "mlb_box_stats", "rolling_windows", "player_scores",
    "vorp", "player_momentum", "ros_simulation", "decision_optimization", "snapshot",
}

HOURLY_JOBS = {
    "bdl_injuries", "savant_ingestion",
}

DAILY_JOBS = {
    "fangraphs_ros", "yahoo_id_sync", "statcast_daily", "weekly_recalibration",
}

_FAILED_STATUSES = {"FAILED", "PARTIAL", "RUNNING"}
_SUCCESS_STATUSES = {"SUCCESS", "SKIPPED"}

_THRESHOLDS = {
    **{j: 4 for j in CRITICAL_CHAIN},
    **{j: 2 for j in HOURLY_JOBS},
    **{j: 26 for j in DAILY_JOBS},
}

_JOB_CLASS = {
    **{j: "critical_chain" for j in CRITICAL_CHAIN},
    **{j: "hourly" for j in HOURLY_JOBS},
    **{j: "daily" for j in DAILY_JOBS},
}

ALL_JOBS = CRITICAL_CHAIN | HOURLY_JOBS | DAILY_JOBS


def check_pipeline_health(db: Session) -> dict:
    now = datetime.now()

    counts = {"healthy": 0, "stale": 0, "failed": 0, "unknown": 0}
    jobs: dict[str, dict] = {}

    for job_type in sorted(ALL_JOBS):
        threshold_hours = _THRESHOLDS[job_type]
        cutoff = now - timedelta(hours=threshold_hours)

        latest = (
            db.query(DataIngestionLog)
            .filter(DataIngestionLog.job_type == job_type)
            .order_by(DataIngestionLog.started_at.desc())
            .first()
        )

        recent_success = (
            db.query(DataIngestionLog)
            .filter(
                DataIngestionLog.job_type == job_type,
                DataIngestionLog.status.in_(list(_SUCCESS_STATUSES)),
                DataIngestionLog.started_at >= cutoff,
            )
            .order_by(DataIngestionLog.started_at.desc())
            .first()
        )

        last_run_time = latest.started_at.isoformat() if latest else None
        last_status = latest.status if latest else None

        if latest is None:
            job_status = "unknown"
        elif latest.status in _FAILED_STATUSES:
            # Stuck RUNNING also counts as failed — priority over stale
            job_status = "failed"
        elif recent_success is None:
            job_status = "stale"
        else:
            job_status = "healthy"

        counts[job_status] = counts.get(job_status, 0) + 1

        jobs[job_type] = {
            "status": job_status,
            "last_run_time": last_run_time,
            "last_status": last_status,
            "threshold_hours": threshold_hours,
            "job_class": _JOB_CLASS[job_type],
        }

    return {
        "checked_at": now.replace(microsecond=0).isoformat(),
        "summary": {
            "healthy": counts["healthy"],
            "stale": counts["stale"],
            "failed": counts["failed"],
            "total": len(ALL_JOBS),
        },
        "jobs": jobs,
    }
