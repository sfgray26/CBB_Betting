"""
JobQueueService — PostgreSQL-backed async job queue for heavy operations.

Architecture: API layer submits jobs (returns job_id immediately).
APScheduler polls process_pending_jobs() every 5 seconds.
No Redis required — PostgreSQL is the broker.

ADR-005: All heavy operations (lineup optimization, player valuation) MUST
         go through this queue, never run synchronously in the request path.
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime
from zoneinfo import ZoneInfo

from sqlalchemy import text

from backend.contracts import LineupOptimizationRequest

logger = logging.getLogger(__name__)


def _now_et() -> datetime:
    return datetime.now(ZoneInfo("America/New_York"))


class RetryableJobError(RuntimeError):
    """Transient job failure that should be retried."""


class FatalJobError(RuntimeError):
    """Permanent job failure that should not be retried."""


class JobQueueService:

    def submit_job(
        self,
        db,
        job_type: str,
        payload: dict,
        priority: int = 5,
        league_key: str = None,
        team_key: str = None,
    ) -> str:
        job_id = str(uuid.uuid4())
        db.execute(
            text(
                """
                INSERT INTO job_queue
                    (id, job_type, payload, status, priority, created_at,
                     retry_count, max_retries, league_key, team_key)
                VALUES
                    (:id, :job_type, CAST(:payload AS JSONB), 'pending', :priority,
                     :created_at, 0, 3, :league_key, :team_key)
                """
            ),
            {
                "id": job_id,
                "job_type": job_type,
                "payload": json.dumps(payload),
                "priority": priority,
                "created_at": _now_et(),
                "league_key": league_key,
                "team_key": team_key,
            },
        )
        db.commit()
        return job_id

    def get_job_status(self, db, job_id: str) -> dict:
        row = db.execute(
            text(
                """
                SELECT id, status, job_type, created_at, completed_at,
                       result, error, retry_count
                FROM job_queue
                WHERE id = :job_id
                """
            ),
            {"job_id": job_id},
        ).fetchone()

        if row is None:
            return {"error": "not_found"}

        return {
            "job_id": str(row.id),
            "status": row.status,
            "job_type": row.job_type,
            "created_at": row.created_at,
            "completed_at": row.completed_at,
            "result": row.result,
            "error": row.error,
            "retry_count": row.retry_count,
        }

    async def process_pending_jobs(self, db) -> int:
        rows = db.execute(
            text(
                """
                SELECT id, job_type, payload, retry_count, max_retries
                FROM job_queue
                WHERE status = 'pending'
                ORDER BY priority ASC, created_at ASC
                LIMIT 3
                FOR UPDATE SKIP LOCKED
                """
            )
        ).fetchall()

        processed = 0
        for row in rows:
            job_id = str(row.id)
            db.execute(
                text(
                    """
                    UPDATE job_queue
                    SET status = 'processing', picked_at = :now
                    WHERE id = :job_id
                    """
                ),
                {"now": _now_et(), "job_id": job_id},
            )
            db.commit()

            try:
                result = await asyncio.to_thread(self._dispatch_job, db, row._asdict())
                # Gap 2 fix: treat logical errors (returned as dicts) as job failures
                if isinstance(result, dict) and result.get("status") == "error":
                    raise RetryableJobError(result.get("error", "Job returned error status"))
                db.execute(
                    text(
                        """
                        UPDATE job_queue
                        SET status = 'completed',
                            completed_at = :now,
                            result = CAST(:result AS jsonb)
                        WHERE id = :job_id
                        """
                    ),
                    {
                        "now": _now_et(),
                        "result": json.dumps(result),
                        "job_id": job_id,
                    },
                )
            except FatalJobError as exc:
                db.execute(
                    text(
                        """
                        UPDATE job_queue
                        SET status = 'failed',
                            completed_at = :now,
                            error = :error,
                            retry_count = :retry_count
                        WHERE id = :job_id
                        """
                    ),
                    {
                        "now": _now_et(),
                        "error": str(exc),
                        "retry_count": row.retry_count,
                        "job_id": job_id,
                    },
                )
                logger.warning("Job %s failed permanently: %s", job_id, exc)
            except RetryableJobError as exc:
                new_retry_count = row.retry_count + 1
                if new_retry_count >= row.max_retries:
                    db.execute(
                        text(
                            """
                            UPDATE job_queue
                            SET status = 'failed',
                                completed_at = :now,
                                error = :error,
                                retry_count = :retry_count
                            WHERE id = :job_id
                            """
                        ),
                        {
                            "now": _now_et(),
                            "error": str(exc),
                            "retry_count": new_retry_count,
                            "job_id": job_id,
                        },
                    )
                else:
                    db.execute(
                        text(
                            """
                            UPDATE job_queue
                            SET status = 'pending',
                                retry_count = :retry_count
                            WHERE id = :job_id
                            """
                        ),
                        {"retry_count": new_retry_count, "job_id": job_id},
                    )
                logger.warning("Job %s failed (attempt %d): %s", job_id, new_retry_count, exc)
            except Exception as exc:
                logger.error(
                    "Job %s raised unexpected exception (treated as fatal, no retry): %s: %s",
                    job_id,
                    type(exc).__name__,
                    exc,
                )
                db.execute(
                    text(
                        """
                        UPDATE job_queue
                        SET status = 'failed',
                            completed_at = :now,
                            error = :error,
                            retry_count = :retry_count
                        WHERE id = :job_id
                        """
                    ),
                    {
                        "now": _now_et(),
                        "error": f"[unexpected:{type(exc).__name__}] {exc}",
                        "retry_count": row.retry_count,
                        "job_id": job_id,
                    },
                )

            db.commit()
            processed += 1

        return processed

    def _dispatch_job(self, db, job_row: dict) -> dict:
        job_type = job_row["job_type"]
        if job_type == "lineup_optimization":
            payload = job_row["payload"]
            if isinstance(payload, str):
                try:
                    payload = json.loads(payload)
                except json.JSONDecodeError as exc:
                    raise FatalJobError(f"Invalid JSON payload: {exc}") from exc
            try:
                request = LineupOptimizationRequest.model_validate(payload)
            except Exception as exc:
                raise FatalJobError(f"Invalid lineup optimization payload: {exc}") from exc
            return self._run_lineup_optimization(db, request)
        raise FatalJobError(f"Unknown job_type: {job_type}")

    def _run_lineup_optimization(self, db, request: LineupOptimizationRequest) -> dict:
        # Raises on any failure — caller (process_pending_jobs) handles retry/fail logic
        from backend.fantasy_baseball.smart_lineup_selector import SmartLineupSelector, get_smart_selector  # noqa: PLC0415
        from backend.fantasy_baseball.projections_loader import load_full_board  # noqa: PLC0415
        from backend.fantasy_baseball.yahoo_client_resilient import get_resilient_yahoo_client  # noqa: PLC0415

        # Hydrate Yahoo roster at worker execution time (not at submission time)
        try:
            yahoo = get_resilient_yahoo_client()
            roster = yahoo.get_roster()
        except Exception as exc:
            raise RetryableJobError(f"Yahoo roster fetch failed: {exc}") from exc

        # Load projections (cached via lru_cache — fast after first call)
        try:
            projections = load_full_board()
        except Exception as exc:
            logger.warning("Projections board unavailable, proceeding with empty list: %s", exc)
            projections = []

        selector = get_smart_selector()
        try:
            assignments, warnings = selector.solve_smart_lineup(
                roster,
                projections,
                game_date=request.target_date,
            )
        except Exception as exc:
            raise RetryableJobError(f"Lineup optimization failed: {exc}") from exc
        return {
            "status": "ok",
            "target_date": request.target_date,
            "risk_tolerance": request.risk_tolerance.value,
            "lineup": assignments,
            "warnings": warnings,
        }


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------

def submit_job(
    db,
    job_type: str,
    payload: dict,
    priority: int = 5,
    league_key: str = None,
    team_key: str = None,
) -> str:
    return JobQueueService().submit_job(db, job_type, payload, priority, league_key, team_key)


def get_job_status(db, job_id: str) -> dict:
    return JobQueueService().get_job_status(db, job_id)


async def process_pending_jobs(db) -> int:
    return await JobQueueService().process_pending_jobs(db)
