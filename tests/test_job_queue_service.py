import asyncio
import json
from collections import namedtuple
from unittest.mock import AsyncMock, MagicMock, patch

from backend.services.job_queue_service import (
    FatalJobError,
    JobQueueService,
    RetryableJobError,
)


JobRow = namedtuple("JobRow", ["id", "job_type", "payload", "retry_count", "max_retries"])


def _run(coro):
    return asyncio.run(coro)


def test_dispatch_job_rejects_invalid_payload_immediately():
    service = JobQueueService()

    with patch("backend.services.job_queue_service.LineupOptimizationRequest.model_validate", side_effect=ValueError("missing fields")):
        try:
            service._dispatch_job(MagicMock(), {
                "job_type": "lineup_optimization",
                "payload": {"target_date": "2026-04-03"},
            })
        except FatalJobError as exc:
            assert "Invalid lineup optimization payload" in str(exc)
        else:
            raise AssertionError("Expected FatalJobError for invalid payload")


def test_process_pending_jobs_requeues_retryable_failure():
    service = JobQueueService()
    db = MagicMock()
    db.execute.return_value.fetchall.return_value = [
        JobRow("job-1", "lineup_optimization", json.dumps({}), 0, 3)
    ]

    with patch.object(service, "_dispatch_job", side_effect=RetryableJobError("network timeout")), \
         patch("backend.services.job_queue_service.asyncio.to_thread", new=AsyncMock(side_effect=lambda func, *args: func(*args))):
        processed = _run(service.process_pending_jobs(db))

    assert processed == 1
    pending_update = next(
        call for call in db.execute.call_args_list
        if len(call.args) > 1 and call.kwargs == {} and call.args[1].get("job_id") == "job-1" and call.args[1].get("retry_count") == 1
    )
    assert pending_update.args[1]["retry_count"] == 1


def test_process_pending_jobs_fails_fatal_error_without_retry_increment():
    service = JobQueueService()
    db = MagicMock()
    db.execute.return_value.fetchall.return_value = [
        JobRow("job-2", "lineup_optimization", json.dumps({}), 0, 3)
    ]

    with patch.object(service, "_dispatch_job", side_effect=FatalJobError("invalid payload")), \
         patch("backend.services.job_queue_service.asyncio.to_thread", new=AsyncMock(side_effect=lambda func, *args: func(*args))):
        processed = _run(service.process_pending_jobs(db))

    assert processed == 1
    failed_update = next(
        call for call in db.execute.call_args_list
        if len(call.args) > 1 and call.kwargs == {} and call.args[1].get("job_id") == "job-2" and call.args[1].get("error") == "invalid payload"
    )
    assert failed_update.args[1]["retry_count"] == 0


def test_process_pending_jobs_marks_retryable_failure_terminal_at_max_retries():
    service = JobQueueService()
    db = MagicMock()
    db.execute.return_value.fetchall.return_value = [
        JobRow("job-3", "lineup_optimization", json.dumps({}), 2, 3)
    ]

    with patch.object(service, "_dispatch_job", side_effect=RetryableJobError("temporary yahoo outage")), \
         patch("backend.services.job_queue_service.asyncio.to_thread", new=AsyncMock(side_effect=lambda func, *args: func(*args))):
        processed = _run(service.process_pending_jobs(db))

    assert processed == 1
    failed_update = next(
        call for call in db.execute.call_args_list
        if len(call.args) > 1 and call.kwargs == {} and call.args[1].get("job_id") == "job-3" and call.args[1].get("error") == "temporary yahoo outage"
    )
    assert failed_update.args[1]["retry_count"] == 3


def test_process_pending_jobs_marks_unknown_job_type_fatal():
    service = JobQueueService()
    db = MagicMock()
    db.execute.return_value.fetchall.return_value = [
        JobRow("job-4", "unknown_job", json.dumps({}), 0, 3)
    ]

    with patch("backend.services.job_queue_service.asyncio.to_thread", new=AsyncMock(side_effect=lambda func, *args: func(*args))):
        processed = _run(service.process_pending_jobs(db))

    assert processed == 1
    failed_update = next(
        call for call in db.execute.call_args_list
        if len(call.args) > 1 and call.kwargs == {} and call.args[1].get("job_id") == "job-4" and "Unknown job_type" in call.args[1].get("error", "")
    )
    assert failed_update.args[1]["retry_count"] == 0
