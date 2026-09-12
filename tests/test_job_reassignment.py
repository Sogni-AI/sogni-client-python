"""A render the server moves to another worker stays in its own job.

Mirrors sogni-client scripts/check-job-reassignment.cjs.

The server moves a render on several paths: a worker that failed (announced
with ``jobRetry``), one that disconnected and never reclaimed its render, and a
personal LoRA that went away (both silent). Each time the new worker mints a NEW
``imgID``, which is what the SDK reports as the job id.

Two things break if the SDK treats that as a new render:

1. the project gains a job it never asked for, and
2. the abandoned attempt's job sits at ``processing`` with its runtime budget
   still running, and when that expires ``_handle_job_runtime_timeout`` sends
   ``artistCanceled`` to the server and cancels the project, retry included.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

import pytest

from sogni_client.events import EventEmitter
from sogni_client.projects import Project, ProjectsApi

pytestmark = pytest.mark.asyncio


class SocketStub(EventEmitter):
    def __init__(self) -> None:
        super().__init__()
        self.sent: list[tuple[str, Any]] = []

    async def send(self, message_type: str, data: Any) -> None:
        self.sent.append((message_type, data))


class ClientStub(EventEmitter):
    def __init__(self) -> None:
        super().__init__()
        self.socket = SocketStub()
        self.app_source = "pytest"


class Setup:
    def __init__(self, **params: Any) -> None:
        self.client = ClientStub()
        self.api = ProjectsApi(self.client)  # type: ignore[arg-type]
        self.project = Project(
            {
                "type": "video",
                "modelId": "test-model",
                "numberOfMedia": 1,
                "positivePrompt": "x",
                "network": "fast",
                **params,
            },
            self.api,
        )
        if self.project._timeout_handle is not None:
            self.project._timeout_handle.cancel()
            self.project._timeout_handle = None
        self.api._projects.append(self.project)
        self.job_events: list[dict[str, Any]] = []
        self.api.on("job", self.job_events.append)

    def started(self, img_id: str, job_index: int | None = 0) -> None:
        frame: dict[str, Any] = {
            "type": "jobStarted",
            "jobID": self.project.id,
            "imgID": img_id,
            "workerName": f"worker-{img_id}",
        }
        if job_index is not None:
            frame["jobIndex"] = job_index
        self.client.socket.emit("jobState", frame)

    def progress(self, img_id: str, step: int) -> None:
        self.client.socket.emit(
            "jobProgress",
            {"jobID": self.project.id, "imgID": img_id, "step": step, "stepCount": 20},
        )

    def retry(self, img_id: str, job_index: int = 0) -> None:
        self.client.socket.emit(
            "jobRetry",
            {
                "jobID": self.project.id,
                "imgID": img_id,
                "jobIndex": job_index,
                "attempt": 1,
                "maxAttempts": 1,
                "isFromWorker": True,
                "error": "genfailure",
                "error_message": "Generation failed",
            },
        )

    def cancels_sent(self) -> list[tuple[str, Any]]:
        return [
            (kind, data)
            for kind, data in self.client.socket.sent
            if kind == "jobError" and data.get("error") == "artistCanceled"
        ]


ScheduledTimer = tuple[float, Callable[..., Any], tuple[Any, ...]]


def capture_timers(monkeypatch: pytest.MonkeyPatch) -> list[ScheduledTimer]:
    """Record every ``call_later`` so a test can fire a budget directly.

    Firing the callback bypasses ``TimerHandle.cancel``, so the job's own guard
    is what has to hold.
    """

    loop = asyncio.get_running_loop()
    real_call_later = loop.call_later
    scheduled: list[ScheduledTimer] = []

    def call_later(delay: float, callback: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        scheduled.append((delay, callback, args))
        return real_call_later(delay, callback, *args, **kwargs)

    monkeypatch.setattr(loop, "call_later", call_later)
    return scheduled


def runtime_budgets(scheduled: list[ScheduledTimer]) -> list[ScheduledTimer]:
    return [timer for timer in scheduled if timer[0] >= 60 * 60]


async def settle() -> None:
    for _ in range(3):
        await asyncio.sleep(0)


async def test_silent_reassignment_stale_budget_never_cancels_the_project(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # THE BUG on the silent path that has shipped for months (a disconnect
    # reclaim): no jobRetry, just the next worker's frame at the same jobIndex.
    # The stale budget is fired directly, bypassing cancel, so the job's own
    # guard is what has to hold: the reclaimed job is `processing` again, so
    # status alone cannot tell the dead attempt's budget from the live one.
    s = Setup()
    scheduled = capture_timers(monkeypatch)

    s.started("IMG-FIRST", 0)
    [stale_budget] = runtime_budgets(scheduled)
    s.started("IMG-SECOND", 0)
    _delay, callback, args = stale_budget
    callback(*args)
    await settle()

    assert len(s.project.jobs) == 1
    assert s.project.jobs[0].status == "processing", "the live retry keeps rendering"
    assert s.cancels_sent() == [], "a stale budget must never cancel the project on the server"
    assert s.project.status != "failed"


async def test_announced_retry_keeps_a_single_media_project_alive() -> None:
    s = Setup()
    s.started("IMG-FIRST")
    job = s.project.jobs[0]
    assert job.status == "processing"
    assert job._runtime_timeout is not None, "a processing render arms its runtime budget"

    events_before = len(s.job_events)
    s.retry("IMG-FIRST")

    # Deliberately not surfaced as a job event: as a job error it would fail
    # this single-media project, the exact render the retry exists to save.
    assert len(s.job_events) == events_before, "jobRetry must not emit a job event"
    assert job.status == "pending"
    assert job.worker_name is None
    assert s.project.status != "failed"
    assert s.project.finished is False
    assert job._runtime_timeout is None, "the departed worker's runtime budget must stop"


async def test_announced_retry_stale_budget_never_cancels_the_project(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    s = Setup()
    scheduled = capture_timers(monkeypatch)

    s.started("IMG-FIRST")
    budgets = runtime_budgets(scheduled)
    assert budgets, "a Fast video render arms an hour-plus runtime budget"
    s.retry("IMG-FIRST")
    # Even if that timer had not been cleared, firing it now must do nothing.
    _delay, callback, args = budgets[0]
    callback(*args)
    await settle()

    assert s.cancels_sent() == [], "the stale budget must not cancel the project on the server"
    assert s.project.status != "failed"


async def test_retry_reclaims_the_same_job_under_its_new_id_from_a_clean_start() -> None:
    s = Setup()
    s.started("IMG-FIRST")
    s.progress("IMG-FIRST", 15)
    assert s.project.jobs[0].step == 15
    original = s.project.jobs[0]

    s.retry("IMG-FIRST")
    s.started("IMG-SECOND")

    assert len(s.project.jobs) == 1, "the retry must not add a second job"
    assert s.project.jobs[0] is original, "the retry reuses the same Job instance"
    assert original.id == "IMG-SECOND"
    assert original.job_index == 0
    assert original.jobIndex == 0
    assert s.project.job("IMG-FIRST") is None
    assert original.status == "processing"
    assert original.worker_name == "worker-IMG-SECOND"

    # step only moves forward, so without the reset the new attempt would be
    # pinned at the old attempt's high-water mark.
    s.progress("IMG-SECOND", 2)
    assert original.step == 2, "the new attempt must not inherit the old high-water mark"
    assert original._runtime_timeout is not None, "the retry arms its own fresh budget"


async def test_reclaim_clears_the_abandoned_attempts_worker_preview_progress_and_eta() -> None:
    s = Setup()
    s.started("IMG-FIRST")
    s.client.socket.emit(
        "jobProgress",
        {
            "jobID": s.project.id,
            "imgID": "IMG-FIRST",
            "step": 9,
            "stepCount": 20,
            "progress": 0.6,
            "etaMin": 10,
            "etaMax": 20,
        },
    )
    s.client.socket.emit("jobETA", {"jobID": s.project.id, "imgID": "IMG-FIRST", "etaSeconds": 30})
    job = s.project.jobs[0]
    job._update({"previewUrl": "https://cdn.example/preview.jpg"})

    s.retry("IMG-FIRST")

    for key in (
        "workerName",
        "previewUrl",
        "externalProgress",
        "eta",
        "etaStartedAt",
        "etaSeconds",
        "etaRange",
    ):
        assert job._data.get(key) is None, key
    assert job.step == 0
    assert job.status == "pending"


async def test_silent_reassignment_reclaims_by_job_index() -> None:
    # Disconnect reclaim / personal-LoRA requeue: no jobRetry at all, only the
    # next worker's frame with the same jobIndex.
    s = Setup()
    s.started("IMG-FIRST")
    original = s.project.jobs[0]
    s.progress("IMG-FIRST", 12)

    s.started("IMG-SECOND")

    assert len(s.project.jobs) == 1, "a silent reassignment must not add a second job"
    assert s.project.jobs[0] is original
    assert original.id == "IMG-SECOND"
    s.progress("IMG-SECOND", 1)
    assert original.step == 1


async def test_requeued_project_stops_every_running_render_budget() -> None:
    # Silent paths never send jobRetry, so the queue-wait window is closed by the
    # project going back to queued: nothing is on a worker any more.
    s = Setup()
    s.started("IMG-FIRST")
    assert s.project.jobs[0]._runtime_timeout is not None

    s.client.socket.emit("jobState", {"type": "queued", "jobID": s.project.id, "queuePosition": 4})

    assert s.project.jobs[0]._runtime_timeout is None, (
        "a re-queued project must not keep a render budget running"
    )
    assert s.project.status == "queued"


async def test_batch_renders_reclaim_their_own_jobs_by_index_in_any_order() -> None:
    s = Setup(numberOfMedia=3)
    s.started("IMG-A", 0)
    s.started("IMG-B", 1)
    s.started("IMG-C", 2)
    job_a, job_b, job_c = s.project.jobs

    s.retry("IMG-A", 0)
    s.retry("IMG-C", 2)
    assert job_b.status == "processing", "a sibling that never failed is untouched"

    s.started("IMG-C2", 2)
    s.started("IMG-A2", 0)

    assert len(s.project.jobs) == 3
    assert job_a.id == "IMG-A2"
    assert job_b.id == "IMG-B"
    assert job_c.id == "IMG-C2"


async def test_a_new_render_at_another_index_is_not_mistaken_for_a_reassignment() -> None:
    s = Setup(numberOfMedia=2)
    s.started("IMG-A", 0)
    s.started("IMG-B", 1)

    assert len(s.project.jobs) == 2, "a first attempt at another index is a new job"
    assert [job.id for job in s.project.jobs] == ["IMG-A", "IMG-B"]


async def test_without_job_index_only_the_single_announced_render_is_taken() -> None:
    s = Setup(numberOfMedia=2)
    s.started("IMG-A", None)
    s.started("IMG-B", None)
    assert len(s.project.jobs) == 2

    # Nothing announced: an unindexed new id is a new job, never a sibling.
    s.started("IMG-NEW", None)
    assert len(s.project.jobs) == 3
    job_a = s.project.job("IMG-A")
    assert job_a is not None and job_a.status == "processing"


async def test_without_job_index_the_one_announced_render_is_reclaimed() -> None:
    s = Setup(numberOfMedia=2)
    s.started("IMG-A", None)
    s.started("IMG-B", None)
    job_a = s.project.job("IMG-A")
    assert job_a is not None
    s.client.socket.emit(
        "jobRetry",
        {
            "jobID": s.project.id,
            "imgID": "IMG-A",
            "attempt": 1,
            "maxAttempts": 1,
            "isFromWorker": True,
            "error": "genfailure",
            "error_message": "Generation failed",
        },
    )

    s.started("IMG-A2", None)

    assert len(s.project.jobs) == 2
    assert job_a.id == "IMG-A2"
    assert job_a.status == "processing"


async def test_without_job_index_two_announced_renders_are_never_guessed_between() -> None:
    s = Setup(numberOfMedia=2)
    s.started("IMG-A", None)
    s.started("IMG-B", None)
    for img_id in ("IMG-A", "IMG-B"):
        s.client.socket.emit(
            "jobRetry",
            {
                "jobID": s.project.id,
                "imgID": img_id,
                "attempt": 1,
                "maxAttempts": 1,
                "isFromWorker": True,
                "error": "genfailure",
                "error_message": "Generation failed",
            },
        )

    s.started("IMG-X", None)

    assert len(s.project.jobs) == 3
    assert [job.id for job in s.project.jobs] == ["IMG-A", "IMG-B", "IMG-X"]


async def test_a_real_failure_after_a_retry_still_fails() -> None:
    s = Setup()
    s.started("IMG-FIRST")
    s.retry("IMG-FIRST")
    s.started("IMG-SECOND")
    s.client.socket.emit(
        "jobError",
        {
            "jobID": s.project.id,
            "imgID": "IMG-SECOND",
            "isFromWorker": True,
            "error": "genfailure",
            "error_message": "Generation failed",
        },
    )
    await settle()

    assert len(s.project.jobs) == 1
    assert s.project.jobs[0].status == "failed"
    assert s.project.status == "failed"
    with pytest.raises(Exception, match="Generation failed"):
        await s.project.wait_for_completion(timeout=1)


async def test_retry_then_completion_finishes_with_one_job() -> None:
    s = Setup()
    s.started("IMG-FIRST")
    s.retry("IMG-FIRST")
    s.started("IMG-SECOND")
    await s.api._apply_job_result(
        {
            "jobID": s.project.id,
            "imgID": "IMG-SECOND",
            "resultUrl": "https://cdn.example/result.mp4",
            "performedStepCount": 20,
        }
    )
    s.client.socket.emit("jobState", {"type": "jobCompleted", "jobID": s.project.id})

    assert len(s.project.jobs) == 1
    assert s.project.status == "completed"
    assert await s.project.wait_for_completion(timeout=1) == ["https://cdn.example/result.mp4"]


async def test_retry_frames_for_unknown_or_finished_projects_are_ignored() -> None:
    s = Setup()
    s.client.socket.emit(
        "jobRetry",
        {
            "jobID": "UNKNOWN",
            "imgID": "X",
            "attempt": 1,
            "maxAttempts": 1,
            "isFromWorker": True,
            "error": "genfailure",
            "error_message": "",
        },
    )
    s.project._update({"status": "completed"})
    s.retry("IMG-GONE")
    s.client.socket.emit("jobRetry", None)
    assert s.project.jobs == []
