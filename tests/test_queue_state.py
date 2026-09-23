"""Queue descriptions remain independent from render progress and recovery timing."""

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from sogni_client.auth import ApiKeyAuthManager
from sogni_client.events import EventEmitter
from sogni_client.projects import Project, ProjectsApi
from sogni_client.queue_state import normalize_job_waiting_reasons, normalize_waiting_reason
from sogni_client.transport import WebSocketClient

WAIT = {"reason": "concurrency_limit", "message": "Waiting for another render to finish."}
MODEL_WAIT = {"reason": "model_concurrency_limit", "message": "Waiting for this model."}


@pytest.fixture
async def setup():
    client = EventEmitter()
    client.socket = EventEmitter()
    client.socket.app_id = "queue-test"
    api = ProjectsApi(client)
    project = Project(
        {"type": "image", "modelId": "test-model", "numberOfMedia": 3, "steps": 20}, api
    )
    api._projects.append(project)
    value = SimpleNamespace(api=api, client=client, project=project)
    yield value
    for tracked in {*api.tracked_projects, project}:
        tracked._dispose()
        if (
            tracked._completion is not None
            and tracked._completion.done()
            and not tracked._completion.cancelled()
        ):
            tracked._completion.exception()
    for task in tuple(api._background_tasks):
        task.cancel()
    await asyncio.sleep(0)


def entry(index, reason=WAIT, **extra):
    return {"jobIndex": index, "waitingReason": reason, **extra}


def queue(s, entries=None, reason=WAIT):
    s.client.socket.emit(
        "projectQueue",
        {
            "jobID": s.project.id,
            "waitingReason": reason,
            "jobWaitingReasons": entries if entries is not None else [entry(0)],
        },
    )


def pending_job(s, index=0, img_id="IMG-0"):
    return s.project._add_job(
        {
            "id": img_id,
            "projectId": s.project.id,
            "status": "pending",
            "jobIndex": index,
        }
    )


@pytest.mark.parametrize(
    "reason",
    ["concurrency_limit", "model_concurrency_limit", "payment_pending", "no_workers", "queued"],
)
def test_known_reason_and_public_fields_only(reason):
    raw = {
        "reason": reason,
        "message": "Current wait",
        "mediaType": "video",
        "paymentModel": "subscription",
        "subscriptionTier": "unlimited_pro",
        "modelFamily": "minimax_h3",
        "internalAccountId": "private",
        "activeCount": 4,
    }
    normalized = normalize_waiting_reason(raw)
    assert normalized == {
        key: value for key, value in raw.items() if key not in {"internalAccountId", "activeCount"}
    }


@pytest.mark.parametrize(
    "raw",
    [
        None,
        [],
        "queued",
        {},
        {"reason": "future_reason", "message": "wait"},
        {"reason": "queued", "message": " "},
        {"reason": "queued", "message": "a" * 601},
        {"reason": "queued", "message": 1},
    ],
)
def test_malformed_or_unknown_reason_is_neutral(raw):
    assert normalize_waiting_reason(raw) is None


def test_job_entries_are_bounded_unique_and_never_invent_ids():
    values = [
        entry(0.0),
        entry(0),
        entry(True),
        entry(-1),
        entry(1.5),
        entry(2, imgID="img-2"),
        entry(99),
        entry(3, waitingReason=None),
        entry(4, imgID="x" * 129),
    ]
    assert normalize_job_waiting_reasons(values, 9) == [entry(0), entry(2, imgID="img-2"), entry(4)]
    assert normalize_job_waiting_reasons([entry(0), entry(1), entry(2)], 2) == [entry(0), entry(1)]
    assert normalize_waiting_reason({**WAIT, "mediaType": "secret", "paymentModel": []}) == WAIT


def test_socket_default_and_explicit_opt_out():
    auth = ApiKeyAuthManager()
    default = WebSocketClient("wss://socket.example", auth, "test", "fast")
    opted_out = WebSocketClient(
        "wss://socket.example",
        auth,
        "test",
        "fast",
        socket_event_subscriptions={"projectQueue": False},
    )
    assert default.socket_event_subscriptions == {"projectQueue": True}
    assert opted_out.socket_event_subscriptions == {"projectQueue": False}


@pytest.mark.parametrize("initial", [True, False])
async def test_older_server_ack_keeps_queue_preference(initial):
    socket = WebSocketClient(
        "wss://socket.example",
        ApiKeyAuthManager(),
        "test",
        "fast",
        socket_event_subscriptions={"projectQueue": initial},
    )
    socket.emit("socketEventSubscriptionsUpdated", {"socketEventSubscriptions": {"appAlert": True}})
    assert socket.socket_event_subscriptions == {"appAlert": True, "projectQueue": initial}
    socket.emit(
        "socketEventSubscriptionsUpdated",
        {"socketEventSubscriptions": {"projectQueue": not initial}},
    )
    assert socket.socket_event_subscriptions["projectQueue"] is not initial


@pytest.mark.parametrize(
    "update, expected",
    [
        ({"projectQueue": False}, False),
        ({"event": "project_queue", "enabled": False}, False),
        ({"subscriptions": {"PROJECT-QUEUE": False}}, False),
        ({"unsubscribe": ["projectQueue"]}, False),
        ({"reset": True}, False),
        ({"reset": True, "subscribe": "projectQueue"}, True),
        ({"subscribe": "projectQueue", "unsubscribe": "projectQueue"}, False),
        ({"appAlert": True}, True),
    ],
)
async def test_dynamic_queue_intent_survives_older_server_ack(update, expected):
    socket = WebSocketClient("wss://socket.example", ApiKeyAuthManager(), "test", "fast")
    socket.send = AsyncMock()
    await socket.set_socket_event_subscriptions(update)
    socket.emit("socketEventSubscriptionsUpdated", {"socketEventSubscriptions": {}})
    assert socket.socket_event_subscriptions["projectQueue"] is expected


async def test_queue_entries_do_not_create_jobs_or_change_existing_event_shapes(setup):
    s = setup
    events, legacy = [], []
    s.api.on("queueChanged", events.append)
    s.api.on("project", legacy.append)
    queue(s, [entry(0), entry(2, MODEL_WAIT)])
    assert s.project.jobs == []
    assert s.project.status == "pending"
    assert legacy == []
    assert events == [
        {
            "projectId": s.project.id,
            "waitingReason": WAIT,
            "jobWaitingReasons": [entry(0), entry(2, MODEL_WAIT)],
        }
    ]
    queue(s, [entry(0), entry(2, MODEL_WAIT)])
    assert len(events) == 1
    assert s.project.toJSON()["jobWaitingReasons"] == s.project.job_waiting_reasons
    copy = s.project.jobWaitingReasons
    copy[0]["waitingReason"]["message"] = "mutated"
    assert s.project.waitingReason == WAIT


async def test_pending_job_matches_zero_index_without_id_and_omission_clears(setup):
    s = setup
    job = pending_job(s)
    queue(s, [entry(0), entry(1)])
    assert job.waitingReason == WAIT
    assert job.toJSON()["waitingReason"] == WAIT
    queue(s, [entry(1)])
    assert job.waiting_reason is None
    s.client.socket.emit("projectQueue", {"jobID": s.project.id, "waitingReason": None})
    assert s.project.waiting_reason is None
    assert s.project.job_waiting_reasons == []


@pytest.mark.parametrize("status", ["initiating", "processing", "completed", "failed", "canceled"])
async def test_live_job_lifecycle_clears_own_entry_preserving_other_waits(setup, status):
    s = setup
    job = pending_job(s)
    queue(s, [entry(0), entry(1, MODEL_WAIT)])
    job._update({"status": status})
    assert job.waiting_reason is None
    assert s.project.job_waiting_reasons == [entry(1, MODEL_WAIT)]
    assert s.project.waiting_reason == MODEL_WAIT
    queue(s, [entry(0), entry(1, MODEL_WAIT)])
    assert job.waiting_reason is None
    assert s.project.job_waiting_reasons == [entry(1, MODEL_WAIT)]


async def test_partial_processing_batch_keeps_progress_and_only_pending_reason(setup):
    s = setup
    running = pending_job(s)
    waiting = pending_job(s, 1, "IMG-1")
    s.client.socket.emit(
        "jobProgress", {"jobID": s.project.id, "imgID": running.id, "step": 7, "stepCount": 20}
    )
    queue(s, [entry(0), entry(1)])
    assert s.project.status == "processing"
    assert running.status == "processing" and running.step == 7
    assert running.waiting_reason is None
    assert waiting.waiting_reason == WAIT
    assert s.project.job_waiting_reasons == [entry(1)]


@pytest.mark.parametrize("status", ["completed", "failed", "canceled"])
async def test_terminal_parent_and_late_queue_frame_clear_all_state(setup, status):
    s = setup
    job = pending_job(s)
    queue(s)
    s.project._update({"status": status})
    queue(s)
    assert s.project.waiting_reason is None
    assert s.project.job_waiting_reasons == []
    assert job.waiting_reason is None


async def test_retry_removes_old_reason_until_fresh_queue_frame(setup):
    s = setup
    job = pending_job(s)
    queue(s)
    s.client.socket.emit("jobRetry", {"jobID": s.project.id, "imgID": job.id, "jobIndex": 0})
    assert job.status == "pending"
    assert job.waiting_reason is None
    assert s.project.job_waiting_reasons == []
    queue(s)
    assert job.waiting_reason == WAIT


async def test_legacy_queued_fields_and_unknown_project_are_safe(setup):
    s = setup
    s.client.socket.emit("projectQueue", {"jobID": "foreign", "waitingReason": WAIT})
    s.client.socket.emit("projectQueue", [])
    assert s.project.waiting_reason is None
    s.client.socket.emit(
        "jobState",
        {
            "type": "queued",
            "jobID": s.project.id,
            "waitingReason": WAIT,
            "jobWaitingReasons": [entry(0)],
            "queuePosition": 2,
        },
    )
    assert s.project.waiting_reason == WAIT
    s.client.socket.emit("jobState", {"type": "queued", "jobID": s.project.id})
    assert s.project.waiting_reason == WAIT
    s.client.socket.emit(
        "jobState",
        {"type": "queued", "jobID": s.project.id, "waitingReason": None, "jobWaitingReasons": []},
    )
    assert s.project.waiting_reason is None


@pytest.mark.parametrize("job_index", [0, None])
async def test_mixed_case_image_ids_are_preserved_but_match_existing_job(setup, job_index):
    s = setup
    job = pending_job(s, index=job_index, img_id="img-mixed")
    events = []
    s.api.on("queueChanged", events.append)
    queue(s, [entry(0, imgID="ImG-MiXeD")])
    assert job.waiting_reason == WAIT
    assert events[-1]["jobWaitingReasons"][0]["imgID"] == "ImG-MiXeD"
    job._update({"status": "processing"})
    assert s.project.job_waiting_reasons == []


async def test_recovery_applies_queue_before_assignment_and_keeps_job_index(setup):
    s = setup
    await s.api._replay_raw_project(
        s.project,
        {
            "status": "active",
            "waitingReason": WAIT,
            "jobWaitingReasons": [entry(0), entry(1)],
            "workerJobs": [{"imgID": "IMG-0", "jobIndex": 0, "status": "jobStarted"}],
        },
        True,
    )
    assert s.project.jobs[0].job_index == 0
    assert s.project.jobs[0].waiting_reason is None
    assert s.project.job_waiting_reasons == [entry(1)]


async def test_delayed_sync_snapshot_cannot_replace_newer_queue_message(setup):
    s = setup
    requested, release = asyncio.Event(), asyncio.Event()

    async def get(*args):
        requested.set()
        await release.wait()
        return {
            "activeProjects": [
                {
                    "id": s.project.id,
                    "status": "queued",
                    "waitingReason": WAIT,
                    "jobWaitingReasons": [entry(0)],
                }
            ]
        }

    s.client.socket.get = get
    task = asyncio.create_task(s.api.sync())
    await requested.wait()
    queue(s, [entry(1, MODEL_WAIT)], MODEL_WAIT)
    release.set()
    await task
    assert s.project.waiting_reason == MODEL_WAIT
    assert s.project.job_waiting_reasons == [entry(1, MODEL_WAIT)]


async def test_reconciliation_lock_does_not_allow_snapshot_to_restore_started_job(setup):
    s = setup
    job = pending_job(s)
    await s.api._sync_lock.acquire()
    task = asyncio.create_task(
        s.api._queue_sync(
            {
                "activeProjects": [
                    {
                        "id": s.project.id,
                        "status": "queued",
                        "waitingReason": WAIT,
                        "jobWaitingReasons": [entry(0)],
                    }
                ]
            },
            "manual",
            time.time(),
        )
    )
    await asyncio.sleep(0)
    s.client.socket.emit(
        "jobState", {"type": "jobStarted", "jobID": s.project.id, "imgID": job.id, "jobIndex": 0}
    )
    s.api._sync_lock.release()
    await task
    assert s.project.waiting_reason is None
    assert job.waiting_reason is None


async def test_rest_sync_uses_request_start_revision(setup):
    s = setup
    requested, release = asyncio.Event(), asyncio.Event()

    async def get(*args):
        requested.set()
        await release.wait()
        return {"waitingReason": WAIT, "jobWaitingReasons": [entry(0)]}

    s.api.get = get
    task = asyncio.create_task(s.project._sync_to_server())
    await requested.wait()
    queue(s, [entry(1, MODEL_WAIT)], MODEL_WAIT)
    release.set()
    await task
    assert s.project.waiting_reason == MODEL_WAIT
    assert s.project.job_waiting_reasons == [entry(1, MODEL_WAIT)]


async def test_account_change_drops_queue_state_and_old_ids(setup):
    s = setup
    queue(s)
    s.client.emit("sessionChanged")
    queue(s)
    assert s.api.tracked_projects == []
    assert s.project.waiting_reason is None


async def test_large_batch_keeps_every_queued_index_without_fake_jobs(setup):
    s = setup
    s.project._update({"params": {**s.project.params, "numberOfMedia": 30}})
    queue(s, [entry(index) for index in range(30)])
    assert len(s.project.job_waiting_reasons) == 30
    assert s.project.job_waiting_reasons[-1]["jobIndex"] == 29
    assert s.project.jobs == []


async def test_eta_frame_removes_wait_without_changing_existing_job_status(setup):
    s = setup
    job = pending_job(s)
    queue(s)
    s.client.socket.emit("jobETA", {"jobID": s.project.id, "imgID": job.id, "etaSeconds": 10})
    assert job.waiting_reason is None
    assert s.project.job_waiting_reasons == []
    assert job.eta_seconds == 10


async def test_result_receipt_invalidates_snapshot_before_download_finishes(setup):
    s = setup
    job = pending_job(s)
    queue(s)
    revision = s.project._queue_revision
    requested, release = asyncio.Event(), asyncio.Event()

    async def download(*args):
        requested.set()
        await release.wait()
        return "https://example.test/result.png"

    s.api.download_url = download
    s.client.socket.emit("jobResult", {"jobID": s.project.id, "imgID": job.id})
    await requested.wait()
    s.project._set_queue_state({"waitingReason": WAIT, "jobWaitingReasons": [entry(0)]}, revision)
    assert s.project.waiting_reason is None
    assert job.waiting_reason is None
    release.set()
    await asyncio.gather(*tuple(s.api._background_tasks))
    assert job.status == "completed"


async def test_authenticated_snapshot_captured_before_background_replay(setup):
    s = setup
    s.client.socket.emit(
        "authenticated",
        {
            "clientType": "artist",
            "activeProjects": [
                {
                    "id": s.project.id,
                    "status": "queued",
                    "waitingReason": WAIT,
                    "jobWaitingReasons": [entry(0)],
                }
            ],
        },
    )
    queue(s, [entry(1, MODEL_WAIT)], MODEL_WAIT)
    await asyncio.gather(*tuple(s.api._background_tasks))
    assert s.project.waiting_reason == MODEL_WAIT
    assert s.project.job_waiting_reasons == [entry(1, MODEL_WAIT)]


async def test_queue_only_clear_never_restarts_stopped_render_watchdog(setup):
    s = setup
    job = pending_job(s)
    job._update({"status": "processing"})
    job._stop_runtime_timeout()
    job._update({"waitingReason": None})
    assert job._runtime_timeout is None


async def test_disconnect_invalidates_old_snapshot_and_reconnect_can_restore(setup):
    s = setup
    queue(s)
    requested, release = asyncio.Event(), asyncio.Event()

    async def get(*args):
        requested.set()
        await release.wait()
        return {
            "activeProjects": [
                {
                    "id": s.project.id,
                    "status": "queued",
                    "waitingReason": WAIT,
                    "jobWaitingReasons": [entry(0)],
                }
            ]
        }

    s.client.socket.get = get
    task = asyncio.create_task(s.api.sync())
    await requested.wait()
    s.client.emit("connecting")
    assert s.project.waiting_reason is None
    release.set()
    await task
    assert s.project.waiting_reason is None
    await s.api.sync()
    assert s.project.waiting_reason == WAIT


async def test_account_switch_rejects_delayed_snapshot_before_queue_replay(setup):
    s = setup
    requested, release = asyncio.Event(), asyncio.Event()

    async def get(*args):
        requested.set()
        await release.wait()
        return {
            "activeProjects": [
                {
                    "id": s.project.id,
                    "status": "queued",
                    "waitingReason": WAIT,
                    "jobWaitingReasons": [entry(0)],
                }
            ]
        }

    s.client.socket.get = get
    task = asyncio.create_task(s.api.sync())
    await requested.wait()
    s.client.emit("sessionChanged")
    release.set()
    with pytest.raises(RuntimeError, match="account changed"):
        await task
    assert s.api.tracked_projects == []
    assert s.project.waiting_reason is None
