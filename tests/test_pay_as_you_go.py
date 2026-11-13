"""The pay-as-you-go offer and the Premium Spark switch, mirroring sogni-client's
``scripts/check-pay-as-you-go.cjs``."""

import asyncio
from types import SimpleNamespace

import pytest

from sogni_client.events import EventEmitter
from sogni_client.pay_as_you_go import (
    normalize_pay_as_you_go_offer,
    normalize_pay_as_you_go_switch_result,
)
from sogni_client.projects import Project, ProjectsApi

OFFER = {"available": True, "costInSpark": "12.5", "coveredByPremiumSpark": True}
QUEUED = {"reason": "queued", "message": "Queued."}


class SocketStub(EventEmitter):
    def __init__(self):
        super().__init__()
        self.app_id = "paygo-test"
        self.sent = []
        self.send_delay = 0.0
        self.fail_send = False

    async def send(self, message_type, data):
        if self.fail_send:
            raise ConnectionError("socket closed")
        self.sent.append((message_type, data))
        if self.send_delay:
            await asyncio.sleep(self.send_delay)


@pytest.fixture
async def setup():
    client = EventEmitter()
    client.socket = SocketStub()
    api = ProjectsApi(client)
    project = Project(
        {"type": "image", "modelId": "test-model", "numberOfMedia": 1, "steps": 4}, api
    )
    project._data["id"] = "PROJECT-1"
    api._projects.append(project)
    events = []
    api.on("queueChanged", events.append)
    value = SimpleNamespace(
        api=api, client=client, socket=client.socket, project=project, events=events
    )
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


def with_offer(s, offer=OFFER, rows=None):
    s.socket.emit(
        "projectQueue",
        {
            "jobID": "PROJECT-1",
            "waitingReason": QUEUED,
            "jobWaitingReasons": rows
            if rows is not None
            else [{"jobIndex": 0, "waitingReason": QUEUED}],
            "payAsYouGo": offer,
        },
    )


async def settle():
    for _ in range(5):
        await asyncio.sleep(0)


def reply(s, **data):
    s.socket.emit("projectPaymentSwitch", {"jobID": "PROJECT-1", **data})


# Normalization accepts only the documented shape.
def test_offer_normalization():
    assert normalize_pay_as_you_go_offer(OFFER) == OFFER
    assert normalize_pay_as_you_go_offer({**OFFER, "available": False}) is None
    assert normalize_pay_as_you_go_offer({**OFFER, "costInSpark": 12}) is None
    assert normalize_pay_as_you_go_offer({**OFFER, "costInSpark": "-1"}) is None
    assert normalize_pay_as_you_go_offer({**OFFER, "costInSpark": "1" * 65}) is None
    assert normalize_pay_as_you_go_offer(None) is None
    assert normalize_pay_as_you_go_offer([OFFER]) is None
    assert normalize_pay_as_you_go_offer({**OFFER, "coveredByPremiumSpark": "yes"}) == {
        **OFFER,
        "coveredByPremiumSpark": False,
    }


def test_switch_result_normalization():
    assert normalize_pay_as_you_go_switch_result(
        "P", {"status": "switched", "costInSpark": "12"}
    ) == {
        "projectId": "P",
        "status": "switched",
        "paymentModel": "paid_spark",
        "costInSpark": "12",
    }
    assert (
        normalize_pay_as_you_go_switch_result("P", {"status": "refused", "reason": "started"})[
            "reason"
        ]
        == "started"
    )
    # Unknown future reasons arrive as not_eligible.
    assert (
        normalize_pay_as_you_go_switch_result("P", {"status": "refused", "reason": "future"})[
            "reason"
        ]
        == "not_eligible"
    )
    # not_found never proves nothing was charged.
    assert normalize_pay_as_you_go_switch_result(
        "P", {"status": "refused", "reason": "not_found"}
    ) == {
        "projectId": "P",
        "status": "unknown",
        "reason": "not_found",
    }
    assert normalize_pay_as_you_go_switch_result("P", {"status": "later"})["status"] == "unknown"
    assert normalize_pay_as_you_go_switch_result(
        "P", {"status": "refused", "reason": "started", "message": " "}
    ) == {
        "projectId": "P",
        "status": "refused",
        "reason": "started",
    }


# A project without the field (older servers, or not offered) reports None.
async def test_no_offer_is_none(setup):
    s = setup
    s.socket.emit(
        "projectQueue", {"jobID": "PROJECT-1", "waitingReason": QUEUED, "jobWaitingReasons": []}
    )
    assert s.project.pay_as_you_go is None
    assert s.project.payAsYouGo is None


# The offer arrives with the queue state, then clears when the server drops it.
async def test_offer_arrives_and_clears(setup):
    s = setup
    with_offer(s)
    assert s.project.pay_as_you_go == OFFER
    assert s.events[-1]["payAsYouGo"] == OFFER
    assert len(s.events) == 1
    with_offer(s)
    assert len(s.events) == 1
    with_offer(s, offer=None)
    assert s.project.pay_as_you_go is None
    assert s.events[-1]["payAsYouGo"] is None
    assert len(s.events) == 2


# A started result ends the offer: the switch needs every result waiting.
async def test_started_result_clears_offer(setup):
    s = setup
    with_offer(s)
    s.project._add_job(
        {"id": "IMG-0", "projectId": "PROJECT-1", "status": "pending", "jobIndex": 0}
    )
    s.project._clear_job_waiting_reason("IMG-0", 0)
    assert s.project.pay_as_you_go is None
    assert s.events[-1]["payAsYouGo"] is None


# The switch sends one request, ignores other projects' replies, and clears the offer.
async def test_switch_success(setup):
    s = setup
    with_offer(s)
    task = asyncio.ensure_future(s.api.switch_to_premium_spark("PROJECT-1"))
    duplicate = asyncio.ensure_future(s.project.switch_to_premium_spark())
    await settle()
    assert s.socket.sent == [("switchProjectToPremiumSpark", {"projectId": "PROJECT-1"})]
    s.socket.emit("projectPaymentSwitch", {"jobID": "OTHER", "status": "switched"})
    await settle()
    assert not task.done()
    reply(s, status="switched", paymentModel="paid_spark", costInSpark="12.5")
    result = await task
    assert result == {
        "projectId": "PROJECT-1",
        "status": "switched",
        "paymentModel": "paid_spark",
        "costInSpark": "12.5",
    }
    assert await duplicate == result
    assert len(s.socket.sent) == 1
    assert s.project.pay_as_you_go is None
    assert s.events[-1]["payAsYouGo"] is None


async def test_reply_matches_id_case_insensitively(setup):
    s = setup
    task = asyncio.ensure_future(s.api.switch_to_premium_spark("project-1"))
    await settle()
    s.socket.emit("projectPaymentSwitch", {"jobID": "PROJECT-1", "status": "switched"})
    assert (await task)["status"] == "switched"


# A refusal keeps its reason and charges nothing; no answer resolves as a timeout.
async def test_refusal_and_timeout(setup):
    s = setup
    task = asyncio.ensure_future(s.api.switch_to_premium_spark("PROJECT-1"))
    await settle()
    reply(
        s,
        status="refused",
        reason="insufficient_premium_spark",
        costInSpark="12.5",
        message="Not enough.",
    )
    assert await task == {
        "projectId": "PROJECT-1",
        "status": "refused",
        "reason": "insufficient_premium_spark",
        "costInSpark": "12.5",
        "message": "Not enough.",
    }
    result = await s.api.switch_to_premium_spark("PROJECT-1", timeout=0.02)
    assert result["status"] == "unknown"
    assert result["reason"] == "timeout"


# An unknown server status is an unknown outcome, never a refusal.
async def test_unknown_status(setup):
    s = setup
    task = asyncio.ensure_future(s.api.switch_to_premium_spark("PROJECT-1"))
    await settle()
    reply(s, status="pending")
    assert await task == {
        "projectId": "PROJECT-1",
        "status": "unknown",
        "reason": "unrecognized_reply",
    }


# The timer starts once the request is sent, not while send() waits.
async def test_timer_starts_after_send(setup):
    s = setup
    s.socket.send_delay = 0.03
    task = asyncio.ensure_future(s.api.switch_to_premium_spark("PROJECT-1", timeout=0.04))
    await asyncio.sleep(0.05)
    reply(s, status="switched")
    assert (await task)["status"] == "switched"


# An invalid timeout falls back to the default instead of timing out at once.
@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf"), True])
async def test_invalid_timeout_uses_default(setup, timeout):
    s = setup
    task = asyncio.ensure_future(s.api.switch_to_premium_spark("PROJECT-1", timeout=timeout))
    await asyncio.sleep(0.02)
    assert not task.done()
    reply(s, status="switched")
    assert (await task)["status"] == "switched"


# A reconnect re-sends once from the new socket, and the timer restarts.
async def test_reconnect_resends_once(setup):
    s = setup
    task = asyncio.ensure_future(s.api.switch_to_premium_spark("PROJECT-1"))
    await settle()
    assert len(s.socket.sent) == 1
    s.client.emit("connecting")
    s.client.emit("connected")
    await settle()
    assert len(s.socket.sent) == 2
    s.client.emit("connecting")
    s.client.emit("connected")
    await settle()
    assert len(s.socket.sent) == 2
    reply(s, status="switched")
    assert (await task)["status"] == "switched"


# A connect that completes the first send is not a reconnect: no duplicate request.
async def test_connect_without_drop_sends_nothing(setup):
    s = setup
    task = asyncio.ensure_future(s.api.switch_to_premium_spark("PROJECT-1"))
    await settle()
    s.client.emit("connected")
    await settle()
    assert len(s.socket.sent) == 1
    reply(s, status="switched")
    await task


# A dropped connection clears the offer and says so in queueChanged.
async def test_transport_loss_clears_offer(setup):
    s = setup
    with_offer(s)
    s.client.emit("connecting")
    assert s.project.pay_as_you_go is None
    assert s.events[-1]["payAsYouGo"] is None
    assert s.events[-1]["waitingReason"] is None


async def test_transport_loss_clears_offer_without_reasons(setup):
    s = setup
    with_offer(s, rows=[])
    s.socket.emit(
        "projectQueue",
        {"jobID": "PROJECT-1", "waitingReason": None, "jobWaitingReasons": [], "payAsYouGo": OFFER},
    )
    count = len(s.events)
    s.client.emit("connecting")
    assert s.project.pay_as_you_go is None
    assert len(s.events) == count + 1


# A project finishing with only an offer still emits queueChanged.
async def test_finishing_project_clears_offer(setup):
    s = setup
    s.socket.emit(
        "projectQueue",
        {"jobID": "PROJECT-1", "waitingReason": None, "jobWaitingReasons": [], "payAsYouGo": OFFER},
    )
    count = len(s.events)
    s.project._update({"status": "completed"})
    assert s.project.pay_as_you_go is None
    assert len(s.events) == count + 1
    assert s.events[-1]["payAsYouGo"] is None


# Only 'started' and 'not_eligible' clear the offer locally; the server's next
# projectQueue update decides every other refusal.
@pytest.mark.parametrize(
    "reason",
    [
        "no_idle_worker",
        "concurrency_limit",
        "staff_only",
        "unavailable",
        "insufficient_premium_spark",
        "insufficient_funds",
        "payment_failed",
    ],
)
async def test_refusals_that_keep_the_offer(setup, reason):
    s = setup
    with_offer(s)
    task = asyncio.ensure_future(s.api.switch_to_premium_spark("PROJECT-1"))
    await settle()
    reply(s, status="refused", reason=reason)
    assert (await task)["reason"] == reason
    assert s.project.pay_as_you_go == OFFER


@pytest.mark.parametrize("reason", ["started", "not_eligible"])
async def test_refusals_that_clear_the_offer(setup, reason):
    s = setup
    with_offer(s)
    count = len(s.events)
    task = asyncio.ensure_future(s.api.switch_to_premium_spark("PROJECT-1"))
    await settle()
    reply(s, status="refused", reason=reason)
    assert (await task)["reason"] == reason
    assert s.project.pay_as_you_go is None
    assert len(s.events) == count + 1


# not_found never proves nothing was charged: it is always unknown, and keeps the offer.
async def test_not_found_is_unknown(setup):
    s = setup
    with_offer(s)
    task = asyncio.ensure_future(s.api.switch_to_premium_spark("PROJECT-1"))
    await settle()
    reply(s, status="refused", reason="not_found")
    assert await task == {"projectId": "PROJECT-1", "status": "unknown", "reason": "not_found"}
    assert s.project.pay_as_you_go == OFFER


# The first send switched, its reply went to the dropped socket, the project
# finished, and the re-send after the reconnect finds nothing: unknown, not refused.
async def test_resend_then_not_found_is_unknown(setup):
    s = setup
    task = asyncio.ensure_future(s.api.switch_to_premium_spark("PROJECT-1"))
    await settle()
    s.client.emit("connecting")
    s.client.emit("connected")
    await settle()
    assert len(s.socket.sent) == 2
    s.socket.emit(
        "projectPaymentSwitch", {"jobID": None, "status": "refused", "reason": "not_found"}
    )
    await settle()
    assert not task.done()
    reply(s, status="refused", reason="not_found")
    assert (await task)["status"] == "unknown"


# A session change settles a pending switch as unknown, never as a refusal.
async def test_session_change_is_unknown(setup):
    s = setup
    task = asyncio.ensure_future(s.api.switch_to_premium_spark("PROJECT-1"))
    await settle()
    s.client.emit("sessionChanged")
    assert await task == {
        "projectId": "PROJECT-1",
        "status": "unknown",
        "reason": "session_changed",
    }
    assert s.api._pay_as_you_go_switches == {}


# A request that certainly never left the client is the client's own `not_sent`
# refusal, never the server's `unavailable`, and it keeps the offer.
async def test_failed_send_is_not_sent(setup):
    s = setup
    with_offer(s)
    s.socket.fail_send = True
    assert await s.api.switch_to_premium_spark("PROJECT-1") == {
        "projectId": "PROJECT-1",
        "status": "refused",
        "reason": "not_sent",
    }
    assert s.project.pay_as_you_go == OFFER
    # The reply listener was removed: a late reply changes nothing.
    reply(s, status="switched")
    await settle()
    assert s.project.pay_as_you_go == OFFER


# Recovery snapshots carry no offer and never clear one.
async def test_recovery_snapshot_keeps_offer(setup):
    s = setup
    with_offer(s)
    s.project._set_queue_state({"waitingReason": QUEUED, "jobWaitingReasons": []})
    assert s.project.pay_as_you_go == OFFER
