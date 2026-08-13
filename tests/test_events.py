from __future__ import annotations

import asyncio

import pytest

from sogni_client.events import DataEntity, EventEmitter


def test_event_emitter_on_off_and_remover() -> None:
    emitter = EventEmitter()
    received: list[int] = []
    remove = emitter.on("value", received.append)

    emitter.emit("value", 1)
    remove()
    emitter.emit("value", 2)

    assert received == [1]


def test_event_emitter_once_removes_listener_before_invocation() -> None:
    emitter = EventEmitter()
    received: list[str] = []

    emitter.once("ready", received.append)
    emitter.emit("ready", "first")
    emitter.emit("ready", "second")

    assert received == ["first"]


def test_event_emitter_remove_all_listeners_supports_one_or_every_event() -> None:
    emitter = EventEmitter()
    received: list[str] = []
    emitter.on("a", lambda _data: received.append("a"))
    emitter.on("b", lambda _data: received.append("b"))

    emitter.remove_all_listeners("a")
    emitter.emit("a")
    emitter.emit("b")
    emitter.removeAllListeners()
    emitter.emit("b")

    assert received == ["b"]


@pytest.mark.asyncio
async def test_event_emitter_schedules_async_listeners_without_blocking_emit() -> None:
    emitter = EventEmitter()
    finished = asyncio.Event()

    async def listener(value: int) -> None:
        await asyncio.sleep(0)
        assert value == 7
        finished.set()

    emitter.on("value", listener)
    emitter.emit("value", 7)

    assert not finished.is_set()
    await asyncio.wait_for(finished.wait(), timeout=1)


def test_data_entity_emits_only_actually_changed_keys() -> None:
    entity = DataEntity({"status": "pending", "count": 1})
    updates: list[list[str]] = []
    entity.on("updated", updates.append)

    entity._update({"status": "pending"})
    entity._update({"status": "running", "count": 2})

    assert updates == [["status", "count"]]
    assert entity.to_dict() == {"status": "running", "count": 2}


def test_data_entity_snapshots_are_deep_copies_and_alias_matches() -> None:
    entity = DataEntity({"nested": {"items": [1]}})

    snapshot = entity.to_json()
    snapshot["nested"]["items"].append(2)

    assert entity.to_dict() == {"nested": {"items": [1]}}
    assert entity.toJSON() == {"nested": {"items": [1]}}
