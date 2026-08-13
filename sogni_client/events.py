"""Small synchronous event primitive used by the public SDK entities."""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections import defaultdict
from collections.abc import Callable
from typing import Any, TypeVar

T = TypeVar("T")
Listener = Callable[[Any], Any]


class EventEmitter:
    """Node-style event emitter with removable listeners.

    Callbacks run synchronously. If a callback returns an awaitable, it is
    scheduled on the current event loop so socket readers are never blocked by
    user code.
    """

    def __init__(self) -> None:
        self._listeners: dict[str, list[Listener]] = defaultdict(list)

    def on(self, event: str, listener: Listener) -> Callable[[], None]:
        self._listeners[event].append(listener)

        def remove() -> None:
            self.off(event, listener)

        return remove

    def once(self, event: str, listener: Listener) -> Callable[[], None]:
        def wrapped(data: Any) -> Any:
            remove()
            return listener(data)

        remove = self.on(event, wrapped)
        return remove

    def off(self, event: str, listener: Listener) -> None:
        listeners = self._listeners.get(event)
        if not listeners:
            return
        self._listeners[event] = [candidate for candidate in listeners if candidate is not listener]
        if not self._listeners[event]:
            self._listeners.pop(event, None)

    def remove_all_listeners(self, event: str | None = None) -> None:
        if event is None:
            self._listeners.clear()
        else:
            self._listeners.pop(event, None)

    # JavaScript-compatible spelling for code translated from the JS SDK.
    removeAllListeners = remove_all_listeners

    def emit(self, event: str, data: Any = None) -> None:
        for listener in tuple(self._listeners.get(event, ())):
            try:
                result = listener(data)
                if inspect.isawaitable(result):
                    task = asyncio.create_task(result)
                    task.add_done_callback(self._log_task_error)
            except Exception:
                logging.getLogger("sogni_client").exception("Listener for %s failed", event)

    @staticmethod
    def _log_task_error(task: asyncio.Task[Any]) -> None:
        if task.cancelled():
            return
        error = task.exception()
        if error is not None:
            logging.getLogger("sogni_client").error(
                "Async event listener failed", exc_info=(type(error), error, error.__traceback__)
            )


class DataEntity(EventEmitter):
    """Mutable observable record with snapshot serialization."""

    def __init__(self, data: dict[str, Any]) -> None:
        super().__init__()
        self._data = dict(data)

    def _update(self, delta: dict[str, Any]) -> None:
        changed = [key for key, value in delta.items() if self._data.get(key) != value]
        if not changed:
            return
        self._data.update(delta)
        self.emit("updated", changed)

    def to_dict(self) -> dict[str, Any]:
        import copy

        return copy.deepcopy(self._data)

    def to_json(self) -> dict[str, Any]:
        return self.to_dict()

    toJSON = to_json
