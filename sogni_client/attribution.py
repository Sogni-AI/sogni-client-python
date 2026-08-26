"""Connection and per-workload attribution helpers.

The wire contract mirrors the TypeScript SDK while accepting Python
``snake_case`` and JavaScript ``camelCase`` field names.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

_MAX_METADATA_LENGTH = 128
_MAX_VERSION_LENGTH = 32
_MAX_OPERATION_ID_LENGTH = 128
_UNSAFE_HEADER_CHARACTER = re.compile(r"[\x00-\x1f\x7f]")
_VERSION_PATTERN = re.compile(r"^[0-9][0-9A-Za-z.+_-]*$")
_OPERATION_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")

_INTERACTION_KINDS = {"human_ui", "external_agent", "service", "unknown"}
_WORKLOAD_KINDS = {"direct", "agent_mediated", "service", "unknown"}
_OPERATION_SCOPES = {"top_level", "child", "unknown"}
_AGENT_SURFACES = {
    "native_web",
    "native_mobile",
    "native_desktop",
    "plugin",
    "personal_skill",
    "mcp",
    "cli",
    "sdk",
    "openai_compatible",
    "direct_api",
    "unknown",
}
_EXECUTION_MODES = {"browser", "durable", "server", "unknown"}

_FIELD_ALIASES = {
    "interactionKind": "interaction_kind",
    "workloadKind": "workload_kind",
    "agentFramework": "agent_framework",
    "agentFrameworkVersion": "agent_framework_version",
    "agentSurface": "agent_surface",
    "agentSurfaceVersion": "agent_surface_version",
    "executionMode": "execution_mode",
    "operationScope": "operation_scope",
    "operationId": "operation_id",
    "rootOperationId": "root_operation_id",
    "parentOperationId": "parent_operation_id",
}


def _camel_values(value: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    result = dict(value)
    for camel, snake in _FIELD_ALIASES.items():
        if camel not in result and snake in result:
            result[camel] = result[snake]
    return result


def _bounded_string(value: Any, maximum: int) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    if not normalized or len(normalized) > maximum or _UNSAFE_HEADER_CHARACTER.search(normalized):
        return None
    return normalized


def _enum(value: Any, allowed: set[str]) -> str | None:
    normalized = value.strip() if isinstance(value, str) else None
    return normalized if normalized in allowed else None


def _version(value: Any) -> str | None:
    normalized = _bounded_string(value, _MAX_VERSION_LENGTH)
    return normalized if normalized and _VERSION_PATTERN.fullmatch(normalized) else None


def _operation_id(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    if (
        not value
        or len(value) > _MAX_OPERATION_ID_LENGTH
        or value.strip() != value
        or _UNSAFE_HEADER_CHARACTER.search(value)
        or not _OPERATION_ID_PATTERN.fullmatch(value)
    ):
        return None
    return value


def _agent_metadata(value: Mapping[str, Any] | None) -> dict[str, str]:
    source = _camel_values(value)
    candidates = {
        "agentFramework": _bounded_string(source.get("agentFramework"), _MAX_METADATA_LENGTH),
        "agentFrameworkVersion": _version(source.get("agentFrameworkVersion")),
        "agentSurface": _enum(source.get("agentSurface"), _AGENT_SURFACES),
        "agentSurfaceVersion": _version(source.get("agentSurfaceVersion")),
        "executionMode": _enum(source.get("executionMode"), _EXECUTION_MODES),
    }
    return {key: item for key, item in candidates.items() if item is not None}


def normalize_connection_attribution(value: Mapping[str, Any] | None) -> dict[str, str] | None:
    source = _camel_values(value)
    result = _agent_metadata(source)
    interaction = _enum(source.get("interactionKind"), _INTERACTION_KINDS)
    if interaction:
        result["interactionKind"] = interaction
    return result or None


def resolve_workload_attribution(
    defaults: Mapping[str, Any] | None,
    overrides: Mapping[str, Any] | None,
    fallback_operation_id: str | None = None,
) -> dict[str, str] | None:
    """Merge immutable defaults with one request and derive operation lineage."""

    if not defaults and not overrides:
        return None
    default_values = _camel_values(defaults)
    override_values = _camel_values(overrides)
    merged = dict(default_values)
    merged.update({key: value for key, value in override_values.items() if value is not None})
    if (
        "agentFramework" in override_values
        and override_values.get("agentFramework") != default_values.get("agentFramework")
        and override_values.get("agentFrameworkVersion") is None
    ):
        merged.pop("agentFrameworkVersion", None)
    if (
        "agentSurface" in override_values
        and override_values.get("agentSurface") != default_values.get("agentSurface")
        and override_values.get("agentSurfaceVersion") is None
    ):
        merged.pop("agentSurfaceVersion", None)

    result = _agent_metadata(merged)
    candidates = {
        "workloadKind": _enum(merged.get("workloadKind"), _WORKLOAD_KINDS),
        "operationScope": _enum(merged.get("operationScope"), _OPERATION_SCOPES),
        "operationId": _operation_id(merged.get("operationId")),
        "rootOperationId": _operation_id(merged.get("rootOperationId")),
        "parentOperationId": _operation_id(merged.get("parentOperationId")),
    }
    result.update({key: item for key, item in candidates.items() if item is not None})

    if result.get("workloadKind") and result["workloadKind"] != "agent_mediated":
        result.pop("agentFramework", None)
        result.pop("agentFrameworkVersion", None)
    if not result:
        return None
    if "operationId" not in result:
        fallback = _operation_id(fallback_operation_id)
        if fallback:
            result["operationId"] = fallback
    if "operationId" in result and "operationScope" not in result:
        result["operationScope"] = (
            "child"
            if result.get("parentOperationId")
            or (
                result.get("rootOperationId")
                and result.get("rootOperationId") != result.get("operationId")
            )
            else "top_level"
        )
    if (
        result.get("operationScope") == "child"
        and result.get("rootOperationId")
        and not result.get("parentOperationId")
    ):
        result["parentOperationId"] = result["rootOperationId"]
    if result.get("operationScope") == "top_level" and result.get("operationId"):
        result["rootOperationId"] = result["operationId"]
        result.pop("parentOperationId", None)
    return result


def connection_attribution_query(value: Mapping[str, Any] | None) -> dict[str, str]:
    normalized = normalize_connection_attribution(value)
    if not normalized:
        return {}
    fields = (
        "interactionKind",
        "agentFramework",
        "agentFrameworkVersion",
        "agentSurface",
        "agentSurfaceVersion",
        "executionMode",
    )
    return {field: normalized[field] for field in fields if field in normalized}


def workload_attribution_to_wire_fields(value: Mapping[str, Any] | None) -> dict[str, str]:
    if not value:
        return {}
    source = _camel_values(value)
    fields = (
        "workloadKind",
        "agentFramework",
        "agentFrameworkVersion",
        "agentSurface",
        "agentSurfaceVersion",
        "executionMode",
        "operationScope",
        "operationId",
        "rootOperationId",
        "parentOperationId",
    )
    return {field: source[field] for field in fields if isinstance(source.get(field), str)}


def build_sogni_attribution_headers(
    *,
    app_source: str | None = None,
    connection: Mapping[str, Any] | None = None,
    workload: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    headers: dict[str, str] = {}
    normalized_app_source = _bounded_string(app_source, _MAX_METADATA_LENGTH)
    if normalized_app_source:
        headers["X-App-Source"] = normalized_app_source
    connection_values = _camel_values(connection)
    workload_values = _camel_values(workload)
    header_fields = {
        "interactionKind": "X-Sogni-Interaction-Kind",
        "workloadKind": "X-Sogni-Workload-Kind",
        "agentFramework": "X-Sogni-Agent-Framework",
        "agentFrameworkVersion": "X-Sogni-Agent-Framework-Version",
        "agentSurface": "X-Sogni-Agent-Surface",
        "agentSurfaceVersion": "X-Sogni-Agent-Surface-Version",
        "executionMode": "X-Sogni-Execution-Mode",
        "operationScope": "X-Sogni-Operation-Scope",
        "operationId": "X-Sogni-Operation-Id",
        "rootOperationId": "X-Sogni-Root-Operation-Id",
        "parentOperationId": "X-Sogni-Parent-Operation-Id",
    }
    if isinstance(connection_values.get("interactionKind"), str):
        headers[header_fields["interactionKind"]] = connection_values["interactionKind"]
    for field, header in header_fields.items():
        if field != "interactionKind" and isinstance(workload_values.get(field), str):
            headers[header] = workload_values[field]
    return headers


# JavaScript-compatible aliases.
normalizeConnectionAttribution = normalize_connection_attribution
resolveWorkloadAttribution = resolve_workload_attribution
workloadAttributionToWireFields = workload_attribution_to_wire_fields
buildSogniAttributionHeaders = build_sogni_attribution_headers
