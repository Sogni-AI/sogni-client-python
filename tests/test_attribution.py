from __future__ import annotations

from sogni_client.attribution import (
    build_sogni_attribution_headers,
    connection_attribution_query,
    normalize_connection_attribution,
    resolve_workload_attribution,
    workload_attribution_to_wire_fields,
)


def test_connection_attribution_accepts_python_names_and_drops_invalid_values() -> None:
    normalized = normalize_connection_attribution(
        {
            "interaction_kind": "external_agent",
            "agent_framework": " codex ",
            "agent_framework_version": "5.21.1",
            "agent_surface": "sdk",
            "execution_mode": "server",
            "agent_surface_version": "not a version",
        }
    )

    assert normalized == {
        "interactionKind": "external_agent",
        "agentFramework": "codex",
        "agentFrameworkVersion": "5.21.1",
        "agentSurface": "sdk",
        "executionMode": "server",
    }
    assert connection_attribution_query(normalized) == normalized


def test_workload_attribution_merges_defaults_and_derives_lineage() -> None:
    resolved = resolve_workload_attribution(
        {
            "workloadKind": "agent_mediated",
            "agentFramework": "codex",
            "agentFrameworkVersion": "1.0",
            "agentSurface": "sdk",
        },
        {
            "agent_framework": "custom-agent",
            "agent_framework_version": None,
            "root_operation_id": "ROOT-1",
        },
        "CHILD-1",
    )

    assert resolved == {
        "workloadKind": "agent_mediated",
        "agentFramework": "custom-agent",
        "agentSurface": "sdk",
        "operationId": "CHILD-1",
        "rootOperationId": "ROOT-1",
        "operationScope": "child",
        "parentOperationId": "ROOT-1",
    }
    assert workload_attribution_to_wire_fields(resolved) == resolved


def test_attribution_headers_keep_connection_and_workload_metadata_separate() -> None:
    headers = build_sogni_attribution_headers(
        app_source=" python-sdk ",
        connection={"interactionKind": "human_ui", "agentFramework": "do-not-copy"},
        workload={
            "workloadKind": "agent_mediated",
            "agentFramework": "codex",
            "operationId": "OP-1",
        },
    )

    assert headers == {
        "X-App-Source": "python-sdk",
        "X-Sogni-Interaction-Kind": "human_ui",
        "X-Sogni-Workload-Kind": "agent_mediated",
        "X-Sogni-Agent-Framework": "codex",
        "X-Sogni-Operation-Id": "OP-1",
    }
