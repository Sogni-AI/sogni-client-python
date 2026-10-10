import pytest

from sogni_client.projects import create_job_request_message
from sogni_client.recovery import project_params_from_recovered_project


@pytest.mark.parametrize("expanded", [True, False, None])
def test_prompt_expanded_transport_and_recovery(expanded):
    params = {
        "type": "image",
        "modelId": "flux1-schnell-fp8",
        "positivePrompt": "A garden",
        "numberOfMedia": 1,
    }
    if expanded is not None:
        params["promptExpanded"] = expanded
    options = {
        "type": "image",
        "sampler": {"allowed": [], "default": None},
        "scheduler": {"allowed": [], "default": None},
    }
    request = create_job_request_message("expansion-project", params, options)
    recovered = project_params_from_recovered_project(
        {
            "model": {"id": params["modelId"]},
            **({"promptExpanded": expanded} if expanded is not None else {}),
        }
    )
    if expanded is None:
        assert "promptExpanded" not in request
        assert "promptExpanded" not in recovered
    else:
        assert request["promptExpanded"] is expanded
        assert recovered["promptExpanded"] is expanded
    assert "promptExpanded" not in request["keyFrames"][0]
