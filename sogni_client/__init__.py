"""Async Python SDK for the Sogni Supernet."""

from .account import AccountApi, CurrentAccount
from .announcements import AnnouncementsApi
from .attribution import (
    build_sogni_attribution_headers,
    buildSogniAttributionHeaders,
    normalize_connection_attribution,
    normalizeConnectionAttribution,
    resolve_workload_attribution,
    resolveWorkloadAttribution,
    workload_attribution_to_wire_fields,
    workloadAttributionToWireFields,
)
from .auth import ApiKeyAuthManager, CookieAuthManager, TokenAuthManager
from .chat import (
    ChatApi,
    ChatStream,
    ChatToolsApi,
    SogniTools,
    is_sogni_tool_call,
    isSogniToolCall,
    parse_tool_call_arguments,
    parseToolCallArguments,
)
from .client import AsyncSogniClient, SogniClient
from .errors import (
    SUBSCRIPTION_ERROR_CODES,
    ApiError,
    ChatJobError,
    ProjectError,
    SogniError,
    is_subscription_limit_error,
    isSubscriptionLimitError,
)
from .events import DataEntity, EventEmitter
from .projects import (
    VIDEO_WORKFLOW_ASSETS,
    Job,
    Project,
    ProjectsApi,
    create_job_request_message,
)
from .recovery import (
    PROJECT_LOST_ERROR,
    PROJECT_LOST_ORIGINAL_CODE,
    is_project_lost_error,
    isProjectLostError,
    project_params_from_recovered_project,
    projectParamsFromRecoveredProject,
)
from .replay import ReplayApi
from .stats import StatsApi
from .utils import (
    PIXAL3D_IMAGE_TO_3D_MODEL_ID,
    SAM3_IMAGE_SEGMENT_MODEL_ID,
    calculate_video_frames,
    is_audio_model,
    is_model_artifact_model,
    is_segmentation_model,
    is_video_model,
    isAudioModel,
    isModelArtifactModel,
    isSegmentationModel,
    isVideoModel,
    parse_creative_workflow_sse_chunk,
    parseCreativeWorkflowSseChunk,
    requires_starting_image,
    requiresStartingImage,
)
from .workflows import (
    CREATIVE_WORKFLOW_WAITING_REASONS,
    CreativeWorkflowsApi,
    CreativeWorkflowTemplatesApi,
)

__version__ = "5.34.1"

__all__ = [
    "CREATIVE_WORKFLOW_WAITING_REASONS",
    "PIXAL3D_IMAGE_TO_3D_MODEL_ID",
    "PROJECT_LOST_ERROR",
    "PROJECT_LOST_ORIGINAL_CODE",
    "SAM3_IMAGE_SEGMENT_MODEL_ID",
    "SUBSCRIPTION_ERROR_CODES",
    "VIDEO_WORKFLOW_ASSETS",
    "AccountApi",
    "AnnouncementsApi",
    "ApiError",
    "ApiKeyAuthManager",
    "AsyncSogniClient",
    "ChatApi",
    "ChatJobError",
    "ChatStream",
    "ChatToolsApi",
    "CookieAuthManager",
    "CreativeWorkflowTemplatesApi",
    "CreativeWorkflowsApi",
    "CurrentAccount",
    "DataEntity",
    "EventEmitter",
    "Job",
    "Project",
    "ProjectError",
    "ProjectsApi",
    "ReplayApi",
    "SogniClient",
    "SogniError",
    "SogniTools",
    "StatsApi",
    "TokenAuthManager",
    "calculate_video_frames",
    "build_sogni_attribution_headers",
    "buildSogniAttributionHeaders",
    "create_job_request_message",
    "is_audio_model",
    "is_model_artifact_model",
    "is_project_lost_error",
    "is_segmentation_model",
    "is_sogni_tool_call",
    "is_subscription_limit_error",
    "is_video_model",
    "isAudioModel",
    "isModelArtifactModel",
    "isProjectLostError",
    "isSegmentationModel",
    "isSogniToolCall",
    "isSubscriptionLimitError",
    "isVideoModel",
    "normalize_connection_attribution",
    "normalizeConnectionAttribution",
    "parse_creative_workflow_sse_chunk",
    "parse_tool_call_arguments",
    "parseCreativeWorkflowSseChunk",
    "parseToolCallArguments",
    "project_params_from_recovered_project",
    "projectParamsFromRecoveredProject",
    "requires_starting_image",
    "requiresStartingImage",
    "resolve_workload_attribution",
    "resolveWorkloadAttribution",
    "workload_attribution_to_wire_fields",
    "workloadAttributionToWireFields",
]
