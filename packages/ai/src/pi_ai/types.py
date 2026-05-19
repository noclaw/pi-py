from __future__ import annotations

import time
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

ThinkingLevel = Literal["minimal", "low", "medium", "high", "xhigh"]
ModelThinkingLevel = Literal["off", "minimal", "low", "medium", "high", "xhigh"]
StopReason = Literal["stop", "length", "toolUse", "error", "aborted"]
CacheRetention = Literal["none", "short", "long"]

THINKING_LEVELS: list[ModelThinkingLevel] = ["off", "minimal", "low", "medium", "high", "xhigh"]


# ── Content blocks ────────────────────────────────────────────────────────────

class TextContent(BaseModel):
    type: Literal["text"] = "text"
    text: str
    text_signature: Optional[str] = None


class ThinkingContent(BaseModel):
    type: Literal["thinking"] = "thinking"
    thinking: str
    thinking_signature: Optional[str] = None
    redacted: bool = False


class ImageContent(BaseModel):
    type: Literal["image"] = "image"
    data: str       # base64-encoded
    mime_type: str  # e.g. "image/jpeg"


class ToolCall(BaseModel):
    type: Literal["toolCall"] = "toolCall"
    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    thought_signature: Optional[str] = None


# ── Usage ─────────────────────────────────────────────────────────────────────

class UsageCost(BaseModel):
    input: float = 0.0
    output: float = 0.0
    cache_read: float = 0.0
    cache_write: float = 0.0
    total: float = 0.0


class Usage(BaseModel):
    input: int = 0
    output: int = 0
    cache_read: int = 0
    cache_write: int = 0
    total_tokens: int = 0
    cost: UsageCost = Field(default_factory=UsageCost)


# ── Messages ──────────────────────────────────────────────────────────────────

class UserMessage(BaseModel):
    role: Literal["user"] = "user"
    content: Union[str, list[Union[TextContent, ImageContent]]]
    timestamp: int = Field(default_factory=lambda: int(time.time() * 1000))


AssistantContent = Union[TextContent, ThinkingContent, ToolCall]


class AssistantMessage(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    role: Literal["assistant"] = "assistant"
    content: list[AssistantContent] = Field(default_factory=list)
    api: str
    provider: str
    model: str
    response_model: Optional[str] = None
    response_id: Optional[str] = None
    usage: Usage = Field(default_factory=Usage)
    stop_reason: StopReason = "stop"
    error_message: Optional[str] = None
    timestamp: int = Field(default_factory=lambda: int(time.time() * 1000))


class ToolResultMessage(BaseModel):
    role: Literal["toolResult"] = "toolResult"
    tool_call_id: str
    tool_name: str
    content: list[Union[TextContent, ImageContent]] = Field(default_factory=list)
    is_error: bool = False
    timestamp: int = Field(default_factory=lambda: int(time.time() * 1000))


Message = Union[UserMessage, AssistantMessage, ToolResultMessage]


# ── Tools & Context ───────────────────────────────────────────────────────────

class Tool(BaseModel):
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema


class Context(BaseModel):
    system_prompt: Optional[str] = None
    messages: list[Message]
    tools: Optional[list[Tool]] = None


# ── Model metadata ────────────────────────────────────────────────────────────

class ModelCost(BaseModel):
    input: float        # $/million tokens
    output: float
    cache_read: float = 0.0
    cache_write: float = 0.0


class OpenAICompletionsCompat(BaseModel):
    supports_store: bool = False
    supports_developer_role: bool = False
    supports_reasoning_effort: bool = False
    supports_usage_in_streaming: bool = True
    max_tokens_field: Literal["max_completion_tokens", "max_tokens"] = "max_completion_tokens"
    requires_tool_result_name: bool = False
    requires_assistant_after_tool_result: bool = False
    requires_thinking_as_text: bool = False
    requires_reasoning_content_on_assistant_messages: bool = False
    thinking_format: Optional[Literal["openai", "openrouter", "deepseek", "together", "zai", "qwen"]] = None
    cache_control_format: Optional[Literal["anthropic"]] = None
    send_session_affinity_headers: bool = False
    supports_long_cache_retention: bool = True
    supports_strict_mode: bool = True
    zai_tool_stream: bool = False


class AnthropicMessagesCompat(BaseModel):
    supports_eager_tool_input_streaming: bool = True
    supports_long_cache_retention: bool = True
    send_session_affinity_headers: bool = False
    supports_cache_control_on_tools: bool = True


class Model(BaseModel):
    id: str
    name: str
    api: str
    provider: str
    base_url: str
    reasoning: bool = False
    thinking_level_map: Optional[dict[str, Optional[str]]] = None
    input: list[Literal["text", "image"]] = Field(default_factory=lambda: ["text"])
    cost: ModelCost
    context_window: int
    max_tokens: int
    headers: Optional[dict[str, str]] = None
    compat: Optional[Union[OpenAICompletionsCompat, AnthropicMessagesCompat]] = None


# ── Image generation types ────────────────────────────────────────────────────

ImagesStopReason = Literal["stop", "error", "aborted"]


class ImagesModel(BaseModel):
    """Metadata for an image-generation model."""
    id: str
    name: str
    api: str            # e.g. "openrouter-images"
    provider: str
    base_url: str
    input: list[Literal["text", "image"]] = Field(default_factory=lambda: ["text"])
    output: list[Literal["text", "image"]] = Field(default_factory=lambda: ["image"])
    cost: ModelCost
    headers: Optional[dict[str, str]] = None


class ImagesContext(BaseModel):
    """Input context for image generation."""
    input: list[Union[TextContent, ImageContent]]


class AssistantImages(BaseModel):
    """Result of a generate_images() call."""
    api: str
    provider: str
    model: str
    output: list[Union[TextContent, ImageContent]] = Field(default_factory=list)
    response_id: Optional[str] = None
    usage: Optional[Usage] = None
    stop_reason: ImagesStopReason = "stop"
    error_message: Optional[str] = None
    timestamp: int = Field(default_factory=lambda: int(time.time() * 1000))


class ImagesOptions(BaseModel):
    """Options for generate_images()."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    api_key: Optional[str] = None
    signal: Optional[Any] = None
    on_payload: Optional[Any] = None   # async (payload, model) -> payload | None
    on_response: Optional[Any] = None  # async (response_info, model) -> None
    headers: Optional[dict[str, str]] = None
    timeout_ms: Optional[int] = None
    max_retries: Optional[int] = None


# ── Stream options ────────────────────────────────────────────────────────────

class StreamOptions(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    api_key: Optional[str] = None
    signal: Optional[Any] = None       # asyncio.Event — set it to abort the request
    cache_retention: CacheRetention = "short"
    session_id: Optional[str] = None
    on_payload: Optional[Any] = None   # async (payload, model) -> payload | None
    on_response: Optional[Any] = None  # async (response_info, model) -> None
    headers: Optional[dict[str, str]] = None
    timeout_ms: Optional[int] = None
    max_retries: Optional[int] = None
    metadata: Optional[dict[str, Any]] = None


class SimpleStreamOptions(StreamOptions):
    reasoning: Optional[ThinkingLevel] = None
    thinking_budgets: Optional[dict[str, int]] = None
