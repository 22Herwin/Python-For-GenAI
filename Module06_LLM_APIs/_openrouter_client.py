# Module 06 - OpenAI & Anthropic APIs
# Shared client factory - lets every 0X_*.py file in this folder run against
# either a real API key (ANTHROPIC_API_KEY / OPENAI_API_KEY) or a single
# OPENROUTER_API_KEY, with no other code changes needed in the calling file.
# OMNIROUTE_API_KEY takes precedence and routes both SDKs to the local gateway.
# OMNIROUTE_MODEL selects its provider-prefixed model; no upstream fallback is
# performed by this helper if the local gateway fails.
#
# OpenRouter only exposes an OpenAI-compatible endpoint (/v1/chat/completions),
# not Anthropic's native /v1/messages schema. get_anthropic_client() below
# returns the real `anthropic` SDK client when ANTHROPIC_API_KEY is set, or a
# small adapter that translates Anthropic-shaped calls (system prompt, tools,
# vision, streaming) into OpenAI-shaped ones when only OPENROUTER_API_KEY is
# set. Every file keeps calling client.messages.create(...) exactly as before.
#
# NOTE: count_tokens() (11_token_management.py) has no OpenRouter equivalent -
# it's a free Anthropic-only endpoint, so that file still needs a real
# ANTHROPIC_API_KEY and is not routed through this adapter.

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).with_name(".env"))

# Windows consoles default to a codepage (e.g. cp1252) that can't print every
# Unicode character a model might return (emoji, arrows, checkmarks). Force
# UTF-8 stdout so a print(response_text) never crashes with UnicodeEncodeError.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# An explicitly configured OmniRoute key takes precedence over other providers.
OMNIROUTE_BASE_URL = os.getenv("OMNIROUTE_BASE_URL", "http://127.0.0.1:20128/v1").rstrip("/")
_USE_OMNIROUTE = bool(os.getenv("OMNIROUTE_API_KEY"))
OMNIROUTE_MODEL = os.getenv("OMNIROUTE_MODEL", "cc/claude-sonnet-4-6")

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

_USE_OPENROUTER_FOR_ANTHROPIC = not os.getenv("ANTHROPIC_API_KEY") and bool(os.getenv("OPENROUTER_API_KEY"))
_USE_OPENROUTER_FOR_OPENAI = not os.getenv("OPENAI_API_KEY") and bool(os.getenv("OPENROUTER_API_KEY"))

# Model string to pass to client.messages.create(model=...) / chat.completions.create(model=...).
# Falls back to the real provider's own model id when a direct API key is set,
# so behaviour for anyone using a real key is unchanged.
MODEL = (
    os.getenv("OPENROUTER_MODEL", "anthropic/claude-sonnet-4.5")
    if _USE_OPENROUTER_FOR_ANTHROPIC
    else "claude-sonnet-4-5"
)
OPENAI_MODEL = (
    os.getenv("OPENROUTER_MODEL", "anthropic/claude-sonnet-4.5")
    if _USE_OPENROUTER_FOR_OPENAI
    else "gpt-4o"
)

if _USE_OMNIROUTE:
    MODEL = OPENAI_MODEL = OMNIROUTE_MODEL


def has_anthropic_access() -> bool:
    return _USE_OMNIROUTE or bool(os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENROUTER_API_KEY"))


def has_openai_access() -> bool:
    return _USE_OMNIROUTE or bool(os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY"))


def get_openai_client():
    """OpenAI SDK routed through OmniRoute, direct OpenAI, or OpenRouter."""
    from openai import OpenAI

    if _USE_OMNIROUTE:
        return OpenAI(api_key=os.environ["OMNIROUTE_API_KEY"], base_url=OMNIROUTE_BASE_URL,
                      timeout=120.0, max_retries=0)
    if os.getenv("OPENAI_API_KEY"):
        return OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    if os.getenv("OPENROUTER_API_KEY"):
        return OpenAI(api_key=os.environ["OPENROUTER_API_KEY"], base_url=OPENROUTER_BASE_URL)
    return None


def get_anthropic_client():
    """Anthropic SDK through OmniRoute/direct API, or the OpenRouter adapter."""
    if _USE_OMNIROUTE:
        import anthropic

        # Anthropic SDK appends /v1/messages itself.
        base_url = OMNIROUTE_BASE_URL.removesuffix("/v1")
        return anthropic.Anthropic(api_key=os.environ["OMNIROUTE_API_KEY"],
                                   base_url=base_url, timeout=120.0, max_retries=0)
    if os.getenv("ANTHROPIC_API_KEY"):
        import anthropic

        return anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    if os.getenv("OPENROUTER_API_KEY"):
        return _AnthropicViaOpenRouter(os.environ["OPENROUTER_API_KEY"])
    return None


# ── Anthropic-shaped adapter over OpenRouter's OpenAI-compatible API ─────────

class _Usage:
    def __init__(self, input_tokens: int, output_tokens: int):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class _TextBlock:
    def __init__(self, text: str):
        self.type = "text"
        self.text = text


class _ToolUseBlock:
    def __init__(self, block_id: str, name: str, tool_input: dict):
        self.type = "tool_use"
        self.id = block_id
        self.name = name
        self.input = tool_input


class _AnthropicMessage:
    def __init__(self, content, usage, stop_reason):
        self.content = content
        self.usage = usage
        self.stop_reason = stop_reason


_STOP_REASON_MAP = {"tool_calls": "tool_use", "stop": "end_turn", "length": "max_tokens"}


def _image_block_to_openai(block: dict) -> dict:
    source = block["source"]
    if source["type"] == "url":
        url = source["url"]
    else:
        url = f"data:{source['media_type']};base64,{source['data']}"
    return {"type": "image_url", "image_url": {"url": url}}


def _to_openai_messages(system, messages):
    """Translate Anthropic-shaped system/messages into an OpenAI messages list -
    including replaying a previous response.content (text/tool_use blocks)
    back in, and splitting tool_result blocks out into role="tool" messages."""
    openai_messages = []
    if system:
        openai_messages.append({"role": "system", "content": system})

    for msg in messages:
        role, content = msg["role"], msg["content"]

        if isinstance(content, str):
            openai_messages.append({"role": role, "content": content})
            continue

        text_parts, tool_calls, tool_results, image_parts = [], [], [], []
        for block in content:
            block_type = block["type"] if isinstance(block, dict) else block.type
            if block_type == "text":
                text_parts.append(block["text"] if isinstance(block, dict) else block.text)
            elif block_type == "image":
                image_parts.append(_image_block_to_openai(block))
            elif block_type == "tool_use":
                tool_calls.append({
                    "id": block["id"] if isinstance(block, dict) else block.id,
                    "type": "function",
                    "function": {
                        "name": block["name"] if isinstance(block, dict) else block.name,
                        "arguments": json.dumps(block["input"] if isinstance(block, dict) else block.input),
                    },
                })
            elif block_type == "tool_result":
                tool_results.append({
                    "role": "tool",
                    "tool_call_id": block["tool_use_id"],
                    "content": block["content"],
                })

        if tool_calls:
            openai_messages.append({
                "role": "assistant",
                "content": "".join(text_parts) or None,
                "tool_calls": tool_calls,
            })
        elif text_parts or image_parts:
            openai_messages.append({
                "role": role,
                "content": [*({"type": "text", "text": t} for t in text_parts), *image_parts],
            })
        openai_messages.extend(tool_results)

    return openai_messages


def _to_openai_tools(tools):
    if not tools:
        return None
    converted = []
    for tool in tools:
        if "input_schema" in tool:
            converted.append({
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool["input_schema"],
                },
            })
        else:
            converted.append(tool)
    return converted


class _StreamContext:
    def __init__(self, chunks):
        self._chunks = chunks
        self._text_parts = []

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    @property
    def text_stream(self):
        for chunk in self._chunks:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content
            if delta:
                self._text_parts.append(delta)
                yield delta

    def get_final_message(self):
        # OpenRouter doesn't reliably report token usage on streamed
        # responses, so this is a best-effort 0 rather than an exact count.
        return _AnthropicMessage([_TextBlock("".join(self._text_parts))], _Usage(0, 0), "end_turn")


class _AnthropicViaOpenRouter:
    def __init__(self, api_key: str):
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL)
        self.messages = self._Messages(self)

    class _Messages:
        def __init__(self, outer: "_AnthropicViaOpenRouter"):
            self._outer = outer

        def create(self, model, max_tokens, messages, system=None, tools=None, **_ignored):
            resp = self._outer._client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                messages=_to_openai_messages(system, messages),
                tools=_to_openai_tools(tools),
            )
            choice = resp.choices[0]
            content = []
            if choice.message.tool_calls:
                for call in choice.message.tool_calls:
                    content.append(_ToolUseBlock(call.id, call.function.name, json.loads(call.function.arguments)))
            if choice.message.content:
                content.append(_TextBlock(choice.message.content))
            usage = _Usage(resp.usage.prompt_tokens, resp.usage.completion_tokens)
            stop_reason = _STOP_REASON_MAP.get(choice.finish_reason, choice.finish_reason)
            return _AnthropicMessage(content, usage, stop_reason)

        def count_tokens(self, **kwargs):
            raise NotImplementedError(
                "count_tokens() is a free Anthropic-only endpoint with no OpenRouter "
                "equivalent - use a real ANTHROPIC_API_KEY for 11_token_management.py."
            )

        def stream(self, model, max_tokens, messages, system=None, **_ignored):
            chunks = self._outer._client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                messages=_to_openai_messages(system, messages),
                stream=True,
            )
            return _StreamContext(chunks)
