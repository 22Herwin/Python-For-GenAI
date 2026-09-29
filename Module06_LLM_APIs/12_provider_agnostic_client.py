# Module 06 - OpenAI & Anthropic APIs
# 6.7 Building a Provider-Agnostic Client
# When you want to swap between Anthropic and OpenAI without rewriting your
# pipeline, use an abstraction layer.
#
# NOTE: Instantiating AnthropicClient/OpenAIClient and calling .chat() makes a
# REAL, billed API call. The classes themselves can be imported and inspected
# without any API key.

from abc import ABC, abstractmethod
from dataclasses import dataclass
import os
from dotenv import load_dotenv

from _openrouter_client import get_anthropic_client, get_openai_client, has_anthropic_access, MODEL, OPENAI_MODEL

load_dotenv()


@dataclass
class ChatMessage:
    role: str      # "user" or "assistant"
    content: str


@dataclass
class ChatResponse:
    text: str
    input_tokens: int
    output_tokens: int
    model: str


class BaseLLMClient(ABC):
    @abstractmethod
    def chat(
        self,
        messages: list[ChatMessage],
        system: str = "",
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> ChatResponse: ...


class AnthropicClient(BaseLLMClient):
    def __init__(self, model: str = MODEL):
        self.model = model
        self._client = get_anthropic_client()

    def chat(self, messages, system="", max_tokens=1024, temperature=0.7) -> ChatResponse:
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": m.role, "content": m.content} for m in messages],
        )
        return ChatResponse(
            text=resp.content[0].text,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            model=self.model,
        )


class OpenAIClient(BaseLLMClient):
    def __init__(self, model: str = OPENAI_MODEL):
        self.model = model
        self._client = get_openai_client()

    def chat(self, messages, system="", max_tokens=1024, temperature=0.7) -> ChatResponse:
        api_messages = []
        if system:
            api_messages.append({"role": "system", "content": system})
        api_messages += [{"role": m.role, "content": m.content} for m in messages]

        resp = self._client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=api_messages,
        )
        return ChatResponse(
            text=resp.choices[0].message.content,
            input_tokens=resp.usage.prompt_tokens,
            output_tokens=resp.usage.completion_tokens,
            model=self.model,
        )


if __name__ == "__main__":
    # Same code, different backend - swap the class, nothing else changes.
    if not has_anthropic_access():
        print("ANTHROPIC_API_KEY / OPENROUTER_API_KEY is not set. Add one to a .env file in this folder.")
    else:
        client: BaseLLMClient = AnthropicClient()
        msgs = [ChatMessage(role="user", content="What is a vector database?")]
        result = client.chat(msgs, system="Be concise.")
        print(result.text)
        print(f"Cost estimate: {result.input_tokens} in, {result.output_tokens} out")
