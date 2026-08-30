import json
import os
import re

from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatResult
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_ollama import ChatOllama, OllamaEmbeddings

load_dotenv()


class QwenToolChatOllama(ChatOllama):
    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        result = super()._generate(messages, stop, run_manager, **kwargs)
        for gen in result.generations:
            msg = gen.message
            if isinstance(msg, AIMessage) and isinstance(msg.content, str):
                content = msg.content.strip()
                match = re.search(
                    r'\{[^{}]*"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*\{.*?\}\s*\}',
                    content,
                    re.DOTALL,
                )
                if match:
                    try:
                        tool_data = json.loads(match.group(0))
                    except Exception:
                        continue

                    if "name" in tool_data and "arguments" in tool_data:
                        msg.content = ""
                        msg.tool_calls = [
                            {
                                "name": tool_data["name"],
                                "args": tool_data["arguments"],
                                "id": "call_" + tool_data["name"],
                            }
                        ]
        return result


def get_llm():
    provider = os.getenv("LLM_PROVIDER", "ollama").lower()
    if provider == "ollama":
        return QwenToolChatOllama(model="qwen2.5-coder:7b")
    if provider == "gemini":
        return ChatGoogleGenerativeAI(
            model="gemini-3.6-flash",
            google_api_key=os.getenv("GEMINI_API_KEY"),
        )
    if provider == "gemini-lite":
        return ChatGoogleGenerativeAI(
            model="gemini-3.5-flash-lite",
            google_api_key=os.getenv("GEMINI_API_KEY"),
        )
    raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")


def get_embeddings():
    provider = os.getenv("EMBEDDING_PROVIDER", "ollama").lower()
    if provider == "ollama":
        return OllamaEmbeddings(model="nomic-embed-text")
    if provider == "gemini":
        return GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=os.getenv("GEMINI_API_KEY"),
        )
    raise ValueError(f"Unsupported EMBEDDING_PROVIDER: {provider}")


def message_content_to_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, dict) and isinstance(block.get("text"), str):
                text_parts.append(block["text"])
            elif isinstance(block, str):
                text_parts.append(block)
        return "\n".join(text_parts)
    if content is None:
        return ""
    return str(content)
