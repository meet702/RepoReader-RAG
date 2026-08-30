import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

from agent.graph import build_agent
from agent.tools import build_tools
from code_graph.graph_store import load_graph
from evaluation.golden_dataset import TEST_CASES
from llm.provider import get_embeddings, message_content_to_text


QUESTIONS = {
    "what encoding should the readme file use",
    "what calls build_and_check_dists",
}

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def load_repo(repo_name: str):
    persist_directory = os.path.join(PROJECT_ROOT, "db", "chroma_db", repo_name)
    graph_path = os.path.join(PROJECT_ROOT, "db", "code_graph", f"{repo_name}.pkl")
    repo_path = os.path.join(PROJECT_ROOT, "cloned_repos", repo_name)

    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=get_embeddings(),
        collection_metadata={"hnsw:space": "cosine"},
    )
    collection_data = vectorstore.get()
    all_chunks = [
        Document(page_content=content, metadata=metadata)
        for content, metadata in zip(
            collection_data.get("documents", []),
            collection_data.get("metadatas", []),
        )
    ]
    graph = load_graph(graph_path) if os.path.exists(graph_path) else None
    return persist_directory, all_chunks, graph, repo_path


def main():
    provider = os.getenv("LLM_PROVIDER", "ollama")
    embedding_provider = os.getenv("EMBEDDING_PROVIDER", "ollama")
    print(f"LLM_PROVIDER={provider}")
    print(f"EMBEDDING_PROVIDER={embedding_provider}")

    for test_case in TEST_CASES:
        if test_case["question"] not in QUESTIONS:
            continue

        print("\n" + "=" * 80)
        print(f"QUESTION: {test_case['question']}")
        print(f"EXPECTED_TOOL: {test_case['expected_tool']}")
        print(f"EXPECTED_KEYWORDS: {test_case['expected_keywords']}")

        persist_directory, all_chunks, graph, repo_path = load_repo(test_case["repo_name"])
        tools = build_tools(persist_directory, all_chunks, graph, repo_path)
        agent = build_agent(tools)

        final_answer = ""
        messages = [HumanMessage(content=test_case["question"])]
        for index, step in enumerate(agent.stream({"messages": messages}, stream_mode="values"), start=1):
            last_msg = step["messages"][-1]
            msg_type = last_msg.__class__.__name__
            print(f"\nSTEP {index}: {msg_type}")

            if msg_type == "AIMessage" and getattr(last_msg, "tool_calls", None):
                for tool_call in last_msg.tool_calls:
                    print(f"TOOL_CALL: {tool_call['name']}")
                    print(f"TOOL_ARGS: {tool_call.get('args', {})}")
            elif msg_type == "ToolMessage":
                print(f"TOOL_NAME: {last_msg.name}")
                print("TOOL_OUTPUT:")
                print(message_content_to_text(last_msg.content))
            elif msg_type == "AIMessage":
                final_answer = message_content_to_text(last_msg.content)
                print("AI_CONTENT:")
                print(final_answer)
            else:
                print(message_content_to_text(getattr(last_msg, "content", last_msg)))

        print("\nFINAL_ANSWER:")
        print(final_answer)


if __name__ == "__main__":
    main()
