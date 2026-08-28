import os
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_ollama import OllamaEmbeddings

from agent.graph import build_agent
from agent.tools import build_tools
from code_graph.graph_store import load_graph
from evaluation.golden_dataset import TEST_CASES


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ATTEMPTS_PER_CASE = 3
_REPO_CACHE = {}


def load_repo(repo_name: str) -> tuple[str, list[Document], object | None]:
    if repo_name in _REPO_CACHE:
        return _REPO_CACHE[repo_name]

    persist_directory = os.path.join(PROJECT_ROOT, "db", "chroma_db", repo_name)
    graph_path = os.path.join(PROJECT_ROOT, "db", "code_graph", f"{repo_name}.pkl")

    if not os.path.exists(persist_directory):
        raise FileNotFoundError(f"No ChromaDB found at {persist_directory}")

    embedding_model = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model,
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
    repo_state = (persist_directory, all_chunks, graph)
    _REPO_CACHE[repo_name] = repo_state
    return repo_state


def run_agent(repo_name: str, question: str) -> dict:
    persist_directory, all_chunks, graph = load_repo(repo_name)
    tools = build_tools(persist_directory, all_chunks, graph)
    agent = build_agent(tools)

    tool_names = []
    tool_outputs = []
    final_answer = ""

    messages = [HumanMessage(content=question)]
    for step in agent.stream({"messages": messages}, stream_mode="values"):
        last_msg = step["messages"][-1]
        msg_type = last_msg.__class__.__name__

        if msg_type == "ToolMessage":
            tool_outputs.append(str(last_msg.content))
        elif msg_type == "AIMessage" and getattr(last_msg, "tool_calls", None):
            for tool_call in last_msg.tool_calls:
                tool_names.append(tool_call["name"])
        elif msg_type == "AIMessage" and last_msg.content:
            final_answer = last_msg.content

    return {
        "tool_names": tool_names,
        "tool_output": "\n".join(tool_outputs),
        "final_answer": final_answer,
    }


def format_question(question: str, width: int = 42) -> str:
    if len(question) <= width:
        return question.ljust(width)
    return question[: width - 3] + "..."


def format_keyword_results(keyword_matches: list[tuple[str, bool]]) -> str:
    return " ".join(f"{'✅' if found else '❌'} {keyword}" for keyword, found in keyword_matches)


def evaluate_attempt(test_case: dict) -> dict:
    started_at = time.perf_counter()

    try:
        agent_result = run_agent(test_case["repo_name"], test_case["question"])
        error = None
    except Exception as exc:
        agent_result = {"tool_names": [], "tool_output": "", "final_answer": ""}
        error = str(exc)

    elapsed = time.perf_counter() - started_at

    actual_tools = agent_result["tool_names"]
    first_tool = actual_tools[0] if actual_tools else "None"
    combined_text = f"{agent_result['tool_output']}\n{agent_result['final_answer']}".lower()
    keyword_matches = [
        (keyword, keyword.lower() in combined_text)
        for keyword in test_case["expected_keywords"]
    ]

    expected_tool = test_case["expected_tool"]
    tool_match = first_tool == expected_tool
    expected_tool_called = expected_tool in actual_tools
    keywords_match = all(found for _, found in keyword_matches)

    return {
        "actual_tools": actual_tools,
        "tool_match": tool_match,
        "expected_tool_called": expected_tool_called,
        "keyword_matches": keyword_matches,
        "keywords_match": keywords_match,
        "passed": tool_match and keywords_match,
        "elapsed": elapsed,
        "error": error,
    }


def summarize_tool_consistency(attempts: list[dict], expected_tool: str) -> str:
    called_count = sum(1 for attempt in attempts if attempt["expected_tool_called"])

    if called_count == len(attempts):
        return f"✅ consistent: tool called in {called_count}/{len(attempts)} attempts"
    if called_count == 0:
        return f"❌ skipped: tool called in {called_count}/{len(attempts)} attempts"
    return f"⚠️ inconsistent: tool called in {called_count}/{len(attempts)} attempts"


def main() -> None:
    total_runs = len(TEST_CASES) * ATTEMPTS_PER_CASE
    print(
        f"Running {len(TEST_CASES)} evaluation cases "
        f"({ATTEMPTS_PER_CASE} attempts each, {total_runs} total runs)...\n"
    )

    results = []
    for index, test_case in enumerate(TEST_CASES, start=1):
        repo_name = test_case["repo_name"]
        question = test_case["question"]
        expected_tool = test_case["expected_tool"]

        print(f"[{index}/{len(TEST_CASES)}] {repo_name}: {question}")
        attempts = []
        for attempt_number in range(1, ATTEMPTS_PER_CASE + 1):
            attempt = evaluate_attempt(test_case)
            attempts.append(attempt)
            status = "PASS" if attempt["passed"] else "FAIL"
            print(
                f"  Attempt {attempt_number}/{ATTEMPTS_PER_CASE}: {status} "
                f"tools={attempt['actual_tools'] or ['None']} "
                f"time={attempt['elapsed']:.2f}s"
            )

        results.append(
            {
                "question": question,
                "expected_tool": expected_tool,
                "attempts": attempts,
            }
        )

    print()
    print("QUESTION                                   | PASS RATE | TOOL CONSISTENCY | ATTEMPTS")
    print("-" * 120)

    all_passed_count = 0
    some_passed_count = 0
    failed_all_count = 0
    for result in results:
        attempts = result["attempts"]
        passed_attempts = sum(1 for attempt in attempts if attempt["passed"])

        if passed_attempts == len(attempts):
            all_passed_count += 1
        elif passed_attempts > 0:
            some_passed_count += 1
        else:
            failed_all_count += 1

        pass_rate = f"{passed_attempts}/{len(attempts)} attempts passed"
        tool_consistency = summarize_tool_consistency(attempts, result["expected_tool"])
        attempt_summary = " ".join(
            f"{idx}:{'✅' if attempt['passed'] else '❌'}"
            for idx, attempt in enumerate(attempts, start=1)
        )
        print(
            f"{format_question(result['question'])} | "
            f"{pass_rate:<19} | "
            f"{tool_consistency:<48} | "
            f"{attempt_summary}"
        )

        for idx, attempt in enumerate(attempts, start=1):
            keyword_text = format_keyword_results(attempt["keyword_matches"])
            print(
                f"  Attempt {idx}: "
                f"tools={attempt['actual_tools'] or ['None']} "
                f"keywords={keyword_text} "
                f"time={attempt['elapsed']:.2f}s"
            )
            if attempt["error"]:
                print(f"    Error: {attempt['error']}")
            elif not attempt["tool_match"]:
                print(f"    Expected first tool: {result['expected_tool']}")

    print("-" * 120)
    print(
        "SUMMARY: "
        f"{all_passed_count}/{len(results)} test cases passed ALL {ATTEMPTS_PER_CASE} attempts; "
        f"{some_passed_count}/{len(results)} passed SOME attempts; "
        f"{failed_all_count}/{len(results)} failed all attempts."
    )


if __name__ == "__main__":
    main()
