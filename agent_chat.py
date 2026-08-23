import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document

from agent.tools import init_tools
from agent.graph import build_agent

def start_chat():
    repo_name = "sampleproject"
    persist_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), 'db', 'chroma_db', repo_name))

    if not os.path.exists(persist_directory):
        print(f"Error: No ChromaDB found at {persist_directory}. Run ingest.py first.")
        return

    print("Loading vector database and chunks...")
    embedding_model = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model,
        collection_metadata={"hnsw:space": "cosine"}
    )

    # Load all chunks into memory for BM25
    collection_data = vectorstore.get()
    all_chunks = []
    for doc_content, metadata in zip(collection_data['documents'], collection_data['metadatas']):
        all_chunks.append(Document(page_content=doc_content, metadata=metadata))

    print(f"Loaded {len(all_chunks)} chunks.")

    # Wire the tools with the loaded data
    init_tools(persist_directory, all_chunks)

    # Build the LangGraph ReAct agent
    agent = build_agent()

    chat_history = []

    print("\nAsk me questions about the codebase! Type 'quit' to exit.")

    while True:
        try:
            question = input("\nYour question: ").strip()
        except EOFError:
            break

        if not question:
            continue

        if question.lower() == 'quit':
            print("Goodbye!")
            break

        print(f"\n--- Running agent ---")

        # Build the messages list: history + current question
        messages = chat_history + [HumanMessage(content=question)]

        # Stream the agent so we can see each step (tool calls + final answer)
        for step in agent.stream({"messages": messages}, stream_mode="values"):
            last_msg = step["messages"][-1]
            msg_type = last_msg.__class__.__name__

            # ToolMessage = result of a tool call
            if msg_type == "ToolMessage":
                print(f"\n[Tool: {last_msg.name}]")
                # Show a preview of the tool output (first 300 chars)
                preview = last_msg.content[:300].replace('\n', ' ')
                if len(last_msg.content) > 300:
                    preview += "..."
                print(f"  Result preview: {preview}")

            # AIMessage with tool_calls = agent deciding to call a tool
            elif msg_type == "AIMessage" and hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
                for tc in last_msg.tool_calls:
                    print(f"\n[Agent calling tool: {tc['name']}]")
                    print(f"  Query: {tc['args'].get('query', '')}")

            # Final AIMessage with no tool calls = the answer
            elif msg_type == "AIMessage" and (not hasattr(last_msg, 'tool_calls') or not last_msg.tool_calls):
                if last_msg.content:
                    print(f"\n--- Answer ---\n{last_msg.content}")

        # Store the ORIGINAL question (not any rewritten version) in history
        final_messages = agent.get_state({"messages": messages}).values.get("messages", [])
        # Find the final AI answer content
        final_answer = ""
        for msg in reversed(final_messages):
            if msg.__class__.__name__ == "AIMessage" and msg.content and not (hasattr(msg, 'tool_calls') and msg.tool_calls):
                final_answer = msg.content
                break

        chat_history.append(HumanMessage(content=question))
        if final_answer:
            chat_history.append(AIMessage(content=final_answer))


if __name__ == "__main__":
    start_chat()
