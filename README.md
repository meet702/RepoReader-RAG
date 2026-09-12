# 🧠 Codebase RAG Assistant

> An intelligent, Retrieval-Augmented Generation (RAG) platform to interactively chat with and understand your source code, powered by advanced LangGraph agents and ChromaDB vector search.

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white) ![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white) ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white) ![ChromaDB](https://img.shields.io/badge/ChromaDB-FF6F00?style=for-the-badge) ![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge) ![LangGraph](https://img.shields.io/badge/LangGraph-1C3C3C?style=for-the-badge) ![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white) ![License](https://img.shields.io/badge/License-MIT-lightgrey?style=for-the-badge)

---

## 📖 Overview

The **Codebase RAG Assistant** is a production-ready application that allows developers to seamlessly ingest entire source code repositories and query them using natural language. The system leverages state-of-the-art vector embeddings with **ChromaDB** for accurate retrieval and an advanced **LangGraph ReAct agent** capable of multi-step reasoning, routing queries, and retrieving critical context.

Whether you're onboarding onto a new project or debugging complex architecture, this tool brings the power of LLMs directly to your codebase.

## ✨ Key Features

- 📥 **Code Ingestion** — Parses, chunks, and embeds source code files efficiently.
- 🔍 **Vector & Sparse Search** — Hybrid retrieval utilizing ChromaDB for semantic search and BM25 for precise keyword matching.
- 🤖 **Agentic Chat** — LangGraph-powered ReAct agent that intelligently routes queries and retrieves exact context from the codebase.
- 🔌 **API Server** — A robust FastAPI backend exposing RAG capabilities programmatically.
- 🖥️ **Interactive User Interface** — A sleek Streamlit web app for intuitive interaction and visualizations.
- 🐳 **Docker Support** — Turnkey Docker Compose setup to spin up API and UI services in seconds.

## 📁 Project Structure

- `agent/`: LangGraph agent definitions and tools.
- `api/`: FastAPI server implementation.
- `code_graph/`: Utilities for building and querying the code graph.
- `db/`: Directory where the ChromaDB vector database and code graphs are persisted.
- `Ingestion/`: Scripts and pipelines for loading and chunking documents.
- `llm/`: Configurations and providers for language models and embeddings.
- `retrieval/`: Pipeline for querying and retrieving relevant chunks from the database.
- `ui/`: Streamlit web interface.
- `agent_chat.py`: CLI script for agent-based interaction.
- `chat.py`: Basic CLI script for chatting with the vector database.
- `ingestion_pipeline.py` & `retrieval_pipeline.py`: Scripts to test ingestion and retrieval independently.

## 🚀 Getting Started

### 🛠️ Prerequisites

- Python 3.8+
- Docker & Docker Compose (optional, for containerized deployment)

### ⚙️ Setup

1. **Clone the repository:**
   ```bash
   git clone <repository_url>
   cd <repository_name>
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Variables:**
   Create a `.env` file in the root directory (you can use `.env.production` as a reference) and add the necessary API keys for your LLM and embedding providers.

### 🎮 Usage

#### Ingestion
Before querying, you need to ingest a codebase to build the vector database.
Run the ingestion pipeline (ensure you configure it to point to your target repository):
```bash
python ingestion_pipeline.py
```

#### CLI Chat
You can chat with the ingested codebase directly from your terminal:
- **Basic RAG Chat:**
  ```bash
  python chat.py
  ```
- **Agentic Chat (with LangGraph tools):**
  ```bash
  python agent_chat.py
  ```

#### Docker Deployment
To run the API and UI using Docker Compose:
```bash
docker-compose up --build
```
This will start:
- **API:** http://localhost:8000
- **UI (Streamlit):** http://localhost:8501

## 🛠️ Technologies Used

- [LangChain](https://github.com/langchain-ai/langchain) & [LangGraph](https://langchain-ai.github.io/langgraph/)
- [ChromaDB](https://www.trychroma.com/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Streamlit](https://streamlit.io/)
- [Sentence Transformers](https://www.sbert.net/) & BM25
- Docker

## 🛡️ License

[MIT License](LICENSE)
