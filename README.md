# Codebase RAG Assistant

This project is a Retrieval-Augmented Generation (RAG) system specifically designed for chatting with and understanding codebases. It allows you to ingest source code repositories, store their embeddings in a ChromaDB vector database, and interact with the codebase using an advanced AI agent powered by LangChain and LangGraph.

## Features

- **Code Ingestion:** Parses and chunks source code files to create embeddings.
- **Vector Search:** Utilizes ChromaDB for efficient similarity search and BM25 for sparse/keyword search.
- **Agentic Chat:** A LangGraph ReAct agent that can route queries, retrieve context, and provide detailed answers about the codebase.
- **API Server:** A FastAPI application for exposing the RAG capabilities programmatically.
- **User Interface:** A Streamlit-based web interface for easy interaction.
- **Docker Support:** Ready-to-use Docker Compose configuration to spin up the API and UI services seamlessly.

## Project Structure

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

## Getting Started

### Prerequisites

- Python 3.8+
- Docker & Docker Compose (optional, for containerized deployment)

### Setup

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

### Usage

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

## Technologies Used

- [LangChain](https://github.com/langchain-ai/langchain) & [LangGraph](https://langchain-ai.github.io/langgraph/)
- [ChromaDB](https://www.trychroma.com/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Streamlit](https://streamlit.io/)
- [Sentence Transformers](https://www.sbert.net/) & BM25
- Docker
