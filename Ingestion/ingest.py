from github_loader import GitHubLoader
from file_filter import is_supported_file
from ast_chunker import ASTChunker

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma


def ingest_repository(repo_url):

    loader = GitHubLoader(repo_url)

    repo_path = loader.clone_repository()

    try:

        # -------------------------------------------------
        # 1. Get files
        # -------------------------------------------------

        files = loader.get_repository_files(
            repo_path
        )

        print(
            f"\nTotal files found: {len(files)}"
        )

        # -------------------------------------------------
        # 2. Filter files
        # -------------------------------------------------

        supported_files = [
            file
            for file in files
            if is_supported_file(file)
        ]

        print(
            f"Supported files: {len(supported_files)}"
        )

        # -------------------------------------------------
        # 3. AST chunking
        # -------------------------------------------------

        chunker = ASTChunker()

        all_chunks = []

        for file_path in supported_files:

            print(
                f"Processing: {file_path}"
            )

            chunks = chunker.chunk_file(
                file_path,
                repo_path
            )

            all_chunks.extend(chunks)

        print(
            f"\nTotal chunks created: {len(all_chunks)}"
        )

        # -------------------------------------------------
        # 4. Inspect chunks
        # -------------------------------------------------

        for i, chunk in enumerate(
            all_chunks[:10]
        ):

            print("\n" + "=" * 70)

            print(
                f"Chunk {i + 1}"
            )

            print(
                f"Metadata: {chunk.metadata}"
            )

            print(
                chunk.page_content[:500]
            )

        # -------------------------------------------------
        # 5. Embeddings
        # -------------------------------------------------

        embedding_model = OllamaEmbeddings(
            model="nomic-embed-text"
        )

        # -------------------------------------------------
        # 6. ChromaDB
        # -------------------------------------------------

        vectorstore = Chroma.from_documents(
            documents=all_chunks,
            embedding=embedding_model,
            persist_directory="db/chroma_db",
            collection_metadata={
                "hnsw:space": "cosine"
            }
        )

        print(
            "\nSuccessfully stored chunks in ChromaDB."
        )

        return vectorstore

    finally:

        loader.cleanup(repo_path)


if __name__ == "__main__":

    repository_url = input(
        "Enter GitHub repository URL: "
    )

    ingest_repository(
        repository_url
    )