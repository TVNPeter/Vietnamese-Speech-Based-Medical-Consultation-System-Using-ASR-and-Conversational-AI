"""RAG service: document indexing and retrieval with FAISS."""

import json
import logging

import faiss
import numpy as np
from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter
from llama_index.core.schema import Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from src.config import settings
from src.schemas import RetrievedChunk

logger = logging.getLogger(__name__)


class RAGService:
    """Manages FAISS index and document retrieval."""

    def __init__(self) -> None:
        self._embedding_model: HuggingFaceEmbedding | None = None
        self._index: faiss.IndexFlatIP | None = None
        self._chunks: list[dict[str, str]] = []

    async def initialize(self) -> None:
        """Load embedding model and FAISS index (or build from documents)."""
        logger.info("Loading embedding model: %s", settings.EMBEDDING_MODEL_NAME)
        self._embedding_model = HuggingFaceEmbedding(
            model_name=settings.EMBEDDING_MODEL_NAME,
            trust_remote_code=True,
        )

        faiss_path = settings.resolve(settings.FAISS_INDEX_PATH)
        metadata_path = settings.resolve(settings.DOC_METADATA_PATH)

        if faiss_path.is_file() and metadata_path.is_file():
            logger.info("Loading existing FAISS index from %s", faiss_path)
            self._index = faiss.read_index(str(faiss_path))
            self._chunks = json.loads(metadata_path.read_text(encoding="utf-8"))
            logger.info("Loaded %d chunks from index.", len(self._chunks))
        else:
            await self._build_index()

    async def _build_index(self) -> None:
        """Read all .md documents, chunk, embed, and build FAISS index."""
        docs_dir = settings.resolve(settings.DOCUMENTS_DIR)
        md_files = sorted(docs_dir.glob("**/*.md"))

        if not md_files:
            logger.warning("No .md files found in %s — index will be empty.", docs_dir)
            self._index = faiss.IndexFlatIP(1)
            self._chunks = []
            return

        logger.info("Building index from %d .md files...", len(md_files))

        documents: list[Document] = []
        for md_file in md_files:
            text = md_file.read_text(encoding="utf-8")
            doc = Document(
                text=text,
                metadata={"source": md_file.name},
            )
            documents.append(doc)

        # Parse by markdown headers, then split large sections
        md_parser = MarkdownNodeParser()
        nodes = md_parser.get_nodes_from_documents(documents)

        splitter = SentenceSplitter(chunk_size=512, chunk_overlap=64)
        final_nodes = splitter.get_nodes_from_documents(
            nodes,  # type: ignore[arg-type]
        )

        logger.info(
            "Created %d chunks from %d documents.",
            len(final_nodes),
            len(documents),
        )

        # Embed all chunks
        texts = [node.get_content() for node in final_nodes]
        embeddings = self._embedding_model.get_text_embedding_batch(  # type: ignore[union-attr]
            texts,
        )
        embedding_matrix = np.array(embeddings, dtype=np.float32)

        # Normalize for cosine similarity via inner product
        faiss.normalize_L2(embedding_matrix)

        # Build FAISS index
        dim = embedding_matrix.shape[1]
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(embedding_matrix)

        # Store chunk metadata
        self._chunks = [
            {
                "content": node.get_content(),
                "source": node.metadata.get("source", "unknown"),
            }
            for node in final_nodes
        ]

        self._save_index()

    def _save_index(self) -> None:
        """Persist FAISS index and chunk metadata to disk."""
        faiss_path = settings.resolve(settings.FAISS_INDEX_PATH)
        metadata_path = settings.resolve(settings.DOC_METADATA_PATH)

        faiss_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(
            self._index,  # type: ignore[arg-type]
            str(faiss_path),
        )
        metadata_path.write_text(
            json.dumps(self._chunks, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info(
            "Saved FAISS index (%d vectors) to %s",
            self._index.ntotal,  # type: ignore[union-attr]
            faiss_path,
        )

    def query(self, text: str, top_k: int = 5) -> list[RetrievedChunk]:
        """Retrieve the top-k most relevant chunks for a query."""
        if not self._index or self._index.ntotal == 0:
            return []

        query_embedding = self._embedding_model.get_query_embedding(  # type: ignore[union-attr]
            text,
        )
        query_vector = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(query_vector)

        scores, indices = self._index.search(
            query_vector, min(top_k, self._index.ntotal)
        )

        results: list[RetrievedChunk] = []
        for score, idx in zip(scores[0], indices[0], strict=False):
            if idx < 0:
                continue
            chunk = self._chunks[idx]
            results.append(
                RetrievedChunk(
                    content=chunk["content"],
                    source=chunk["source"],
                    score=float(score),
                )
            )
        return results

    @property
    def document_count(self) -> int:
        """Return number of indexed chunks."""
        return self._index.ntotal if self._index else 0
