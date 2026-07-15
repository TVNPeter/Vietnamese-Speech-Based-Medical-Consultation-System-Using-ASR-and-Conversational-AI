"""Hybrid RAG service: FAISS semantic search plus BM25 keyword search."""

import json
import logging
import re

import faiss
import numpy as np
from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter
from llama_index.core.schema import Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from rank_bm25 import BM25Okapi

from src.config import settings
from src.schemas import RetrievedChunk

logger = logging.getLogger(__name__)


class RAGService:
    """Manages local semantic and lexical indexes over the same document chunks."""

    def __init__(self) -> None:
        self._embedding_model: HuggingFaceEmbedding | None = None
        self._index: faiss.IndexFlatIP | None = None
        self._chunks: list[dict[str, str]] = []
        self._bm25: BM25Okapi | None = None

    async def initialize(self) -> None:
        """Load the embedding model and existing indexes, or build them."""
        logger.info("Loading embedding model: %s", settings.EMBEDDING_MODEL_NAME)
        self._embedding_model = HuggingFaceEmbedding(
            model_name=settings.EMBEDDING_MODEL_NAME,
            trust_remote_code=True,
        )
        faiss_path = settings.resolve(settings.FAISS_INDEX_PATH)
        metadata_path = settings.resolve(settings.DOC_METADATA_PATH)
        if faiss_path.is_file() and metadata_path.is_file():
            self._index = faiss.read_index(str(faiss_path))
            self._chunks = json.loads(metadata_path.read_text(encoding="utf-8"))
            self._build_bm25_index()
            logger.info("Loaded %d chunks from disk.", len(self._chunks))
        else:
            await self._build_index()

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Keep Vietnamese accented words and numbers for exact-term search."""
        return re.findall(r"[\wÀ-ỹ]+", text.lower(), flags=re.UNICODE)

    def _build_bm25_index(self) -> None:
        if not self._chunks:
            self._bm25 = None
            return
        self._bm25 = BM25Okapi(
            [self._tokenize(chunk["content"]) for chunk in self._chunks]
        )

    async def _build_index(self) -> None:
        """Read Markdown files, chunk once, then build both retrieval indexes."""
        docs_dir = settings.resolve(settings.DOCUMENTS_DIR)
        md_files = sorted(docs_dir.glob("**/*.md"))
        if not md_files:
            logger.warning("No Markdown files found in %s; index is empty.", docs_dir)
            self._index = faiss.IndexFlatIP(1)
            self._chunks = []
            self._bm25 = None
            return

        documents = [
            Document(
                text=md_file.read_text(encoding="utf-8"),
                metadata={"source": md_file.name},
            )
            for md_file in md_files
        ]
        nodes = MarkdownNodeParser().get_nodes_from_documents(documents)
        final_nodes = SentenceSplitter(
            chunk_size=512, chunk_overlap=64
        ).get_nodes_from_documents(
            nodes  # type: ignore[arg-type]
        )
        texts = [node.get_content() for node in final_nodes]
        embeddings = self._embedding_model.get_text_embedding_batch(texts)  # type: ignore[union-attr]
        embedding_matrix = np.array(embeddings, dtype=np.float32)
        faiss.normalize_L2(embedding_matrix)
        self._index = faiss.IndexFlatIP(embedding_matrix.shape[1])
        self._index.add(embedding_matrix)
        self._chunks = [
            {
                "content": node.get_content(),
                "source": node.metadata.get("source", "unknown"),
            }
            for node in final_nodes
        ]
        self._build_bm25_index()
        self._save_index()
        logger.info(
            "Indexed %d chunks from %d documents.", len(self._chunks), len(documents)
        )

    def _save_index(self) -> None:
        faiss_path = settings.resolve(settings.FAISS_INDEX_PATH)
        metadata_path = settings.resolve(settings.DOC_METADATA_PATH)
        faiss_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(faiss_path))  # type: ignore[arg-type]
        metadata_path.write_text(
            json.dumps(self._chunks, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    async def rebuild(self) -> tuple[int, int]:
        """Refresh both indexes after documents in backend/documents change."""
        await self._build_index()
        return len({chunk["source"] for chunk in self._chunks}), len(self._chunks)

    def query(self, text: str) -> list[RetrievedChunk]:
        """Fuse vector and BM25 ranks with reciprocal-rank fusion (RRF)."""
        if not self._index or self._index.ntotal == 0:
            return []
        query_embedding = self._embedding_model.get_query_embedding(text)  # type: ignore[union-attr]
        query_vector = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(query_vector)
        _, semantic_indices = self._index.search(
            query_vector, min(settings.SEMANTIC_TOP_K, self._index.ntotal)
        )
        fused_scores: dict[int, float] = {}
        for rank, index in enumerate(semantic_indices[0], start=1):
            if index >= 0:
                fused_scores[int(index)] = 1 / (settings.RRF_K + rank)

        if self._bm25 is not None:
            lexical_scores = self._bm25.get_scores(self._tokenize(text))
            lexical_indices = np.argsort(lexical_scores)[::-1][: settings.BM25_TOP_K]
            for rank, index in enumerate(lexical_indices, start=1):
                if lexical_scores[index] > 0:
                    fused_scores[int(index)] = fused_scores.get(int(index), 0.0) + 1 / (
                        settings.RRF_K + rank
                    )

        ranked_indices = sorted(fused_scores, key=fused_scores.get, reverse=True)[
            : settings.RAG_TOP_K
        ]
        return [
            RetrievedChunk(
                content=self._chunks[index]["content"],
                source=self._chunks[index]["source"],
                score=fused_scores[index],
            )
            for index in ranked_indices
        ]

    @property
    def document_count(self) -> int:
        """Return the number of indexed chunks."""
        return self._index.ntotal if self._index else 0
