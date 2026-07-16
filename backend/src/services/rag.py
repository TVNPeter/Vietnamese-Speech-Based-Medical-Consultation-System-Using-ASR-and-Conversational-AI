"""Hybrid RAG service: FAISS semantic search plus BM25 keyword search."""

import csv
import hashlib
import json
import logging
import pickle
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter
from llama_index.core.schema import Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from rank_bm25 import BM25Okapi

from src.config import settings
from src.schemas import RetrievedChunk

logger = logging.getLogger(__name__)

_MOJIBAKE_MARKERS = (
    "\u00c3",
    "\u00c4",
    "\u00c6",
    "\u00c1\u00bb",
    "\u00c2",
    "\u00e2\u0080",
)


def _mojibake_score(value: str) -> int:
    return sum(value.count(marker) for marker in _MOJIBAKE_MARKERS)


def _repair_mojibake(value: str) -> str:
    """Repair UTF-8 text that was accidentally decoded as Latin-1/Windows-1252."""
    if not _mojibake_score(value):
        return value

    byte_values = bytearray()
    for character in value:
        codepoint = ord(character)
        if codepoint <= 255:
            byte_values.append(codepoint)
            continue
        try:
            byte_values.extend(character.encode("cp1252"))
        except UnicodeEncodeError:
            return value

    try:
        repaired = bytes(byte_values).decode("utf-8")
    except UnicodeDecodeError:
        return value
    return repaired if _mojibake_score(repaired) < _mojibake_score(value) else value


def _as_text(value: Any) -> str:
    """Convert a dataset record into useful retrieval text, excluding IDs and URLs."""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts = [_as_text(item) for item in value]
        return "\n".join(part for part in parts if part)
    if not isinstance(value, dict):
        return ""

    content = _as_text(value.get("content"))
    if content:
        return content

    question = _as_text(value.get("question") or value.get("query"))
    answer = _as_text(value.get("answer") or value.get("response"))
    if question and answer:
        return f"Câu hỏi: {question}\n\nCâu trả lời: {answer}"

    prompt = _as_text(value.get("prompt") or value.get("input"))
    completion = _as_text(value.get("completion") or value.get("output"))
    if prompt and completion:
        return f"Câu hỏi: {prompt}\n\nCâu trả lời: {completion}"

    messages = value.get("messages")
    if isinstance(messages, list):
        dialogue = []
        for message in messages:
            if not isinstance(message, dict):
                continue
            text = _as_text(message.get("content"))
            if text:
                dialogue.append(text)
        if dialogue:
            return "\n\n".join(dialogue)

    text = _as_text(value.get("text") or value.get("instruction"))
    if text:
        return text

    allowed_fields = ("title", "description", "name", "category", "label")
    parts = [_as_text(value.get(field)) for field in allowed_fields]
    return "\n".join(part for part in parts if part)


class RAGService:
    """Manages local semantic and lexical indexes over the same document chunks."""

    def __init__(self) -> None:
        self._embedding_model: HuggingFaceEmbedding | None = None
        self._index: faiss.IndexFlatIP | None = None
        self._chunks: list[dict[str, str]] = []
        self._bm25: BM25Okapi | None = None
        self._chroma_client: Any | None = None
        self._chroma_collection: Any | None = None

    def _load_embedding_model(self) -> None:
        if self._embedding_model is not None:
            return

        device = settings.EMBEDDING_DEVICE
        if device == "auto":
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(
            "Loading embedding model %s on %s.",
            settings.EMBEDDING_MODEL_NAME,
            device,
        )
        self._embedding_model = HuggingFaceEmbedding(
            model_name=settings.EMBEDDING_MODEL_NAME,
            trust_remote_code=True,
            device=device,
            embed_batch_size=settings.EMBEDDING_BATCH_SIZE,
        )

    async def initialize(self) -> None:
        """Load the embedding model and existing indexes, or build them."""
        self._load_embedding_model()
        if self._load_prebuilt_chroma_index():
            return
        faiss_path = settings.resolve(settings.FAISS_INDEX_PATH)
        metadata_path = settings.resolve(settings.DOC_METADATA_PATH)
        if faiss_path.is_file() and metadata_path.is_file():
            self._index = faiss.read_index(str(faiss_path))
            self._chunks = json.loads(metadata_path.read_text(encoding="utf-8"))
            self._build_bm25_index()
            logger.info("Loaded %d chunks from disk.", len(self._chunks))
        else:
            await self._build_index()

    def _load_prebuilt_chroma_index(self) -> bool:
        """Load the packaged Chroma/HNSW and BM25 indexes without re-embedding docs."""
        if settings.RAG_BACKEND.lower() != "chroma":
            return False

        import chromadb

        chroma_dir = settings.resolve(settings.CHROMA_PERSIST_DIRECTORY)
        bm25_path = settings.resolve(settings.BM25_INDEX_PATH)
        if not chroma_dir.is_dir() or not bm25_path.is_file():
            raise FileNotFoundError(
                "Prebuilt Chroma/BM25 index is missing. "
                "Set RAG_BACKEND=faiss to create a new index from DOCUMENTS_DIR."
            )

        client = chromadb.PersistentClient(path=str(chroma_dir))
        collection = client.get_collection(settings.CHROMA_COLLECTION_NAME)
        with bm25_path.open("rb") as stream:
            bundle = pickle.load(stream)

        documents = bundle["documents"]
        document_ids = bundle["doc_ids"]
        metadatas = bundle["metadatas"]
        bm25_index = bundle["index"]
        if not (
            len(documents)
            == len(document_ids)
            == len(metadatas)
            == collection.count()
        ):
            raise ValueError("Prebuilt Chroma and BM25 indexes do not contain the same chunks.")

        self._chroma_client = client
        self._chroma_collection = collection
        self._index = None
        self._bm25 = bm25_index
        self._chunks = [
            {
                "id": str(document_id),
                "content": str(document),
                "source": str(metadata.get("source", "unknown")),
            }
            for document_id, document, metadata in zip(
                document_ids, documents, metadatas, strict=True
            )
        ]
        logger.info(
            "Loaded %d prebuilt Chroma chunks from %s.",
            len(self._chunks),
            chroma_dir,
        )
        return True

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Keep Vietnamese accented words and numbers for exact-term search."""
        return re.findall(r"[\w\u00c0-\u1ef9]+", text.lower(), flags=re.UNICODE)

    def _build_bm25_index(self) -> None:
        if not self._chunks:
            self._bm25 = None
            return
        self._bm25 = BM25Okapi(
            [self._tokenize(chunk["content"]) for chunk in self._chunks]
        )

    @staticmethod
    def _source_name(path: Path, docs_dir: Path) -> str:
        return path.relative_to(docs_dir).as_posix()

    def _iter_jsonl_records(
        self, path: Path, docs_dir: Path
    ) -> Iterator[tuple[str, str]]:
        source = self._source_name(path, docs_dir)
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("Skipping malformed JSONL record: %s:%d", path, line_number)
                    continue
                text = _repair_mojibake(_as_text(record))
                if text:
                    yield text, source

    def _iter_json_records(
        self, path: Path, docs_dir: Path
    ) -> Iterator[tuple[str, str]]:
        source = self._source_name(path, docs_dir)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.warning("Skipping malformed JSON file: %s", path)
            return
        records = payload if isinstance(payload, list) else [payload]
        for record in records:
            text = _repair_mojibake(_as_text(record))
            if text:
                yield text, source

    def _iter_csv_records(
        self, path: Path, docs_dir: Path
    ) -> Iterator[tuple[str, str]]:
        source = self._source_name(path, docs_dir)
        with path.open(encoding="utf-8", newline="") as stream:
            for record in csv.DictReader(stream):
                text = _repair_mojibake(_as_text(record))
                if text:
                    yield text, source

    def _iter_parquet_records(
        self, path: Path, docs_dir: Path
    ) -> Iterator[tuple[str, str]]:
        import pyarrow.parquet as pq

        source = self._source_name(path, docs_dir)
        parquet_file = pq.ParquetFile(path)
        for batch in parquet_file.iter_batches(batch_size=512):
            for record in batch.to_pylist():
                text = _repair_mojibake(_as_text(record))
                if text:
                    yield text, source

    def _iter_documents(self, docs_dir: Path) -> Iterator[tuple[str, str]]:
        for path in sorted(docs_dir.glob("**/*.md")):
            text = _repair_mojibake(path.read_text(encoding="utf-8"))
            if text:
                yield text, self._source_name(path, docs_dir)
        for path in sorted(docs_dir.glob("**/*.jsonl")):
            yield from self._iter_jsonl_records(path, docs_dir)
        for path in sorted(docs_dir.glob("**/*.json")):
            yield from self._iter_json_records(path, docs_dir)
        for path in sorted(docs_dir.glob("**/*.csv")):
            yield from self._iter_csv_records(path, docs_dir)
        for path in sorted(docs_dir.glob("**/*.parquet")):
            yield from self._iter_parquet_records(path, docs_dir)

    @staticmethod
    def _chunk_document(
        text: str,
        source: str,
        markdown_parser: MarkdownNodeParser,
        sentence_splitter: SentenceSplitter,
    ) -> Iterator[str]:
        document = Document(text=text, metadata={"source": source})
        markdown_nodes = markdown_parser.get_nodes_from_documents([document])
        nodes = sentence_splitter.get_nodes_from_documents(markdown_nodes)
        for node in nodes:
            content = node.get_content().strip()
            if content:
                yield content

    def _add_embedding_batch(self, contents: list[str], sources: list[str]) -> None:
        embeddings = self._embedding_model.get_text_embedding_batch(contents)  # type: ignore[union-attr]
        matrix = np.asarray(embeddings, dtype=np.float32)
        faiss.normalize_L2(matrix)
        if self._index is None:
            self._index = faiss.IndexFlatIP(matrix.shape[1])
        self._index.add(matrix)
        self._chunks.extend(
            {"content": content, "source": source}
            for content, source in zip(contents, sources, strict=True)
        )

    async def _build_index(self) -> None:
        """Index all supported text datasets under ``DOCUMENTS_DIR`` in bounded batches."""
        self._load_embedding_model()
        docs_dir = settings.resolve(settings.DOCUMENTS_DIR)
        if not docs_dir.is_dir():
            logger.warning("Documents directory does not exist: %s", docs_dir)
            self._index = faiss.IndexFlatIP(1)
            self._chunks = []
            self._bm25 = None
            return

        self._index = None
        self._chroma_collection = None
        self._chroma_client = None
        self._chunks = []
        self._bm25 = None
        markdown_parser = MarkdownNodeParser()
        sentence_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=64)
        seen_chunks: set[bytes] = set()
        pending_contents: list[str] = []
        pending_sources: list[str] = []
        source_files: set[str] = set()
        document_count = 0

        for text, source in self._iter_documents(docs_dir):
            document_count += 1
            source_files.add(source)
            for content in self._chunk_document(
                text, source, markdown_parser, sentence_splitter
            ):
                digest = hashlib.sha256(content.encode("utf-8")).digest()
                if digest in seen_chunks:
                    continue
                seen_chunks.add(digest)
                pending_contents.append(content)
                pending_sources.append(source)
                if len(pending_contents) >= settings.EMBEDDING_BATCH_SIZE:
                    self._add_embedding_batch(pending_contents, pending_sources)
                    pending_contents = []
                    pending_sources = []
            if document_count % 1000 == 0:
                logger.info(
                    "Prepared %d records and indexed %d unique chunks.",
                    document_count,
                    len(self._chunks),
                )

        if pending_contents:
            self._add_embedding_batch(pending_contents, pending_sources)
        if self._index is None:
            self._index = faiss.IndexFlatIP(1)

        self._build_bm25_index()
        self._save_index()
        logger.info(
            "Indexed %d unique chunks from %d records across %d source files.",
            len(self._chunks),
            document_count,
            len(source_files),
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
        """Refresh both indexes from all supported files in ``DOCUMENTS_DIR``."""
        self._load_embedding_model()
        if settings.RAG_BACKEND.lower() == "chroma":
            self._load_prebuilt_chroma_index()
            return len({chunk["source"] for chunk in self._chunks}), len(self._chunks)
        await self._build_index()
        return len({chunk["source"] for chunk in self._chunks}), len(self._chunks)

    def _query_prebuilt_chroma(self, text: str) -> list[RetrievedChunk]:
        """Fuse packaged Chroma semantic ranks and packaged BM25 lexical ranks."""
        query_embedding = self._embedding_model.get_query_embedding(text)  # type: ignore[union-attr]
        result = self._chroma_collection.query(
            query_embeddings=[query_embedding],
            n_results=min(settings.SEMANTIC_TOP_K, len(self._chunks)),
            include=["documents", "metadatas"],
        )
        fused: dict[str, dict[str, str | float]] = {}
        for rank, (document_id, document, metadata) in enumerate(
            zip(
                result["ids"][0],
                result["documents"][0],
                result["metadatas"][0],
                strict=True,
            ),
            start=1,
        ):
            fused[str(document_id)] = {
                "content": str(document),
                "source": str(metadata.get("source", "unknown")),
                "score": 1 / (settings.RRF_K + rank),
            }

        if self._bm25 is not None:
            lexical_scores = self._bm25.get_scores(self._tokenize(text))
            lexical_indices = np.argsort(lexical_scores)[::-1][: settings.BM25_TOP_K]
            for rank, index in enumerate(lexical_indices, start=1):
                if lexical_scores[index] <= 0:
                    continue
                chunk = self._chunks[int(index)]
                document_id = chunk["id"]
                if document_id not in fused:
                    fused[document_id] = {
                        "content": chunk["content"],
                        "source": chunk["source"],
                        "score": 0.0,
                    }
                fused[document_id]["score"] = float(fused[document_id]["score"]) + 1 / (
                    settings.RRF_K + rank
                )

        ranked = sorted(
            fused.values(), key=lambda item: float(item["score"]), reverse=True
        )[: settings.RAG_TOP_K]
        return [
            RetrievedChunk(
                content=str(item["content"]),
                source=str(item["source"]),
                score=float(item["score"]),
            )
            for item in ranked
        ]

    def query(self, text: str) -> list[RetrievedChunk]:
        """Fuse vector and BM25 ranks with reciprocal-rank fusion (RRF)."""
        if self._chroma_collection is not None:
            return self._query_prebuilt_chroma(text)
        if self._index is None or self._index.ntotal == 0:
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
        if self._chroma_collection is not None:
            return len(self._chunks)
        return self._index.ntotal if self._index is not None else 0
