# Backend setup

The RAG service indexes Markdown in `backend/documents/` twice: FAISS handles
semantic matches and BM25 preserves exact terms such as medicine names. Their
rankings are combined with reciprocal-rank fusion.

After adding or changing documents, rebuild both local indexes:

```bash
curl -X POST http://localhost:8000/api/rag/reindex
```

Optional `.env` settings:

```env
SEMANTIC_TOP_K=5
BM25_TOP_K=5
RAG_TOP_K=5
TTS_ENABLED=true
TTS_VOICE=Trúc Ly
```

VieNeu-TTS v3 Turbo loads on the first `POST /api/tts` request and downloads
its own cached weights. Do not commit model weights into this repository.
