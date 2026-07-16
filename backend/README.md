# Backend setup

The RAG service indexes Markdown in `backend/documents/` twice: FAISS handles
semantic matches and BM25 preserves exact terms such as medicine names. Their
rankings are combined with reciprocal-rank fusion.

## Medical ASR pipeline

`POST /api/asr` accepts the browser's `audio/webm` recording and runs:

1. Wav2Vec2 CTC (`backend/models/asr/onnx`)
2. Beam search with `text/drugs.txt` hotword boosting and
   `models/kenlm/kenlm_vi_medical_4gram.bin`
3. ViT5 medical rewrite (`backend/models/vit5/onnx`)

`text/corpus.txt` is the corpus used to build the supplied KenLM binary; it is
kept for traceability and is not loaded on each request. The Python `kenlm`
binding enables the language-model stage. On Windows it is a native build: use
the locally built wheel installed in `backend/.venv`, then set
`ASR_REQUIRE_KENLM=true` to make the service fail fast instead of falling back
to CTC plus drug hotwords. The base `uv sync` remains usable without it.

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
TTS_OUTPUT_FILENAME=vieneu-tts-l-cystine-utf8.wav
```

VieNeu-TTS v3 Turbo loads on the first `POST /api/tts` request and downloads
its own cached weights. Do not commit model weights into this repository.

`POST /api/tts` preserves UTF-8 Vietnamese request text and uses
`vieneu-tts-l-cystine-utf8.wav` as the default output filename.
