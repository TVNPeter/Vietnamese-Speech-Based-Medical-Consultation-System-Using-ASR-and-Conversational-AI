# Backend

Hướng dẫn cài đặt, bàn giao assets và chạy toàn bộ hệ thống nằm ở
[README tại thư mục gốc](../README.md). Tài liệu này chỉ tóm tắt các chi tiết
backend cần lưu ý.

## Thành phần runtime

- **LLM:** `llama-server` chạy `models/qwen3-4b-thinking.gguf` và được backend
  tự khởi động khi app khởi động.
- **RAG:** Chroma prebuilt ở `models/chromadb` kết hợp BM25 prebuilt ở
  `models/bm25_index.pkl`; ranking dùng reciprocal-rank fusion (RRF).
- **ASR:** custom Wav2Vec2 checkpoint ở `models/asr/best_model_hf` → KenLM +
  hotword `text/drugs.txt` → ViT5 ở
  `models/kenlm+vit5/vit5_medical_rewrite_stage2_final`.
- **TTS:** VieNeu-TTS v3 Turbo; endpoint giữ UTF-8 và trả tên file mặc định
  `vieneu-tts-l-cystine-utf8.wav`.

## Chạy backend

```powershell
cd backend
Copy-Item .env.example .env
uv sync
.\.venv\Scripts\Activate.ps1
python -m uvicorn src.app:app --host 127.0.0.1 --port 8000
```

Truy cập Swagger tại <http://localhost:8000/docs>.

## KenLM

Để bắt buộc pipeline ASR sử dụng KenLM, cài Python wheel `kenlm` đã được build
cho Windows/Python 3.12 và đặt `ASR_REQUIRE_KENLM=true` trong `.env`. Nếu chưa có
wheel, để giá trị `false`; ASR vẫn chạy CTC + hotword nhưng chất lượng thấp hơn.

## Reindex RAG

Sau khi thay đổi corpus/index, gọi:

```powershell
Invoke-RestMethod -Method Post http://127.0.0.1:8000/api/rag/reindex
```

Reindex có thể tốn thời gian và yêu cầu đủ tài nguyên GPU/CPU. Để dùng index đã
đóng gói, giữ nguyên `models/chromadb` và `models/bm25_index.pkl`.
