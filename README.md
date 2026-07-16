# Trợ lý Y tế AI tiếng Việt

Hệ thống tư vấn y tế chạy cục bộ, gồm chat RAG, nhận dạng giọng nói tiếng Việt
(ASR) và đọc câu trả lời (VieNeu-TTS). Backend tự khởi động `llama-server`, nên
người dùng chỉ cần chạy backend và giao diện web.

> **Lưu ý y tế:** Đây là công cụ tham khảo, không thay thế bác sĩ, chẩn đoán hoặc
> điều trị. Không dùng trong tình huống cấp cứu.

## Tính năng

- Chat tiếng Việt với Qwen3 chạy local qua `llama-server`.
- Hybrid RAG: Chroma semantic search + BM25 + RRF; ưu tiên nguồn y tế tiếng Việt
  (Vinmec, MEDLATEC, Bộ Y tế, Bệnh viện 108...) qua Tavily khi được cấu hình.
- ASR pipeline: Wav2Vec2 custom → KenLM + hotword `text/drugs.txt` → ViT5 rewrite.
- VieNeu-TTS v3 Turbo, giữ nguyên UTF-8 và trả file
  `vieneu-tts-l-cystine-utf8.wav`.
- React/Vite chat UI có microphone, TTS, nguồn tham khảo và lịch sử chat cục bộ.

## Yêu cầu máy

Đã kiểm thử trên Windows 10/11. Cần:

- Python 3.12 và [uv](https://docs.astral.sh/uv/).
- Node.js LTS (npm đi kèm).
- 12 GB RAM trở lên; khuyến nghị 24 GB RAM.
- GPU có Vulkan là tùy chọn nhưng khuyến nghị cho Qwen. Nếu không có GPU, cần
  dùng bản `llama-server` CPU tương ứng và giảm tốc độ kỳ vọng.
- Khoảng 8–12 GB trống sau khi giải nén assets và cài dependencies.

## Assets bắt buộc

Khi bàn giao, phải giải nén/copy assets **giữ nguyên cấu trúc thư mục** sau:

```text
project-root/
├── backend/
│   ├── llama-server/llama-server.exe
│   ├── templates/qwen3_no_think.jinja
│   └── .env                 # tạo từ .env.example, không lấy từ máy dev
├── models/
│   ├── qwen3-4b-thinking.gguf
│   ├── chromadb/
│   ├── bm25_index.pkl
│   ├── asr/best_model_hf/
│   └── kenlm+vit5/
│       ├── kenlm_vi_medical_4gram.bin
│       └── vit5_medical_rewrite_stage2_final/
└── text/
    ├── drugs.txt
    └── corpus.txt            # chỉ dùng để truy vết/huấn luyện KenLM
```

`models/chromadb` và `models/bm25_index.pkl` là index RAG đã dựng sẵn; thiếu một
trong hai thì RAG prebuilt không khởi động. `models.zip`, `data.zip` và file text
nén không được Git theo dõi, vì vậy cần giao cùng source code hoặc giải nén sẵn
trước khi bàn giao.

## Chạy lần đầu trên Windows

Mở PowerShell tại thư mục gốc của project.

### 1. Tạo cấu hình cục bộ

```powershell
Copy-Item .\backend\.env.example .\backend\.env
```

Mở `backend/.env` và chỉ sửa các mục cần thiết:

```env
# Optional: để trống nếu chỉ dùng RAG local.
TAVILY_API_KEY=

# Đặt true nếu muốn Tavily ưu tiên trang y tế tiếng Việt.
RAG_PREFER_VIETNAMESE_WEB=true
```

Các đường dẫn model trong `.env.example` đều là đường dẫn tương đối, nên không
cần thay thành `C:\Users\...` khi giao cho máy khác.

### 2. Cài backend

```powershell
cd .\backend
uv sync
```

Để bật đầy đủ KenLM trên Windows, người bàn giao cần cung cấp wheel `kenlm` đã
được build/kiểm thử cho Python 3.12, rồi cài vào môi trường:

```powershell
uv pip install ..\wheels\kenlm-<version>-cp312-win_amd64.whl
```

Nếu chưa có wheel này, đặt `ASR_REQUIRE_KENLM=false` trong `.env`. ASR vẫn chạy
CTC + hotword, nhưng chất lượng nhận diện sẽ thấp hơn.

### 3. Chạy backend

```powershell
cd .\backend
.\.venv\Scripts\Activate.ps1
python -m uvicorn src.app:app --host 127.0.0.1 --port 8000
```

Lần khởi động đầu có thể mất một lúc vì hệ thống nạp embedding, index RAG và Qwen.
Kiểm tra backend đã sẵn sàng:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/stats
```

Không cần chạy `llama-server.exe` thủ công: backend sẽ tự chạy nó từ
`backend/llama-server/`.

### 4. Chạy frontend

Mở **một PowerShell khác**:

```powershell
cd <duong-dan-den-project>\frontend
npm install
npm run dev
```

Mở địa chỉ Vite in ra trên terminal, thường là <http://localhost:5173>.

## Kiểm tra sau bàn giao

1. Mở giao diện, gửi: `Tác dụng phụ của warfarin là gì?`.
2. Xác nhận có phản hồi và mục **Nguồn tham khảo**.
3. Bấm **Nghe trả lời** để kiểm tra TTS; file phải trả về tên
   `vieneu-tts-l-cystine-utf8.wav`.
4. Dùng endpoint `/api/asr` hoặc microphone để kiểm tra ASR. Cần phát âm rõ tên
   thuốc; checkpoint hiện tại vẫn có thể nhầm các tên thuốc gần âm.

## API chính

| Endpoint | Chức năng |
| --- | --- |
| `POST /api/chat/stream` | Chat RAG, trả Server-Sent Events (SSE). |
| `POST /api/asr` | Upload `wav`/`webm` để nhận transcript. |
| `POST /api/tts` | Chuyển text tiếng Việt thành WAV. |
| `POST /api/rag/reindex` | Dựng lại index sau khi thay đổi corpus. |
| `GET /api/stats` | Kiểm tra index và backend đã sẵn sàng. |

Swagger: <http://localhost:8000/docs>

## GPU và chế độ CPU

- `LLAMA_GPU_LAYERS=99`: offload Qwen sang GPU qua `llama-server` Vulkan.
- `EMBEDDING_DEVICE=auto`: tự dùng CUDA khi PyTorch nhận GPU, nếu không sẽ dùng CPU.
- Với Tesla P40 đã kiểm thử: để `ASR_WAV2VEC2_USE_GPU=false`, còn ViT5 rewrite
  có thể dùng GPU qua `ASR_USE_GPU=true`.
- Nếu máy người mua không có Vulkan/GPU phù hợp, thay `backend/llama-server` bằng
  bản CPU tương thích rồi đặt `LLAMA_GPU_LAYERS=0`.

## Xử lý lỗi thường gặp

### Port 8000 đã được sử dụng

```powershell
Get-NetTCPConnection -LocalPort 8000 -State Listen |
  Select-Object -ExpandProperty OwningProcess
Stop-Process -Id <PID> -Force
```

Sau đó chạy lại backend.

### Backend báo thiếu model hoặc index

Kiểm tra lại cấu trúc trong phần **Assets bắt buộc**. Đặc biệt cần có
`models/qwen3-4b-thinking.gguf`, `models/chromadb` và `models/bm25_index.pkl`.

### Không có âm thanh TTS

Kiểm tra endpoint `POST /api/tts`. VieNeu có thể tải cache/weights ở lần gọi đầu;
muốn bàn giao hoàn toàn offline cần pre-warm TTS và đóng gói cache theo giấy phép
của VieNeu/Hugging Face.

## Checklist trước khi bán/bàn giao

- [ ] Chạy smoke test backend + frontend trên một Windows user/profile sạch.
- [ ] Đóng gói source cùng `models/`, `text/`, `backend/llama-server/` và wheel
  KenLM nếu yêu cầu `ASR_REQUIRE_KENLM=true`.
- [ ] Chỉ giao `backend/.env.example`; không giao `.env` dev, HF token hay Tavily key.
- [ ] Ghi rõ cấu hình tối thiểu, loại GPU/driver đã kiểm thử và cách chạy CPU fallback.
- [ ] Cung cấp phiên bản source/commit, checksum cho file nén và kênh hỗ trợ.
- [ ] Kiểm tra quyền phân phối thương mại của từng model, dataset, TTS weight và
  binary trước khi bán. `LICENSE` (MIT) chỉ áp dụng cho phần source code của repo,
  không tự động cấp quyền bán lại assets bên thứ ba.

## Phát triển

```powershell
cd .\frontend
npm run lint
npm run build

cd ..\backend
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## License

Source code dùng [MIT License](LICENSE). Model, dataset, binary và weight đi kèm
phải được kiểm tra license riêng trước khi tái phân phối hoặc sử dụng thương mại.
