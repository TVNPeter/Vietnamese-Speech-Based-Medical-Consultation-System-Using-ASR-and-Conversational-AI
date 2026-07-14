# Vietnamese Speech-Based Medical Consultation System 🏥

A modern, full-stack AI medical consultation system designed to provide accurate medical information in Vietnamese. The system integrates advanced Retrieval-Augmented Generation (RAG), Large Language Models (LLMs), Automatic Speech Recognition (ASR), and Text-to-Speech (TTS) capabilities.

## ✨ Features

- **Conversational AI**: Uses an LLM (`qwen3-4b-thinking.gguf` via `llama-server`) to provide coherent and helpful medical advice.
- **RAG Pipeline**: Retrieves relevant medical context from Markdown documents using `llama-index`, HuggingFace embeddings (`Dqdung205/medical_vietnamese_embedding`), and FAISS for efficient semantic search.
- **Voice Interaction**:
  - **ASR**: Click the microphone to speak your symptoms directly into the chat.
  - **TTS**: Click the speaker icon to hear the AI's response spoken aloud.
- **Modern UI**: A responsive, glassmorphism dark-theme interface built with React.

## 🛠️ Tech Stack

### Backend (`/backend`)
- **Python 3.12**
- **FastAPI**: Async API server with server-sent events (SSE) for streaming responses.
- **Llama-Index & FAISS**: For document chunking and vector storage.
- **llama-server**: Self-hosted LLM execution using Vulkan acceleration.
- **uv**: Fast Python package installer and resolver.

### Frontend (`/frontend`)
- **React 19 & Vite**
- **Bun**: Ultra-fast JavaScript runtime and package manager.
- **React Markdown**: For rendering rich text and tables in bot responses.
- **Lucide React**: For beautiful iconography.

## 🚀 Getting Started

### Prerequisites
- [uv](https://github.com/astral-sh/uv) (for backend)
- [bun](https://bun.sh/) (for frontend)
- Python 3.12+

### 1. Model Setup
Download and place the required models:
- **LLM**: Place `qwen3-4b-thinking.gguf` in `backend/models/`.
- **ASR**: Place your ASR model files in `backend/models/asr/`.
- **TTS**: Place your TTS model files in `backend/models/tts/`.
- **Documents**: Place your `.md` medical knowledge files in `backend/documents/`. The FAISS index will be built automatically on startup.

### 2. Backend Setup
```bash
cd backend
# Create and activate virtual environment, install dependencies
uv sync
# Or manually:
# uv venv --python 3.12
# uv pip install -e .

# Run the backend server
uv run main.py
```
*The backend will run at `http://localhost:8000`.*

### 3. Frontend Setup
```bash
cd frontend
# Install dependencies
bun install

# Start the development server
bun run dev
```
*The frontend will be available at `http://localhost:5173`.*

## ⚠️ Disclaimer
This system is an AI assistant intended for informational purposes only. It is **not** a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.
