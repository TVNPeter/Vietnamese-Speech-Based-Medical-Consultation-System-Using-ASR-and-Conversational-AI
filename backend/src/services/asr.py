"""Vietnamese medical ASR: Wav2Vec2 CTC, KenLM/hotwords, then ViT5 rewrite."""

import asyncio
import io
import json
import logging
import os
import threading
from pathlib import Path

from src.config import settings

logger = logging.getLogger(__name__)


class ASRService:
    """Lazy, process-local implementation of the medical speech pipeline."""

    def __init__(self) -> None:
        self._model_path = settings.resolve(settings.ASR_MODEL_PATH)
        self._kenlm_path = settings.resolve(settings.ASR_KENLM_PATH)
        self._vit5_path = settings.resolve(settings.ASR_VIT5_MODEL_PATH)
        self._hotwords_path = settings.resolve(settings.ASR_HOTWORDS_PATH)
        self._lock = threading.Lock()
        self._asr_session: object | None = None
        self._vit5_encoder: object | None = None
        self._vit5_decoder: object | None = None
        self._ctc_decoder: object | None = None
        self._tokenizer: object | None = None
        self._hotwords: list[str] = []
        self._kenlm_enabled = False
        self._dll_directory_handles: list[object] = []

    def _require_files(self) -> None:
        required_paths = (
            self._model_path / "model.onnx",
            self._model_path / "vocab.json",
            self._vit5_path / "encoder_model.onnx",
            self._vit5_path / "decoder_model.onnx",
            self._vit5_path / "tokenizer.json",
            self._hotwords_path,
        )
        missing = [str(path) for path in required_paths if not path.is_file()]
        if missing:
            raise FileNotFoundError("Missing ASR assets: " + ", ".join(missing))

    def _load_models(self) -> None:
        if self._asr_session is not None:
            return

        self._require_files()
        try:
            import onnxruntime as ort
            from pyctcdecode import build_ctcdecoder
            from tokenizers import Tokenizer
        except ImportError as error:
            raise RuntimeError(
                "ASR runtime dependencies are missing. Run `uv sync` in backend/."
            ) from error

        providers = ["CPUExecutionProvider"]
        if settings.ASR_USE_GPU and "CUDAExecutionProvider" in ort.get_available_providers():
            dll_directories = []
            for directory in (
                settings.ASR_CUDA_DLL_PATH,
                settings.ASR_CUDNN_DLL_PATH,
            ):
                if directory and Path(directory).is_dir() and hasattr(os, "add_dll_directory"):
                    self._dll_directory_handles.append(os.add_dll_directory(directory))
                    dll_directories.append(directory)
            if dll_directories:
                os.environ["PATH"] = os.pathsep.join(
                    [*dll_directories, os.environ.get("PATH", "")]
                )
            providers.insert(0, "CUDAExecutionProvider")
        logger.info(
            "Loading Wav2Vec2 ONNX and ViT5 ONNX models for ASR with %s.",
            providers[0],
        )
        wav2vec2_providers = providers if settings.ASR_WAV2VEC2_USE_GPU else [
            "CPUExecutionProvider"
        ]
        self._asr_session = ort.InferenceSession(
            str(self._model_path / "model.onnx"), providers=wav2vec2_providers
        )
        self._vit5_encoder = ort.InferenceSession(
            str(self._vit5_path / "encoder_model.onnx"),
            providers=providers,
        )
        self._vit5_decoder = ort.InferenceSession(
            str(self._vit5_path / "decoder_model.onnx"),
            providers=providers,
        )
        self._tokenizer = Tokenizer.from_file(str(self._vit5_path / "tokenizer.json"))
        self._hotwords = self._read_hotwords(self._hotwords_path)

        labels = self._load_ctc_labels(self._model_path / "vocab.json")
        language_model_path: str | None = None
        if self._kenlm_path.is_file():
            try:
                import kenlm  # noqa: F401
            except ImportError as error:
                if settings.ASR_REQUIRE_KENLM:
                    raise RuntimeError(
                        "KenLM Python binding is required but is not installed. "
                        "Install kenlm, then restart the backend."
                    ) from error
                logger.warning(
                    "KenLM binary is present but its Python binding is unavailable; "
                    "continuing with CTC hotword boosting only."
                )
            else:
                language_model_path = str(self._kenlm_path)
                self._kenlm_enabled = True
        elif settings.ASR_REQUIRE_KENLM:
            raise FileNotFoundError(f"KenLM model not found: {self._kenlm_path}")
        else:
            logger.warning("KenLM model not found; continuing without language model.")

        self._ctc_decoder = build_ctcdecoder(
            labels,
            kenlm_model_path=language_model_path,
            unigrams=self._hotwords,
        )
        logger.info(
            "ASR pipeline loaded (%d drug hotwords, KenLM=%s).",
            len(self._hotwords),
            self._kenlm_enabled,
        )

    @staticmethod
    def _load_ctc_labels(vocab_path: Path) -> list[str]:
        vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
        model_config = json.loads(vocab_path.with_name("config.json").read_text("utf-8"))
        labels = [""] * model_config["vocab_size"]
        for token, index in vocab.items():
            labels[index] = {
                "<pad>": "",
                "<unk>": "⁇",
                "|": " ",
            }.get(token, token)
            if token == "<unk>":
                labels[index] = "\u2047"
        for index, token in enumerate(
            ("\ue000", "\ue001"), start=max(vocab.values()) + 1
        ):
            labels[index] = token
        return labels

    @staticmethod
    def _read_hotwords(path: Path) -> list[str]:
        return sorted(
            {
                line.strip().lower()
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            }
        )

    async def transcribe(self, audio_bytes: bytes) -> str:
        """Decode a browser audio upload and run the complete ASR pipeline."""
        return await asyncio.to_thread(self._transcribe_blocking, audio_bytes)

    def _transcribe_blocking(self, audio_bytes: bytes) -> str:
        with self._lock:
            self._load_models()
            waveform = self._decode_audio(audio_bytes)
            logits = self._run_wav2vec2(waveform)
            raw_text = self._decode_ctc(logits)
            return self._rewrite_with_vit5(raw_text)

    @staticmethod
    def _decode_audio(audio_bytes: bytes):
        try:
            import av
            import numpy as np
        except ImportError as error:
            raise RuntimeError("Audio decoding dependencies are missing. Run `uv sync`.") from error

        try:
            with av.open(io.BytesIO(audio_bytes), mode="r") as container:
                stream = container.streams.audio[0]
                resampler = av.audio.resampler.AudioResampler(
                    format="flt", layout="mono", rate=16000
                )
                chunks = []
                for frame in container.decode(stream):
                    chunks.extend(resampler.resample(frame))
                chunks.extend(resampler.resample(None))
        except (IndexError, av.FFmpegError) as error:
            raise ValueError("The uploaded file is not a supported audio stream.") from error

        if not chunks:
            raise ValueError("The uploaded audio contains no samples.")
        waveform = np.concatenate([chunk.to_ndarray().reshape(-1) for chunk in chunks])
        if waveform.size < 400:
            raise ValueError("The uploaded audio is too short to transcribe.")
        waveform = waveform.astype(np.float32, copy=False)
        return (waveform - waveform.mean()) / np.sqrt(waveform.var() + 1e-7)

    def _run_wav2vec2(self, waveform):
        import numpy as np

        return self._asr_session.run(  # type: ignore[union-attr]
            ["logits"], {"input_values": np.expand_dims(waveform, axis=0)}
        )[0]

    def _decode_ctc(self, logits) -> str:
        text = self._ctc_decoder.decode(  # type: ignore[union-attr]
            logits[0],
            beam_width=settings.ASR_BEAM_WIDTH,
            hotwords=self._hotwords,
            hotword_weight=settings.ASR_HOTWORD_WEIGHT,
        )
        return " ".join(text.replace("\ue000", "").replace("\ue001", "").split())

    def _rewrite_with_vit5(self, text: str) -> str:
        import numpy as np

        if not text:
            return text
        encoded = self._tokenizer.encode(f"fix_asr: {text}")  # type: ignore[union-attr]
        input_ids = np.array([encoded.ids], dtype=np.int64)
        attention_mask = np.ones_like(input_ids, dtype=np.int64)
        hidden_states = self._vit5_encoder.run(  # type: ignore[union-attr]
            ["last_hidden_state"],
            {"input_ids": input_ids, "attention_mask": attention_mask},
        )[0]
        generated_ids = [0]
        for _ in range(settings.ASR_MAX_REWRITE_TOKENS):
            decoder_ids = np.array([generated_ids], dtype=np.int64)
            logits = self._vit5_decoder.run(  # type: ignore[union-attr]
                ["logits"],
                {
                    "encoder_attention_mask": attention_mask,
                    "input_ids": decoder_ids,
                    "encoder_hidden_states": hidden_states,
                },
            )[0]
            next_token_id = int(logits[0, -1].argmax())
            if next_token_id == 1:
                break
            generated_ids.append(next_token_id)
        rewritten = self._tokenizer.decode(  # type: ignore[union-attr]
            generated_ids[1:], skip_special_tokens=True
        ).strip()
        if rewritten.lower().startswith("fix_asr:"):
            rewritten = rewritten[len("fix_asr:") :].lstrip()
        return rewritten or text
