"""Vietnamese medical ASR: Wav2Vec2 CTC, KenLM/hotwords, then ViT5 rewrite."""

import asyncio
import io
import json
import logging
import threading
from pathlib import Path

from src.config import settings

logger = logging.getLogger(__name__)


def _custom_wav2vec2_class():
    """Return the architecture used when the medical ASR checkpoint was trained."""
    from collections import OrderedDict

    from torch import nn
    from transformers import Wav2Vec2Model, Wav2Vec2PreTrainedModel

    class CustomWav2Vec2ForCTC(Wav2Vec2PreTrainedModel):
        all_tied_weights_keys = {}

        def __init__(self, config):
            super().__init__(config)
            self.wav2vec2 = Wav2Vec2Model(config)
            self.dropout = nn.Dropout(config.final_dropout)
            self.feature_transform = nn.Sequential(
                OrderedDict(
                    [
                        ("linear1", nn.Linear(config.hidden_size, config.hidden_size)),
                        ("bn1", nn.BatchNorm1d(config.hidden_size)),
                        ("activation1", nn.LeakyReLU()),
                        ("drop1", nn.Dropout(config.final_dropout)),
                        ("linear2", nn.Linear(config.hidden_size, config.hidden_size)),
                        ("bn2", nn.BatchNorm1d(config.hidden_size)),
                        ("activation2", nn.LeakyReLU()),
                        ("drop2", nn.Dropout(config.final_dropout)),
                        ("linear3", nn.Linear(config.hidden_size, config.hidden_size)),
                        ("bn3", nn.BatchNorm1d(config.hidden_size)),
                        ("activation3", nn.LeakyReLU()),
                        ("drop3", nn.Dropout(config.final_dropout)),
                    ]
                )
            )
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)

        def forward(self, input_values):
            hidden_states = self.dropout(self.wav2vec2(input_values).last_hidden_state)
            batch_size, sequence_length, hidden_size = hidden_states.shape
            hidden_states = self.feature_transform(
                hidden_states.reshape(batch_size * sequence_length, hidden_size)
            )
            return self.lm_head(
                hidden_states.reshape(batch_size, sequence_length, hidden_size)
            )

    return CustomWav2Vec2ForCTC


class ASRService:
    """Lazy, process-local implementation of the medical speech pipeline."""

    def __init__(self) -> None:
        self._model_path = settings.resolve(settings.ASR_MODEL_PATH)
        self._kenlm_path = settings.resolve(settings.ASR_KENLM_PATH)
        self._vit5_path = settings.resolve(settings.ASR_VIT5_MODEL_PATH)
        self._hotwords_path = settings.resolve(settings.ASR_HOTWORDS_PATH)
        self._lock = threading.Lock()
        self._asr_model: object | None = None
        self._vit5_model: object | None = None
        self._ctc_decoder: object | None = None
        self._tokenizer: object | None = None
        self._hotwords: list[str] = []
        self._kenlm_enabled = False

    def _require_files(self) -> None:
        required_paths = (
            self._model_path / "model.safetensors",
            self._model_path / "vocab.json",
            self._vit5_path / "model.safetensors",
            self._vit5_path / "tokenizer.json",
            self._hotwords_path,
        )
        missing = [str(path) for path in required_paths if not path.is_file()]
        if missing:
            raise FileNotFoundError("Missing ASR assets: " + ", ".join(missing))

    def _load_models(self) -> None:
        if self._asr_model is not None:
            return

        self._require_files()
        try:
            from pyctcdecode import build_ctcdecoder
            import torch
            from tokenizers import Tokenizer
            from transformers import AutoModelForSeq2SeqLM
        except ImportError as error:
            raise RuntimeError(
                "ASR runtime dependencies are missing. Run `uv sync` in backend/."
            ) from error

        device = "cuda" if settings.ASR_WAV2VEC2_USE_GPU and torch.cuda.is_available() else "cpu"
        rewrite_device = "cuda" if settings.ASR_USE_GPU and torch.cuda.is_available() else "cpu"
        logger.info("Loading custom Wav2Vec2 CTC on %s and ViT5 stage 2 on %s.", device, rewrite_device)
        self._asr_model = _custom_wav2vec2_class().from_pretrained(self._model_path).to(device).eval()
        self._tokenizer = Tokenizer.from_file(str(self._vit5_path / "tokenizer.json"))
        self._vit5_model = AutoModelForSeq2SeqLM.from_pretrained(self._vit5_path).to(rewrite_device).eval()
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
        for index, token in enumerate(("<s>", "</s>"), start=max(vocab.values()) + 1):
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
        import torch

        device = next(self._asr_model.parameters()).device  # type: ignore[union-attr]
        inputs = torch.from_numpy(waveform).unsqueeze(0).to(device)
        with torch.inference_mode():
            logits = self._asr_model(inputs)  # type: ignore[union-attr]
        return logits.detach().cpu().numpy()

    def _decode_ctc(self, logits) -> str:
        text = self._ctc_decoder.decode(  # type: ignore[union-attr]
            logits[0],
            beam_width=settings.ASR_BEAM_WIDTH,
            hotwords=self._hotwords,
            hotword_weight=settings.ASR_HOTWORD_WEIGHT,
        )
        return " ".join(text.replace("\ue000", "").replace("\ue001", "").split())

    def _rewrite_with_vit5(self, text: str) -> str:
        import torch

        if not text:
            return text
        device = next(self._vit5_model.parameters()).device  # type: ignore[union-attr]
        encoded = self._tokenizer.encode(f"fix_asr: {text}")  # type: ignore[union-attr]
        input_ids = torch.tensor([encoded.ids], device=device)
        attention_mask = torch.ones_like(input_ids)
        with torch.inference_mode():
            generated_ids = self._vit5_model.generate(  # type: ignore[union-attr]
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=settings.ASR_MAX_REWRITE_TOKENS,
                num_beams=4,
            )
        rewritten = self._tokenizer.decode(  # type: ignore[union-attr]
            generated_ids[0].tolist(), skip_special_tokens=True
        ).strip()
        if rewritten.lower().startswith("fix_asr:"):
            rewritten = rewritten[len("fix_asr:") :].lstrip()
        return rewritten or text
