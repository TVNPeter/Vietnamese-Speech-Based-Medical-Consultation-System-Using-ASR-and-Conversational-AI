"""Unit tests for model-independent medical ASR preparation."""

import json
import tempfile
import unittest
from pathlib import Path

from src.services.asr import ASRService


class ASRPreparationTests(unittest.TestCase):
    def test_ctc_labels_cover_onnx_vocab_size(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory)
            vocab_path = model_dir / "vocab.json"
            vocab_path.write_text(
                json.dumps({"|": 0, "a": 1, "<unk>": 2, "<pad>": 3}),
                encoding="utf-8",
            )
            (model_dir / "config.json").write_text(
                json.dumps({"vocab_size": 6}), encoding="utf-8"
            )

            labels = ASRService._load_ctc_labels(vocab_path)

        self.assertEqual(labels, [" ", "a", "⁇", "", "\ue000", "\ue001"])

    def test_hotwords_are_normalized_and_deduplicated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            hotwords_path = Path(directory) / "drugs.txt"
            hotwords_path.write_text("Aspirin\n\naspirin\nParacetamol\n", encoding="utf-8")

            hotwords = ASRService._read_hotwords(hotwords_path)

        self.assertEqual(hotwords, ["aspirin", "paracetamol"])
