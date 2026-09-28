"""
ml/inference/inference.py

TrainedSignClassifier: the production counterpart to
backend/sign_language/recognizer.py's HeuristicSignClassifier. It is loaded
lazily by recognize_sign() only when DEMO_MODE=false, so the demo prototype
has zero dependency on ONNX Runtime / a trained checkpoint being present.

Expects an exported ONNX model + labels JSON (see ml/training/export_model.py)
at the paths given by SIGNBRIDGE_ONNX_MODEL_PATH / SIGNBRIDGE_LABELS_PATH.
"""

from __future__ import annotations

import json
import os

import numpy as np

from backend.config import get_settings
from backend.logging_config import logger
from backend.sign_language.recognizer import SignPrediction, FrameLandmarks
from ml.preprocessing.normalize import normalize_frame, frames_to_feature_vector

DEFAULT_MODEL_PATH = get_settings().onnx_model_path
DEFAULT_LABELS_PATH = get_settings().labels_path
MAX_SEQ_LEN = get_settings().max_seq_len



class TrainedSignClassifier:
    def __init__(self, model_path: str | None = None, labels_path: str | None = None):
        s = get_settings()
        self.model_path = model_path or s.onnx_model_path
        self.labels_path = labels_path or s.labels_path
        self.use_pytorch = self.model_path.endswith(".pt") or self.model_path.endswith(".pth")

        if self.use_pytorch:
            import torch
            from ml.models.sequence_models import build_lstm_model, build_transformer_model

            if not os.path.exists(self.model_path):
                raise FileNotFoundError(
                    f"No PyTorch checkpoint at {self.model_path}. Run ml/training/train.py before setting DEMO_MODE=false."
                )

            ckpt = torch.load(self.model_path, map_location="cpu")
            self.labels = ckpt["labels"]
            arch = ckpt.get("architecture", "lstm")
            self.model = build_lstm_model(len(self.labels)) if arch == "lstm" else build_transformer_model(len(self.labels))
            self.model.load_state_dict(ckpt["model_state"])
            self.model.eval()
            self.torch = torch
            logger.info(f"Loaded PyTorch production model from {self.model_path}")
        else:
            try:
                import onnxruntime as ort
            except ImportError as e:
                raise ImportError(
                    "TrainedSignClassifier ONNX mode requires onnxruntime. Install with: pip install onnxruntime"
                ) from e

            if not os.path.exists(self.model_path):
                raise FileNotFoundError(
                    f"No exported ONNX model at {self.model_path}. Run training & export pipeline before setting DEMO_MODE=false."
                )

            self.session = ort.InferenceSession(self.model_path)
            if os.path.exists(self.labels_path):
                with open(self.labels_path) as f:
                    self.labels = json.load(f)
            else:
                raise FileNotFoundError(f"No labels JSON file at {self.labels_path}.")
            logger.info(f"Loaded ONNX production model from {self.model_path} with {len(self.labels)} labels")

    def recognize(self, sequence: list[FrameLandmarks]) -> SignPrediction | None:
        max_seq_len = get_settings().max_seq_len
        if not sequence or len(sequence) < max_seq_len:
            return None

        vectors = [frames_to_feature_vector(normalize_frame(f)) for f in sequence]
        arr = np.stack(vectors).astype(np.float32)

        if len(arr) >= max_seq_len:
            arr = arr[:max_seq_len]

        batch = arr[np.newaxis, :, :]  # (1, seq_len, features)

        if self.use_pytorch:
            with self.torch.no_grad():
                tensor_input = self.torch.tensor(batch, dtype=self.torch.float32)
                logits = self.model(tensor_input)[0].numpy()
        else:
            input_name = self.session.get_inputs()[0].name
            logits = self.session.run(None, {input_name: batch})[0][0]

        exp = np.exp(logits - logits.max())
        probs = exp / exp.sum()
        top_idx = int(probs.argmax())

        return SignPrediction(sign=self.labels[top_idx], confidence=float(probs[top_idx]), model="production")


