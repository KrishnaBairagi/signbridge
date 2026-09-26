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

from backend.sign_language.recognizer import SignPrediction, FrameLandmarks
from ml.preprocessing.normalize import normalize_frame, frames_to_feature_vector

def _get_default_model_path() -> str:
    env_path = os.getenv("SIGNBRIDGE_ONNX_MODEL_PATH")
    if env_path:
        return env_path
    if os.path.exists("checkpoints/full_61/model.onnx"):
        return "checkpoints/full_61/model.onnx"
    if os.path.exists("checkpoints/full_61/model.pt"):
        return "checkpoints/full_61/model.pt"
    return "checkpoints/model.pt"


def _get_default_labels_path() -> str:
    env_path = os.getenv("SIGNBRIDGE_LABELS_PATH")
    if env_path:
        return env_path
    if os.path.exists("checkpoints/full_61/model_labels.json"):
        return "checkpoints/full_61/model_labels.json"
    return "checkpoints/model_labels.json"


DEFAULT_MODEL_PATH = _get_default_model_path()
DEFAULT_LABELS_PATH = _get_default_labels_path()
MAX_SEQ_LEN = int(os.getenv("SIGNBRIDGE_MAX_SEQ_LEN", "48"))


class TrainedSignClassifier:
    def __init__(self, model_path: str = DEFAULT_MODEL_PATH, labels_path: str = DEFAULT_LABELS_PATH):
        self.model_path = model_path
        self.labels_path = labels_path
        self.use_pytorch = model_path.endswith(".pt") or model_path.endswith(".pth")

        if self.use_pytorch:
            import torch
            from ml.models.sequence_models import build_lstm_model, build_transformer_model

            if not os.path.exists(model_path):
                raise FileNotFoundError(
                    f"No PyTorch checkpoint at {model_path}. Run ml/training/train.py before setting DEMO_MODE=false."
                )

            ckpt = torch.load(model_path, map_location="cpu")
            self.labels = ckpt["labels"]
            arch = ckpt.get("architecture", "lstm")
            self.model = build_lstm_model(len(self.labels)) if arch == "lstm" else build_transformer_model(len(self.labels))
            self.model.load_state_dict(ckpt["model_state"])
            self.model.eval()
            self.torch = torch
        else:
            try:
                import onnxruntime as ort
            except ImportError as e:
                raise ImportError(
                    "TrainedSignClassifier ONNX mode requires onnxruntime. Install with: pip install onnxruntime"
                ) from e

            if not os.path.exists(model_path):
                raise FileNotFoundError(
                    f"No exported ONNX model at {model_path}. Run training & export pipeline before setting DEMO_MODE=false."
                )

            self.session = ort.InferenceSession(model_path)
            if os.path.exists(labels_path):
                with open(labels_path) as f:
                    self.labels = json.load(f)
            else:
                raise FileNotFoundError(f"No labels JSON file at {labels_path}.")

    def recognize(self, sequence: list[FrameLandmarks]) -> SignPrediction | None:
        if not sequence or len(sequence) < MAX_SEQ_LEN:
            return None

        vectors = [frames_to_feature_vector(normalize_frame(f)) for f in sequence]
        arr = np.stack(vectors).astype(np.float32)

        if len(arr) >= MAX_SEQ_LEN:
            arr = arr[:MAX_SEQ_LEN]

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

