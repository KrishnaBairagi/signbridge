# SignBridge — Real-Time Sign Language Communication Assistant

A working prototype of a two-way, live communication system between a
sign-language user and a hearing person: sign → recognition → English →
speech, and speech → text → displayed to the signer. No file uploads —
everything runs on a live camera/microphone stream.

> **This is a demo prototype, not a certified interpreter.** The sign
> recognition model is a small, rule-based heuristic covering ~21 demo
> signs (see "Demo Mode vs. Production Model" below) — it is not a trained,
> evaluated model and should never be relied on for medical, legal, or
> emergency communication. See [ARCHITECTURE.md](ARCHITECTURE.md) for the
> system diagram and design rationale.

## 1. Features

- Live camera → MediaPipe hand/pose landmarks (extracted **in the browser**,
  so raw video never leaves the device)
- **61-Class Production Model** trained on 3,630 labeled sequences with full two-hand support (228 features/frame: left hand 63 + right hand 63 + pose 99 + 3 presence flags)
- Continuous rolling-window inference (48 frames sequence length, ~1.6s window at 30 FPS) with ONNX Runtime CPU acceleration (~3.63 ms/window)
- Temporal gesture segmentation: start/end detection, debouncing (2 matching predictions), and 0.8s cooldown so held signs aren't spammed
- Rule-based sentence building from recognized tokens, with an optional,
  hallucination-guarded LLM polishing step
- Browser text-to-speech (spoken output) and browser speech-to-text
  (near-continuous listening for the recipient's reply)
- WebSocket-based live two-way session (`/ws/live-conversation`)
- REST endpoints for testing recognition/translation in isolation
- A complete swappable ML interface: supports heuristic demo mode as well as trained production ONNX/PyTorch models
- A complete offline dataset pipeline and regression test suite (43 unit tests + 7 API integration tests)
- Consent-gated storage (nothing is saved by default)

## 2. Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full diagram. In short:

```
Browser: Camera → MediaPipe (landmarks) → WebSocket ⇄ FastAPI backend
                                                        → TemporalSignBuffer
                                                        → recognize_sign()
                                                        → Sentence Builder
                                          ⇄ WebSocket → Browser: speechSynthesis (speaks)

Browser: Microphone → Web Speech API (text) → WebSocket → Browser: displayed to signer
```

## 3. Technology stack

| Layer | Choice |
|---|---|
| Frontend | Self-contained HTML/JS (`frontend/index.html`), MediaPipe Tasks Vision (CDN), Web Speech API |
| Backend | Python, FastAPI, WebSocket |
| Computer vision | MediaPipe (client-side for live use; Python `mediapipe` for offline training) |
| ML | PyTorch (LSTM + Transformer architectures), ONNX Runtime for inference |
| Speech | Browser SpeechSynthesis (TTS) / SpeechRecognition (STT) by default; Azure/Google/Whisper pluggable |
| Database | SQLite |

**Why not the full Next.js frontend from the original brief?** A production
Next.js app needs `npm install` and a build step you'd run locally regardless
of who writes the code. A single static HTML file removes that friction for
a prototype — it opens directly in a browser with zero setup. Porting it into
a Next.js app later is mechanical (the WebSocket/MediaPipe logic doesn't
change, it just moves into React components).

## 4. Installation

```bash
git clone <this project>
cd signbridge
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

## 5. Running

**Backend:**
```bash
uvicorn backend.main:app --reload --port 8000
```

**Frontend:** just open `frontend/index.html` in a browser (Chrome
recommended — it has the best Web Speech API support). If your backend
isn't on `localhost:8000`, set `window.SIGNBRIDGE_WS_URL` in the page before
the script runs, or edit the `WS_URL` constant at the top of the `<script>`.

Grant camera and microphone permissions when prompted.

## 6. Demo Mode vs. Production Model

- **Demo Mode (`DEMO_MODE=true`, default):** uses
  `backend/sign_language/recognizer.py`'s `HeuristicSignClassifier` — simple
  geometric rules (finger curl, hand height relative to face/chest) that
  reliably distinguish only a handful of very distinct hand shapes
  (currently tuned for HELLO, YES, NO, WHERE/WHO, I, NEED — see the file for
  exact coverage). The UI shows a "Demo Recognition Model" badge at all times.
- **Production Model (`DEMO_MODE=false`):** loads the trained 61-class BiLSTM ONNX model (`checkpoints/full_61/model.onnx`).

### Production Model Validation Summary

| Metric | Result | Description |
| :--- | :---: | :--- |
| **Held-Out Offline Test Accuracy** | **91.53%** | Evaluated on 543 held-out sequence windows (497/543 correct, Macro F1: 0.8221). |
| **Dataset-Replay First-Event Accuracy** | **94.10%** | Full live pipeline replay across 305 evaluation videos (287/305 correct first event). |
| **Dataset-Replay Any-Event Accuracy** | **96.07%** | Full live pipeline replay across 305 evaluation videos (293/305 correct event emitted). |
| **ONNX Runtime CPU Latency** | **3.63 ms/window** | Measured single-window inference latency on CPU. |

*Note: Dataset-replay metrics evaluate complete video streams through the live temporal state machine and should not be conflated with real-world webcam performance.*

## 7. Dataset preparation & model training

See [ml/datasets/DATASETS.md](ml/datasets/DATASETS.md) for real, publicly
documented datasets (primary recommendation: **INCLUDE / INCLUDE-50**, an
Indian Sign Language dataset from AI4Bharat/IIT Madras — 263 word signs,
~4,300 videos) and a custom-collection procedure if the demo vocabulary
needs its own recordings.

```bash
# 1. Extract landmarks from labeled videos (data/raw/<LABEL>/*.mp4)
python -m ml.training.extract_landmarks --dataset_root data/raw --out_dir data/landmarks

# 2. Normalize + split into train/val/test
python -m ml.training.prepare_dataset --landmarks_dir data/landmarks --out data/prepared.npz

# 3. Train
python -m ml.training.train --data data/prepared.npz --model lstm --epochs 30 --out checkpoints/model.pt

# 4. Evaluate
python -m ml.training.evaluate --data data/prepared.npz --checkpoint checkpoints/model.pt

# 5. Export for production inference
python -m ml.training.export_model --checkpoint checkpoints/model.pt --out checkpoints/model.onnx
```

Then set `DEMO_MODE=false` in `.env` and restart the backend.

### Adding a new sign to the vocabulary

1. Record ≥15 samples of the new sign (ideally from multiple signers).
2. Add them under `data/raw/<NEW_SIGN>/` and re-run the pipeline above —
   the label set is derived automatically from folder names.
3. Update `DEMO_VOCAB` in `backend/sign_language/recognizer.py` and the
   word list in `frontend/index.html` for documentation purposes (this
   doesn't affect the trained model's actual output classes, which come
   from the exported labels file).

## 8. API documentation

`GET /health` → `{"status": "ok", "demo_mode": "true"}`

`POST /api/sign/predict` — body: `{"sequence": [<landmark frame>, ...]}` →
`{"sign": "HELP", "confidence": 0.94, "model": "demo"}`

`POST /api/translate` — body: `{"tokens": ["I", "NEED", "HELP"]}` →
`{"text": "I need help.", "method": "rule_based"}`

`POST /api/tts` — body: `{"text": "I need help."}` → in browser mode:
`{"mode": "browser", "text": "..."}` (client should call `speechSynthesis`);
in a cloud-provider mode: an audio stream.

`WS /ws/live-conversation` — see the message protocol documented at the top
of `backend/websocket/live_conversation.py`.

## 9. Testing

Run full unit & regression test suite (43 tests):
```bash
python -m unittest discover -s tests -p "test_*.py"
```

Run API and WebSocket integration tests (7 tests):
```bash
pytest tests/test_api.py -v
```

## 10. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| "Could not reach the SignBridge server" | Backend isn't running, or CORS/WS URL mismatch — check `CORS_ALLOWED_ORIGINS` and `WS_URL` in the frontend script |
| "No hands detected" constantly | Improve lighting; make sure the camera can see your hand clearly against the background |
| Speech recognition button does nothing | Use Chrome/Edge — Firefox does not implement `SpeechRecognition`; add a Whisper fallback via `STT_PROVIDER=whisper` if needed |
| Recognition is inaccurate / limited | Expected in Demo Mode — see §6 above and the training pipeline in §7 |

## 11. Privacy considerations

- Camera video and microphone audio are processed entirely in the browser
  and are **never sent to the backend** — only derived landmark coordinates
  and transcribed text cross the network.
- Nothing is written to the database unless the session explicitly sets
  `consent_to_store: true` when starting (see `backend/db/database.py`).
- No cloud AI provider is called unless you configure one in `.env`; by
  default the entire pipeline runs locally against the demo model plus
  browser-native speech APIs.

## 12. Future improvements

- Replace the heuristic demo classifier with a model trained on INCLUDE/
  INCLUDE-50 or a custom ISL dataset (pipeline is ready — see §7)
- Smarter end-of-utterance detection (currently simplified — see
  `ARCHITECTURE.md`'s "Known simplification" section)
- On-device model inference (e.g. via ONNX Runtime Web) to remove the
  landmark round-trip to the backend entirely
- Multi-signer support / signer-independent evaluation
- Authentication for multi-user deployments (the WebSocket currently has
  no auth — fine for a local prototype, not for production)
- Continuous (not just isolated) sign recognition
