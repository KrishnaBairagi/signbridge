# Datasets for training the production SignBridge model

The demo heuristic classifier in `backend/sign_language/recognizer.py` is
**not** trained on any dataset. To train a real model, use one of the
following publicly documented, real datasets (do not substitute an
invented/fictional dataset).

## Primary recommendation: INCLUDE / INCLUDE-50 (Indian Sign Language)

- **Name:** INCLUDE (Indian Lexicon Sign Language Dataset)
- **Source:** AI4Bharat, IIT Madras — https://sign-language.ai4bharat.org/, dataset hosted on Zenodo: https://zenodo.org/records/4010759, code: https://github.com/AI4Bharat/INCLUDE
- **License:** Released by AI4Bharat for research use — check the Zenodo record and repo for the exact license terms before any commercial use.
- **Classes:** 263 word signs across 15 categories (INCLUDE); a 50-class subset, INCLUDE-50, is provided for faster iteration.
- **Samples:** 4,292 videos total (3,475 train / 817 test) for INCLUDE; 766 train / 192 test for INCLUDE-50; ~0.27 million frames across the full set.
- **Type:** Isolated signs (one sign per video), not continuous signing.
- **Signers:** Deaf students from St. Louis School for the Deaf, Adyar, Chennai — a real ISL source, recorded in natural (non-lab-only) conditions.
- **Limitations:** Isolated-sign only (no continuous sentence data), a specific regional ISL variant (India does not have one single standardized sign language), limited signer diversity (fixed group of student signers), so generalization to other signers/regions should be validated.
- **Why recommended:** It is the only large, ISL-specific, publicly documented dataset with an existing reference training pipeline (`AI4Bharat/INCLUDE` on GitHub already includes a MediaPipe-landmark extraction script + Transformer/other baselines), which lines up directly with this project's `ml/training/` scripts.

## Alternative / supplementary: CISLR (Corpus for Indian Sign Language Recognition)

- Largest-vocabulary ISL dataset (~4,765 sign words, ~7,050 videos), but very few samples per class — usable for one-shot/few-shot approaches, not standard supervised training at scale.

## If cross-lingual transfer is useful: WLASL / MSASL (American Sign Language)

- WLASL: 2,000 ASL signs, ~21K videos, 119 signers.
- MSASL: 1,000 ASL signs, ~25K videos, 222 signers.
- Useful for pretraining a general sign-recognition encoder (hand-shape/motion features transfer reasonably well across sign languages even though the lexicons differ), then fine-tuning on ISL data. The INCLUDE paper itself reports fine-tuning ISL-trained decoders on an ASL dataset (ASLLVD) as a generalization check.

## Recommended path for this project's limited vocabulary

The spec's demo vocabulary (HELLO, YES, NO, THANK YOU, PLEASE, HELP, WATER,
FOOD, DOCTOR, HOSPITAL, WHERE, WHAT, WHO, I, YOU, NEED, GO, STOP, GOOD, BAD,
SORRY) does not map one-to-one onto INCLUDE's 263/50 classes. Two options:

1. **Filter INCLUDE** down to the subset of its 263 signs that match or are
   close synonyms of the target vocabulary, and fine-tune on just those classes.
2. **Collect a small custom dataset** for the exact target vocabulary (see
   `custom_dataset_collection_procedure` below) — recommended if INCLUDE's
   classes don't cover the vocabulary well enough, since consistent
   demo/production vocabulary matters more than dataset size at this scale.

## Custom dataset collection procedure (if needed)

1. Recruit 5-10 ISL signers (ideally through a Deaf organization/school —
   never scrape signs from uncredited YouTube videos without consent).
2. For each of the ~21 target words, record 15-20 repetitions per signer
   (different angles/lighting/speed) using `ml/training/extract_landmarks.py`'s
   companion recording script (to be added) or any phone camera.
3. Run `ml/training/extract_landmarks.py` to convert videos to landmark
   sequences immediately, then discard raw video (privacy + storage).
4. Split by SIGNER (not randomly) into train/val/test so the model is
   evaluated on unseen signers — this is the realistic deployment scenario.
5. Document consent obtained from every participant and store only
   de-identified landmark coordinates, never raw video, long-term.
