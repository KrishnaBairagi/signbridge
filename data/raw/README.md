# Public dataset workflow (preferred)

For real training, use a public sign-language dataset instead of recording from your own camera.

The project already recommends the AI4Bharat INCLUDE / INCLUDE-50 dataset in [ml/datasets/DATASETS.md](../ml/datasets/DATASETS.md).

## Recommended open-source datasets

### INCLUDE-50 (fastest start)
- Website: https://sign-language.ai4bharat.org/
- Zenodo: https://zenodo.org/records/4010759
- GitHub: https://github.com/AI4Bharat/INCLUDE

### Full INCLUDE
- Same source as above
- Larger and better for broader coverage

## Expected local layout after download

```text
data/
  raw/
    INCLUDE_50/
      train/
        label_001/
          sample_001.mp4
      val/
      test/
```

Or, if you are using a custom subset:

```text
data/
  raw/
    HELLO/
      sample_001.mp4
      sample_002.mp4
    THANK_YOU/
      sample_001.mp4
    HELP/
      sample_001.mp4
```

## Use the project pipeline

Once the public clips are in place:

```bash
python -m ml.training.extract_landmarks --dataset_root data/raw --out_dir data/landmarks
python -m ml.training.prepare_dataset --landmarks_dir data/landmarks --out data/prepared.npz
python -m ml.training.train --data data/prepared.npz --model lstm --out checkpoints/model.pt
```

## Webcam recording is only a fallback

Custom camera recording is optional only when:
- the public dataset does not match your vocabulary closely enough
- you are collecting a small custom vocabulary for a demo or lab use case

For real-world training, prefer public datasets over local recordings.
