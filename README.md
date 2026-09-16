# PA1

This repo has all four tasks for Programming Assignment 1. Each task lives
in its own folder: `task1/`, `task2/`, `task3/`, `task4/`.

## Setup

```
pip install torch torchvision open_clip_torch scikit-image scikit-learn pandas matplotlib seaborn pyyaml huggingface_hub pyarrow
```

Seed 6304 is used everywhere the assignment asks for a fixed seed, in every
task.

Tasks 2 and 3 share a `shared/` folder at the top of `PA1/`, so their commands
are run from inside `PA1/` itself instead of from inside the task folder.

## Task 1: Inductive Biases and Feature Representations

Folder: `task1/`

Compares three frozen pretrained backbones (ResNet-50, ViT-B/16, and CLIP
ViT-B/32) on STL-10 by training a linear head on top of each one, then
testing all three under grayscale, color swap, shape versus texture cue
conflict, translation, and patch shuffle interventions. Also looks at how
much each backbone's feature representation moves under these interventions
with cosine similarity and t-SNE plots.

To run everything from inside `task1/`:

```
python -m scripts.run_task1
```

or run each stage on its own, in this order:

```
python -m data.make_subset
python -m data.make_cue_conflicts
python -m scripts.extract_features
python -m scripts.train_heads
python -m analysis.evaluate_bias
python -m analysis.feature_similarity
python -m analysis.representation
```

STL-10 and every pretrained model download automatically the first time you
run it.

Results are saved in `results/tables/` (csv files) and `results/figures/`
(png files).

Notes:
- CLIP is loaded as `ViT-B-32-quickgelu` instead of plain `ViT-B-32` since
  that is the version that actually matches OpenAI's original weights.
- The AdaIN style transfer network and its pretrained weights come from
  `github.com/naoto0804/pytorch-AdaIN` (Huang and Belongie, 2017).
- ResNet-50 and ViT-B/16 use torchvision's pretrained weights, and CLIP uses
  the `open_clip_torch` library.
- If the official STL-10 download is slow, a mirror of the same file is
  available at `huggingface.co/datasets/xingslong/stl10_binary`.

## Task 2: Unsupervised Domain Adaptation

Folder: `task2/`, shared PACS loading code in `shared/`

Fine tunes a torchvision ResNet-18 on the three labeled PACS source domains
(Photo, Art Painting, Cartoon) and adapts it toward the unlabeled Sketch
domain with three alignment methods: DAN (MMD feature alignment), DANN
(adversarial alignment through a gradient reversal layer), and CDAN
(class conditional adversarial alignment). Compares all three against a
Source-only baseline on source validation accuracy, Sketch accuracy, a
domain separability probe, and per-class Sketch accuracy changes, then runs
a controlled study sweeping the DAN MMD loss weight.

To run everything from inside `PA1/`:

```
python -m task2.train
python -m task2.evaluate_final
python -m task2.evaluation.domain_separability
python -m task2.evaluation.controlled_study
```

or run each stage on its own, in this order:

```
python -m shared.pacs_protocol
python -m task2.methods.source_only
python -m task2.methods.dan
python -m task2.methods.dann
python -m task2.methods.cdan
python -m task2.evaluation.domain_separability
python -m task2.evaluation.class_analysis
python -m task2.evaluate_final
python -m task2.evaluation.controlled_study
```

Results are saved in `task2/results/tables/` (csv files), training curve
plots in `task2/results/figures/` (png files), and checkpoints in
`task2/results/checkpoints/` (pt files).

Every method fine tunes the full ResNet-18 backbone, which is heavy enough
that it needs a gpu to finish in reasonable time. The code picks up a gpu
automatically through `shared/device.py` and falls back to cpu when none is
available, so the same commands work locally or on Colab.

Running on Colab:

```
!git clone https://github.com/Mohidali2005/ATML_PA1.git
%cd ATML_PA1
!pip install -q huggingface_hub pyarrow
!python -m task2.train
!python -m task2.evaluate_final
!python -m task2.evaluation.controlled_study
```

Pick a T4 gpu runtime before running these cells. Once the run finishes zip
and download `task2/results` to bring the tables and checkpoints back to a
local clone of this repo.

Notes:
- PACS is loaded from the `flwrlabs/pacs` parquet mirror on huggingface
  since there is no official torchvision loader for it.
- Every method fine tunes the entire ResNet-18, not just a linear head, so
  each training run is far slower on a cpu only machine than the frozen
  backbone comparisons in task 1.
- BatchNorm running statistics stay frozen at their pretrained ImageNet
  values for every method so BatchNorm itself cannot silently perform its
  own form of adaptation underneath whatever method is being tested.
- The controlled design study sweeps the DAN MMD loss weight rather than the
  DANN gradient reversal strength, the two options the assignment allows
  choosing between.
- Every method saves a per epoch classification loss and validation macro F1
  curve, plus an alignment curve where applicable, an MMD loss curve for DAN
  and a domain discriminator accuracy curve for DANN and CDAN.
- CDAN's result is not stable across runs, two independent Colab runs of the
  identical code produced very different outcomes, since cuda determinism
  was never pinned. Both outcomes are documented in `CLAUDE.md`.
