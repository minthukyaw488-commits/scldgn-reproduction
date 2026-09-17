# SCLDGN reproduction — BCI Competition IV 2a

An independent, end-to-end reproduction of

> Zhi, H. et al. **"Supervised Contrastive Learning-Based Domain Generalization Network
> for Cross-Subject Motor Decoding."** *IEEE Transactions on Biomedical Engineering*
> 72(1), 405–415, 2025. Code: <https://github.com/hongyizhi/SCLDGN>

run from the raw competition recordings using the authors' own released code and
preprocessing, on all 9 leave-one-subject-out folds.

> **This repository contains no third-party code.** SCLDGN is cloned during setup.
> What is here is the reproduction: results, the patches needed to run it off CUDA,
> the scripts that fill the gaps in the released pipeline, and the logs.

---

## Result

| | mean | std |
|---|---|---|
| **This reproduction** | **69.48 %** | ± 9.80 |
| Paper, Table II, Dataset I | 69.85 % | ± 9.16 |
| Difference | **−0.37 pp** | +0.64 |

| subject | A01 | A02 | A03 | A04 | A05 | A06 | A07 | A08 | A09 |
|---|---|---|---|---|---|---|---|---|---|
| test acc (%) | 78.30 | 57.64 | 82.64 | 56.08 | 62.50 | 62.85 | 76.22 | 76.56 | 72.57 |

Full table, including per-fold training and validation accuracy, selected epoch and
wall-clock time: [`RESULTS.csv`](RESULTS.csv).

**The reported accuracy reproduces**, and so does its per-subject spread — which is
the stronger of the two claims, since it means the variability across subjects behaves
the same way, not just the average.

Total compute: 50.5 hours on an Apple M4 (MPS backend). The authors used an NVIDIA A100.

---

## Configuration

Exactly the authors' `train.py` entry point, unchanged apart from the fold range:

```python
ho(datasetId=0, network='B7', batchSize=32, feature=32, subTorun=[0, 9],
   dropoutP=0., c=0.5, isProj=True,
   tradeOff=1, tradeOff2=0.1, maxEpochs=200, sma=100, algorithm='smcldgn')
```

| | |
|---|---|
| Data | BCI IV 2a — 9 subjects, 4 classes, 22 EEG channels, 2 sessions each |
| Window | 2–6 s after cue at 250 Hz → 1000 samples |
| Preprocessing | Euclidean Alignment per session, then a 9-band Chebyshev-II filter bank (4–40 Hz) |
| Network | `B7` multi-scale depthwise CNN, wrapped in `ERM_SMA` (weight averaging from iteration 100) |
| Loss | `smcldgn` = CE + 1.0 · CORAL + 0.1 · supervised-contrastive (domain-agnostic mixup) |
| Protocol | LOSO; the 8 source subjects split 80/20 per class per session |
| Model selection | lowest validation inaccuracy on **source** subjects only — the held-out subject is evaluated once, at the end |
| Optimiser | Adam, lr 1e-3, batch 32, 200 epochs |

---

## Reproducing this

### 1. Get the code and the data

```bash
git clone https://github.com/hongyizhi/SCLDGN.git
mkdir -p data/bci42a/originalData
```

Download the 18 BCI Competition IV 2a `.gdf` files and place them in
`data/bci42a/originalData/`.

### 2. Apply the compatibility patches

Only needed off CUDA — on an NVIDIA GPU, skip this step.

```bash
cd SCLDGN && git apply ../patches/mps-compat.patch && cd ..
```

### 3. Build the label files

The official loader reads per-session `A0xT.mat` / `A0xE.mat` files with a `classlabel`
field. **These are not part of the standard dataset download**, and the repository does
not say where to get them. Either obtain the competition's true-label release, or
rebuild them from labels you already hold:

```bash
python scripts/make_label_mats.py \
    data/bci42a/originalData data/bci42a/originalData labels.npz
```

The script verifies its output: the 9 training-session label sequences must match the
cue codes (769–772) embedded in the GDF annotations, and every session must carry
exactly 72 trials per class. It refuses to write anything if either check fails.

### 4. Run

```bash
cd SCLDGN && python train.py
```

The first run parses the data — GDF → Euclidean Alignment → filter bank — into
`data/bci42a/multiviewPython/` (≈ 3.8 GB), then trains. Subsequent runs reuse it.

### 5. Collect the results

```bash
python scripts/consolidate_results.py SCLDGN/output/bci42a RESULTS.csv
```

### Dependencies

`mne`, `scipy`, `numpy`, `torch`, `resampy`, `xlwt`, `higher`.
The upstream repository ships no requirements file; these were found by running it.

---

## The patches

Four edits, all of them device plumbing — see [`patches/mps-compat.patch`](patches/mps-compat.patch).
No loss, network, or training logic is touched.

| # | File | Change | Why |
|---|---|---|---|
| 1 | `baseModel/baseModel.py` | add an MPS branch to `setDevice()` | the code selects CUDA-or-CPU only |
| 2 | `lossFunction/scl.py` | `device = features.device` | device was hardcoded to CUDA |
| 3 | `network/MixedConv2d.py` | `.contiguous()` on the grouped-conv split | MPS rejects non-contiguous views (forward) |
| 4 | `baseModel/baseModel.py` | `.contiguous()` after `permute(0,3,1,2)`, 11 sites | the same, in the backward pass |

---

## Notes for anyone else reproducing this

Four things in the released pipeline are not described in the paper. None of them
change the conclusion, but each one silently changes the pipeline for someone working
from the paper text alone.

1. **Euclidean Alignment is applied inside the data parser**
   (`dataset/saveData.py`, via `alignOperation(x, operation='svd')`), not in the
   preprocessing described alongside `train.py`. A reader following the training script
   would not apply it. Verified on the parser's output: mean covariance against
   identity, ‖R−I‖/‖I‖ = 0.0000.

2. **The epoch count disagrees with the code.** The paper states 100 training epochs;
   `train.py` defaults to `maxEpochs=200`. This reproduction used 200 — the code's own
   default.

3. **The label `.mat` files are required but not distributed**, and nothing in the
   repository says where to obtain them. See step 3 above.

4. **The code assumes CUDA.** Tensors reach the network as non-contiguous views from
   `permute()`, which other backends reject in the convolution backward pass.

---

## Two observations from the per-fold numbers

- **Training accuracy is flat at ≈ 89 % across all nine folds, while test accuracy
  ranges from 56 % to 83 %.** The model fits the source subjects equally easily every
  time; what varies is entirely whether that fit transfers.

- **Source-subject validation accuracy does not predict target accuracy.** A02 has the
  highest validation accuracy of any fold (83.12 %) and the lowest test accuracy
  (57.64 %). Since model selection is made on that validation set, this is a structural
  difficulty of the domain-generalization setting rather than a flaw in this run.

---

## Licence

The scripts, patches and documentation in this repository are released under the MIT
licence. They are original work.

SCLDGN itself is **not** included here and is **not** covered by that licence — it is
cloned from its own repository during setup, and its terms are its authors' to state.
At the time of writing, that repository carries no licence file.

Please cite the original paper, not this repository, for the method.
