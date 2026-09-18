# SCLDGN reproduction — BCI Competition IV

An independent reproduction of

> Zhi, H. et al. **"Supervised Contrastive Learning-Based Domain Generalization Network
> for Cross-Subject Motor Decoding."** *IEEE Transactions on Biomedical Engineering*
> 72(1), 405–415, 2025. Code: <https://github.com/hongyizhi/SCLDGN>

run from the raw competition recordings using the authors' own released code and
preprocessing, plus two evaluations the paper does not report: a cross-dataset transfer
to BCI IV 2b, and an attempt to improve that transfer with the channel-mapping stage of
AFPM.

> SCLDGN is **not vendored here** — it is cloned during setup. What this repository
> holds is the reproduction: results, the patches needed to run the code off CUDA, the
> scripts that fill the gaps in the released pipeline, and the logs. The one block of
> transcribed upstream logic (the `smcldgn` objective, needed by the cross-dataset
> driver) is marked as such where it appears.

---

## Results

### 1. Within-dataset — BCI IV 2a, leave-one-subject-out

| | mean | std |
|---|---|---|
| **This reproduction** | **69.48 %** | ± 9.80 |
| Paper, Table II, Dataset I | 69.85 % | ± 9.16 |
| Difference | **−0.37 pp** | +0.64 |

| subject | A01 | A02 | A03 | A04 | A05 | A06 | A07 | A08 | A09 |
|---|---|---|---|---|---|---|---|---|---|
| test acc (%) | 78.30 | 57.64 | 82.64 | 56.08 | 62.50 | 62.85 | 76.22 | 76.56 | 72.57 |

Per-fold training and validation accuracy, selected epoch and wall-clock time:
[`RESULTS.csv`](RESULTS.csv).

**The reported accuracy reproduces**, and so does its per-subject spread — which is the
stronger of the two claims, since it means the variability across subjects behaves the
same way, not just the average.

### 2. Cross-dataset — train on 2a, test on 2b

The paper evaluates within each dataset. This asks a harder question: does a model
trained on one dataset's subjects work on a *different dataset's* subjects, with no
target data of any kind?

| | |
|---|---|
| Train | BCI IV **2a**, all 9 subjects as 9 domains, left/right only |
| Test | BCI IV **2b**, 9 subjects, 3680 trials — never seen in training or model selection |
| Shared input | C3, Cz, C4 · 4 s from cue · 250 Hz · 9-band filter bank, EA per session |

| | |
|---|---|
| **Target accuracy** | **62.43 %** ± 3.89 (per subject) |
| Pooled | 62.47 % — 95 % CI [60.86, 64.09] |
| Chance | 50.00 % (binary) |
| **Subjects above chance** | **9 / 9**, every one at *p* < 0.001 |

| subject | B01 | B02 | B03 | B04 | B05 | B06 | B07 | B08 | B09 |
|---|---|---|---|---|---|---|---|---|---|
| test acc (%) | 58.25 | 59.50 | 63.50 | 62.86 | 60.00 | 63.00 | 69.25 | 67.27 | 58.25 |

Per-subject counts and binomial tests: [`cross_dataset/per_subject.csv`](cross_dataset/per_subject.csv).

Transfer is well above chance on every target subject, but far below the 69.48 % the
same recipe reaches inside 2a — and the two are not directly comparable anyway, since
the cross-dataset task is binary on three channels while the within-dataset task is
four-class on twenty-two.

### 3. Can a richer shared montage improve that transfer? — no

The cross-dataset run above throws away 19 of 2a's 22 channels, because 2b has only
three. The channel-mapping stage of AFPM offers a way to keep them:

> Chen, Li & Wu, **"Alignment-Free Plug-and-Play Transfer Learning for
> Cross-Dataset EEG-Based Brain-Computer Interfaces"** (arXiv:2507.11911)

Both datasets are mapped onto a fixed 17-channel motor template
(FC3…CP4), each dataset filling the slots it carries and zero-filling the rest. 2a
fills all seventeen; 2b fills three. Everything else is held fixed — same network, same
loss, same schedule, same model selection — so only the input montage changes.

Mapping runs *after* Euclidean Alignment, not before: a zero-filled trial has a
singular covariance, so EA has to see the real channels.

| | intersection (3 ch) | **AFPM template (17 ch)** | difference |
|---|---|---|---|
| source validation | 81.48 % | **91.48 %** | **+10.00 pp** |
| **target 2b** | 62.43 % | **63.51 %** | **+1.08 pp** |

| subject | B01 | B02 | B03 | B04 | B05 | B06 | B07 | B08 | B09 |
|---|---|---|---|---|---|---|---|---|---|
| intersection | 58.25 | 59.50 | 63.50 | 62.86 | 60.00 | 63.00 | 69.25 | 67.27 | 58.25 |
| AFPM | 56.50 | 58.75 | 59.50 | **76.90** | 63.33 | 65.25 | 61.50 | 68.86 | 61.00 |
| difference | −1.75 | −0.75 | −4.00 | **+14.04** | +3.33 | +2.25 | −7.75 | +1.59 | +2.75 |

**The target gain is not significant**: paired *t*(8) = 0.535, *p* = 0.607; Wilcoxon
*p* = 0.734; sign test 5/9 up, *p* = 1.000; Cohen's *d* = 0.178. The two pooled
confidence intervals overlap across most of their range ([62.06, 65.20] against
[60.89, 64.04]). Excluding B04, the one subject that moved, the mean difference is
**−0.54 pp**. Full tests: [`cross_dataset/afpm_comparison.csv`](cross_dataset/afpm_comparison.csv).

Why the mapping does not pay: it fixes one distribution mismatch and introduces
another. The model now trains on trials with 17 of 17 slots filled and is tested on
trials with 3 of 17 filled, so its spatial filters and BatchNorm statistics are fitted
to an input density the target never has.

### What the two cross-dataset runs say together

The interesting quantity is not the 1.08 pp. It is the ratio:

```
source validation   81.48 %  ->  91.48 %      +10.00 pp
target 2b           62.43 %  ->  63.51 %       +1.08 pp
```

Fourteen extra channels of real signal bought a large improvement in fitting the source
subjects and almost nothing in transferring to new ones. The same asymmetry shows up
inside 2a: across the nine LOSO folds of §1, source-validation accuracy and target
accuracy correlate at **r = −0.86** — the folds the model fits best are the ones it
transfers to worst.

That is a statement about *model selection*, not about capacity. The released recipe
selects its checkpoint on source validation, which is the only labelled data a
zero-calibration deployment has; these two runs suggest that criterion is close to
uninformative about the quantity anyone actually cares about. An oracle measurement of
how much it costs is the obvious next experiment, and is in progress.

---

Compute: ~52 h for §1 and §2, ~2.5 h for §3, on an Apple M4 (MPS backend). The authors
used an NVIDIA A100.

---

## Configuration

### Within-dataset

The authors' `train.py` entry point, unchanged apart from the fold range:

```python
ho(datasetId=0, network='B7', batchSize=32, feature=32, subTorun=[0, 9],
   dropoutP=0., c=0.5, isProj=True,
   tradeOff=1, tradeOff2=0.1, maxEpochs=200, sma=100, algorithm='smcldgn')
```

| | |
|---|---|
| Data | BCI IV 2a — 9 subjects, 4 classes, 22 channels, 2 sessions each |
| Window | 2–6 s after cue at 250 Hz → 1000 samples |
| Preprocessing | Euclidean Alignment per session, then a 9-band Chebyshev-II filter bank (4–40 Hz) |
| Network | `B7` multi-scale depthwise CNN in `ERM_SMA` (weight averaging from iteration 100) |
| Loss | `smcldgn` = CE + 1.0 · CORAL + 0.1 · supervised-contrastive (domain-agnostic mixup) |
| Protocol | LOSO; the 8 source subjects split 80/20 per class per session |
| Model selection | lowest validation inaccuracy on **source** subjects only |
| Optimiser | Adam, lr 1e-3, batch 32, 200 epochs |

### Cross-dataset

Same network, same loss, same weights, same schedule. Two things necessarily differ:

- **9 domains, not 8.** Every 2a subject is a source here, because the held-out
  subjects live in the other dataset. Batch size follows at 36 = 9 × 4, preserving the
  four trials per domain per batch that 32 = 8 × 4 gives in the official run.
- **Channels and classes are the intersection of the two datasets** — C3/Cz/C4, and
  left vs right hand. `--channels afpm` substitutes the 17-channel AFPM template for
  the intersection; nothing else changes.

Model selection still uses a held-out split of the source subjects. The target dataset
is read exactly once, after training ends.

---

## Reproducing this

### 1. Code and data

```bash
git clone https://github.com/hongyizhi/SCLDGN.git
mkdir -p data/bci42a/originalData
```

Download the 18 BCI IV 2a `.gdf` files into `data/bci42a/originalData/`. For the
cross-dataset run you also need the BCI IV 2b `.gdf` files.

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

Within-dataset:

```bash
cd SCLDGN && python train.py
python scripts/consolidate_results.py SCLDGN/output/bci42a RESULTS.csv
```

Cross-dataset:

```bash
python cross_dataset/build_cross_2a_2b.py     # parse both datasets into one npz
python cross_dataset/train_cross_2a_2b.py     # train on 2a, evaluate on 2b
```

With the AFPM channel mapping (§3):

```bash
python cross_dataset/build_cross_2a_2b.py --channels afpm --out cross_2a_2b_afpm.npz
DATA_OVERRIDE=cross_2a_2b_afpm.npz python cross_dataset/train_cross_2a_2b.py \
    --out afpm_RESULTS.csv
```

The build script exposes `--channels`, `--ea-scope`, `--source-2a` and `--filter` so
individual preprocessing choices can be varied one at a time.

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

## Notes on the released pipeline

Things that are not described in the paper but change what the code does. None of them
alter the conclusion above, but each one would silently change the pipeline for someone
working from the paper text alone.

1. **Euclidean Alignment is applied inside the data parser**
   (`dataset/saveData.py`, via `alignOperation(x, operation='svd')`), not in the
   preprocessing described alongside `train.py`. Verified on the parser's output: mean
   covariance against identity, ‖R−I‖/‖I‖ = 0.0000.

2. **The 2a and 2b paths differ.** `parseBci42aFile` applies that alignment;
   `parseBci42bFile` does not. Applying EA to only one side of a cross-dataset
   comparison would guarantee a distribution mismatch, so the build script here applies
   it to both — which is also what the paper describes.

3. **The epoch count disagrees with the code.** The paper states 100 training epochs;
   `train.py` defaults to `maxEpochs=200`. This reproduction used 200 — the code's own
   default.

4. **The label `.mat` files are required but not distributed.** See step 3 above.

5. **Trial-start event ids are hardcoded per file** and assume one MNE version's
   annotation ordering. The cross-dataset build epochs on the cue annotation instead,
   which is equivalent and version-independent.

6. **The code assumes CUDA.** Tensors reach the network as non-contiguous views from
   `permute()`, which other backends reject in the convolution backward pass.

---

## Licence

The scripts, patches and documentation here are released under the MIT licence; they
are original work. The `smcldgn` loss body inside
[`cross_dataset/train_cross_2a_2b.py`](cross_dataset/train_cross_2a_2b.py) is
transcribed from SCLDGN's `baseModel.py` and is marked in place — it belongs to that
project's authors, not to this licence.

SCLDGN itself is cloned from its own repository during setup. At the time of writing
that repository carries no licence file.

Please cite the original paper, not this repository, for the method.
