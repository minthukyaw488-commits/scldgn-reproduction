"""Build a matched 2a / 2b pair using the official SCLDGN preprocessing.

Both datasets go through the same three steps, in the same order, with the same code:

    epoch 4 s from the cue  ->  Euclidean Alignment per session  ->  9-band filter bank

Channels are restricted to the three 2b carries (C3, Cz, C4) and 2a is restricted to
its left/right classes, so the two datasets share a label space and an input shape.

Two deviations from `saveData.py`, both forced and both documented:

  * Its 2b path never calls `alignOperation`, while its 2a path does. Applying EA to
    only one side of a cross-dataset comparison would guarantee a distribution
    mismatch, so EA is applied to both here -- which is also what the paper describes.

  * Its trial-start event ids are hardcoded per file and assume one MNE version's
    annotation ordering (e.g. it expects id 2 for '768' in B0102T, where this MNE
    reports 3). Epoching on the cue annotation instead is equivalent -- the 2b cue
    sits 3 s after trial start, exactly the offset the official code adds -- and it
    yields the label at the same time.
"""
import argparse
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings('ignore')
MASTER = os.path.dirname(os.path.abspath(__file__)) + '/SCLDGN'
sys.path.insert(1, MASTER)

import mne
from dataset.saveData import parseBci42aFile, alignOperation
from utils.tools import get_transform
from utils import transforms

GDF_2A = '/Users/user/MotorImagery/BCICIV_2a_gdf'
MAT_2A = '/Users/user/MotorImagery/scldgn_official_run/data/bci42a/originalData'
GDF_2B = '/Users/user/MotorImagery/BCICIV_2b_gdf'
ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--ea-scope', choices=['session', 'subject'], default='session',
                help="session: EA per recording file, as parseBci42aFile does. "
                     "subject: one R_bar per subject, pooling that subject's sessions.")
ap.add_argument('--source-2a', choices=['official', 'senior'], default='official',
                help="official: re-parse the GDFs. senior: take the first 1000 samples "
                     "of the pre-exported raw 1.npz.")
ap.add_argument('--filter', choices=['official', 'reimpl'], default='official',
                help="official: utils/transforms.filterBank. reimpl: the standalone "
                     "cheby2 bank written for the first cross-dataset attempt.")
ap.add_argument('--channels', choices=['intersection', 'afpm'], default='intersection',
                help="intersection: the three channels both datasets carry (C3/Cz/C4). "
                     "afpm: map both onto the 17-channel motor template of Chen et al., "
                     "AFPM (arXiv:2507.11911), zero-filling what a dataset lacks.")
ap.add_argument('--out', default='/Users/user/MotorImagery/cross_2a_2b_official.npz')
ARGS = ap.parse_args()
OUT = ARGS.out

C3_CZ_C4 = [7, 9, 11]          # indices of C3, Cz, C4 in the 2a montage

# AFPM's task-specific channel set for motor imagery (Chen, Li & Wu, arXiv:2507.11911,
# Sec. IV-B-2): the primary motor cortex. 2a carries all seventeen; 2b carries three.
AFPM_TEMPLATE = ['FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz',
                 'C2', 'C4', 'C6', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4']
CH_2A = ['Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4',
         'C6', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'P1', 'Pz', 'P2', 'POz']
AFPM_2A_IDX = [CH_2A.index(c) for c in AFPM_TEMPLATE]          # 1..17
AFPM_2B_SLOTS = [AFPM_TEMPLATE.index(c) for c in ('C3', 'Cz', 'C4')]   # 6, 8, 10


def to_template(X, slots):
    """Scatter a dataset's aligned channels into the shared template, zeros elsewhere.

    This is AFPM's channel mapping (its Eq. 4). It runs *after* alignment, not before:
    a zero-filled matrix has a singular covariance, so EA has to see the real channels.
    """
    out = np.zeros((X.shape[0], len(AFPM_TEMPLATE), X.shape[2]), dtype=np.float32)
    out[:, slots, :] = X
    return out
CUE = {'769': 0, '770': 1}     # left hand, right hand -- same coding in both datasets
FS = 250


def regroup_ea(X, subject):
    """Re-apply EA with one R_bar per subject instead of per recording file.
    The per-file alignment already applied by the parser is idempotent in kind --
    whitening whitened data with a pooled R_bar simply replaces the transform."""
    out = np.empty_like(X)
    for s in np.unique(subject):
        m = subject == s
        out[m] = alignOperation(X[m], operation='svd')
    return out


def load_2a():
    """Official parser, restricted to the three channels 2b has. EA is applied inside
    parseBci42aFile, so it is computed on those three channels, matching 2b."""
    X, y, subj, sess = [], [], [], []
    for s in range(9):
        for which, tag in ((0, 'T'), (1, 'E')):
            d = parseBci42aFile(os.path.join(GDF_2A, f'A0{s+1}{tag}.gdf'),
                                os.path.join(MAT_2A, f'A0{s+1}{tag}.mat'),
                                chans=(AFPM_2A_IDX if ARGS.channels == 'afpm'
                                       else C3_CZ_C4))
            x = d['x'].transpose(2, 0, 1)            # (trials, chan, time)
            lab = np.asarray(d['y']).squeeze()
            keep = lab < 2                            # left / right only
            X.append(x[keep]); y.append(lab[keep])
            subj.append(np.full(keep.sum(), s)); sess.append(np.full(keep.sum(), which))
            print(f"  A0{s+1}{tag}: {keep.sum()} left/right trials", flush=True)
    return (np.concatenate(X).astype(np.float32), np.concatenate(y).astype(np.int64),
            np.concatenate(subj), np.concatenate(sess))


def load_2b():
    """The 27 training-session files, epoched on the cue so the label comes from the
    recording itself. EA applied per session, as the 2a path does."""
    X, y, subj, sess = [], [], [], []
    for s in range(9):
        for run in (1, 2, 3):
            f = os.path.join(GDF_2B, f'B0{s+1}0{run}T.gdf')
            raw = mne.io.read_raw_gdf(f, preload=True, verbose='ERROR')
            ev, eid = mne.events_from_annotations(raw, verbose='ERROR')
            inv = {v: k for k, v in eid.items()}
            eeg = raw.get_data()[:3]                  # the 3 EEG channels; rest are EOG
            interval = np.arange(0, 4 * FS)           # 4 s from the cue
            trials = [(e[0], CUE[inv[e[2]]]) for e in ev if inv[e[2]] in CUE]
            x = np.stack([eeg[:, interval + t0] for t0, _ in trials]) * 1e6
            x = alignOperation(x, operation='svd')    # <- the step the official 2b path omits
            if ARGS.channels == 'afpm':
                x = to_template(x, AFPM_2B_SLOTS)
            lab = np.array([c for _, c in trials])
            X.append(x.astype(np.float32)); y.append(lab)
            subj.append(np.full(len(lab), s)); sess.append(np.full(len(lab), run - 1))
            print(f"  B0{s+1}0{run}T: {len(lab)} trials", flush=True)
    return (np.concatenate(X).astype(np.float32), np.concatenate(y).astype(np.int64),
            np.concatenate(subj), np.concatenate(sess))


def filter_bank(X):
    """The official 9-band Chebyshev-II bank from train.py, applied trial by trial."""
    cfg = get_transform(filtBank=[[4, 8], [8, 12], [12, 16], [16, 20], [20, 24],
                                  [24, 28], [28, 32], [32, 36], [36, 40]],
                        fs=FS, filterType='cheby2', order=3, filtType='filter',
                        outputType='sos')
    # get_transform returns the config dict; saveData.py instantiates it like this
    key = list(cfg.keys())[0]
    tf = transforms.__dict__[key](**cfg[key])
    if ARGS.filter == 'reimpl':
        from scipy import signal
        sos = []
        for lo, hi in cfg[key]['filtBank']:
            nF = FS / 2
            n, _ = signal.cheb2ord([lo/nF, hi/nF], [(lo-2)/nF, (hi+2)/nF], 3, 30)
            sos.append(signal.cheby2(n, 30, [(lo-2)/nF, (hi+2)/nF], 'bandpass', output='sos'))

        def tf(pair, _sos=sos):
            x = pair[0]
            return [np.stack([signal.sosfilt(s_, x, axis=-1) for s_ in _sos], axis=-1)]
    out = np.empty((*X.shape, 9), dtype=np.float32)
    for i in range(len(X)):
        out[i] = np.asarray(tf([X[i], 0])[0], dtype=np.float32)
        if i % 1000 == 0:
            print(f"    {i}/{len(X)}", flush=True)
    return out


print(f"config: channels={ARGS.channels}  ea-scope={ARGS.ea_scope}  "
      f"source-2a={ARGS.source_2a}  filter={ARGS.filter}", flush=True)
print("2a (source) ...", flush=True)
if ARGS.source_2a == 'senior':
    d = np.load('/Users/user/MotorImagery/raw 1.npz', allow_pickle=True)
    keep = d['y'] < 2
    Xa = d['X'][keep][:, (AFPM_2A_IDX if ARGS.channels == 'afpm' else C3_CZ_C4),
                      :1000].astype(np.float32)
    ya, sa, ea = (d['y'][keep].astype(np.int64), d['subject'][keep].astype(np.int64),
                  d['session'][keep].astype(np.int64))
    # the parser would have aligned each file; do the equivalent here
    for s_ in np.unique(sa):
        for e_ in np.unique(ea):
            m = (sa == s_) & (ea == e_)
            Xa[m] = alignOperation(Xa[m], operation='svd')
    print(f"  from raw 1.npz: {Xa.shape}", flush=True)
else:
    Xa, ya, sa, ea = load_2a()
print("2b (target) ...", flush=True)
Xb, yb, sb, eb = load_2b()

if ARGS.ea_scope == 'subject':
    print("re-aligning with one R_bar per subject ...", flush=True)
    Xa, Xb = regroup_ea(Xa, sa), regroup_ea(Xb, sb)

print(f"\nraw shapes: 2a {Xa.shape}  2b {Xb.shape}")
for tag, X in (('2a', Xa), ('2b', Xb)):
    live = np.where(np.abs(X[:400]).sum(axis=(0, 2)) > 0)[0]   # skip zero-filled slots
    Xl = X[:400][:, live, :]
    C = len(live)
    R = np.mean([x @ x.T for x in Xl], axis=0); R = R / np.trace(R) * C
    I = np.eye(C)
    print(f"  {tag} EA check  ||R-I||/||I|| = {np.linalg.norm(R-I)/np.linalg.norm(I):.4f}"
          f"   ({C} live of {X.shape[1]} channels)")

print("\nfilter bank 2a ...", flush=True); Xa = filter_bank(Xa)
print("filter bank 2b ...", flush=True); Xb = filter_bank(Xb)

np.savez(OUT, Xs=Xa, ys=ya, ss=sa, es=ea, Xt=Xb, yt=yb, st=sb, et=eb)
print(f"\nsource {Xa.shape}  classes {np.bincount(ya)}  subjects {np.bincount(sa)}")
print(f"target {Xb.shape}  classes {np.bincount(yb)}  subjects {np.bincount(sb)}")
print(f"saved -> {OUT}  ({os.path.getsize(OUT)/1e9:.2f} GB)")
