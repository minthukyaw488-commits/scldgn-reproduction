"""Rebuild the per-session label .mat files the official SCLDGN loader expects.

The standard BCI Competition IV 2a download ships only the 18 .gdf recordings.
`dataset/saveData.py` additionally reads `A0xT.mat` / `A0xE.mat` with a `classlabel`
field. This script writes them, and verifies the training-session labels against the
cue codes (769-772) embedded in the GDF annotations.

Usage:
    python make_label_mats.py <gdf_dir> <out_dir> <labels.npz>

`labels.npz` must contain y, subject, session, trial_idx for all 5184 trials.
"""
import os
import sys
import warnings

import numpy as np
from scipy.io import savemat

warnings.filterwarnings("ignore")

CUE = {"769": 0, "770": 1, "771": 2, "772": 3}


def verify_against_gdf(gdf_dir, y, subject, session, trial_idx):
    """Training sessions carry their labels in the GDF itself -- check we agree."""
    import mne
    for s in range(9):
        raw = mne.io.read_raw_gdf(os.path.join(gdf_dir, f"A0{s+1}T.gdf"),
                                  preload=False, verbose="ERROR")
        events, eid = mne.events_from_annotations(raw, verbose="ERROR")
        inv = {v: k for k, v in eid.items()}
        gdf_y = np.array([CUE[inv[c]] for c in events[:, 2] if inv[c] in CUE])

        m = (subject == s) & (session == 0)          # session 0 == the T session
        ours = y[m][np.argsort(trial_idx[m])]
        if len(ours) != len(gdf_y) or not np.array_equal(ours, gdf_y):
            raise SystemExit(f"A0{s+1}T: labels disagree with the GDF cue codes")
        print(f"  A0{s+1}T  {len(gdf_y)} trials  match")


def main():
    gdf_dir, out_dir, npz = sys.argv[1], sys.argv[2], sys.argv[3]
    os.makedirs(out_dir, exist_ok=True)

    d = np.load(npz, allow_pickle=True)
    y, subject = d["y"], d["subject"]
    session, trial_idx = d["session"], d["trial_idx"]

    print("verifying training-session labels against the GDF annotations:")
    verify_against_gdf(gdf_dir, y, subject, session, trial_idx)

    print("\nwriting .mat files:")
    for s in range(9):
        for which, tag in ((0, "T"), (1, "E")):
            m = (subject == s) & (session == which)
            lab = y[m][np.argsort(trial_idx[m])] + 1     # loader does classlabel - 1
            counts = [int((lab == k).sum()) for k in (1, 2, 3, 4)]
            if counts != [72, 72, 72, 72]:
                raise SystemExit(f"A0{s+1}{tag}: expected 72 per class, got {counts}")
            # int dtype matters: a float here becomes "3.0" in dataLabels.csv,
            # which the loader's int() call rejects.
            savemat(os.path.join(out_dir, f"A0{s+1}{tag}.mat"),
                    {"classlabel": lab.reshape(-1, 1).astype(np.int16)})
            print(f"  A0{s+1}{tag}  72/72/72/72")

    print(f"\nwrote 18 files to {out_dir}")


if __name__ == "__main__":
    main()
