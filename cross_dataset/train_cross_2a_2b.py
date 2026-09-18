"""Cross-dataset SCLDGN: train on BCI IV 2a, test on BCI IV 2b.

  source  2a, all 9 subjects treated as 9 training domains, left/right only
  target  2b, 9 subjects, never seen during training or model selection

Network and loss come from the official SCLDGN modules unchanged -- B7 wrapped in
ERM_SMA, and the `smcldgn` objective (CE + tradeOff*CORAL + tradeOff2*SCL with
domain-agnostic mixup) transcribed from baseModel.py.

What differs from the within-dataset run, and why:

  * ndomain is 9, not 8. Every 2a subject is a source here, because the held-out
    subject lives in the other dataset. batchSize follows at 36 = 9 x 4, keeping the
    same four trials per domain per batch that 32 = 8 x 4 gives in the official run.

  * Model selection uses a held-out split of the *source* subjects, as the official
    code does. The target dataset is read exactly once, after training ends.
"""
import argparse
import copy
import csv
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler

HERE = os.path.dirname(os.path.abspath(__file__))
MASTER = os.path.join(HERE, 'SCLDGN')
sys.path.insert(1, MASTER)

from network import networks
from network.SMA import ERM_SMA
from lossFunction.coral import CorrelationAlignmentLoss
from lossFunction.scl import SupConLoss

DATA = os.environ.get('DATA_OVERRIDE',
                      os.path.join(os.path.dirname(HERE), 'cross_2a_2b_official.npz'))
RAND_SEED = 19960822
SPLIT_RATIO = 0.8


class ArrayDataset(Dataset):
    def __init__(self, X, y, idx):
        self.X, self.y, self.idx = X, y, np.asarray(idx)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = self.idx[i]
        return torch.from_numpy(np.ascontiguousarray(self.X[j].transpose(2, 0, 1))), int(self.y[j])


class RandomDomainSampler(Sampler):
    """Verbatim from SCLDGN/utils/tools.py -- lays each batch out as contiguous
    per-domain blocks so `.chunk(ndomain)` splits by domain."""

    def __init__(self, cumulative_sizes, batch_size, n_domains_per_batch):
        super(Sampler, self).__init__()
        self.n_domains_in_dataset = len(cumulative_sizes)
        self.n_domains_per_batch = n_domains_per_batch
        self.sample_idxes_per_domain = []
        start = 0
        for end in cumulative_sizes:
            self.sample_idxes_per_domain.append(list(range(start, end)))
            start = end
        assert batch_size % n_domains_per_batch == 0
        self.batch_size_per_domain = batch_size // n_domains_per_batch
        self.length = len(list(self.__iter__()))

    def __iter__(self):
        pool = copy.deepcopy(self.sample_idxes_per_domain)
        final, stop = [], False
        while not stop:
            for d in range(self.n_domains_in_dataset):
                idxes = pool[d]
                if len(idxes) < self.batch_size_per_domain:
                    sel = np.random.choice(idxes, self.batch_size_per_domain, replace=True)
                else:
                    sel = random.sample(idxes, self.batch_size_per_domain)
                final.extend(sel)
                for i in sel:
                    if i in pool[d]:
                        pool[d].remove(i)
                if len(pool[d]) < self.batch_size_per_domain:
                    stop = True
        return iter(final)

    def __len__(self):
        return self.length


def split_idx(idx, n, seed):
    """SCLDGN/utils/tools.py split_idx, datasetId=0 branch."""
    idx = list(idx)
    idx1, idx2 = idx[:len(idx) // 2], idx[len(idx) // 2:]
    np.random.RandomState(seed).shuffle(idx1)
    np.random.RandomState(seed).shuffle(idx2)
    n1 = int(len(idx1) * n)
    n2 = int(len(idx2) * n)
    return idx1[:n1] + idx2[:n1], idx1[n2:] + idx2[n2:]


def build_source(y, subject, session, n_classes, seed):
    """Per source subject, per class: the official per-session 80/20 split.
    Returns domain-ordered train indices plus the domain boundaries."""
    train, val, cumulative = [], [], []
    for sub in sorted(np.unique(subject)):
        for cls in range(n_classes):
            m = np.where((subject == sub) & (y == cls))[0]
            m = m[np.argsort(session[m], kind='stable')]
            tr, va = split_idx(m, SPLIT_RATIO, seed)
            train.extend(tr)
            val.extend(va)
        cumulative.append(len(train))
    return np.array(train), np.array(val), cumulative


def smcldgn_loss(model, x, y, ndomain, coral_fn, scl_fn, tradeOff, tradeOff2):
    """Transcribed from SCLDGN/baseModel/baseModel.py :: smcldgn.

    NOT original work -- this block belongs to the SCLDGN authors. It is reproduced
    here because `ho()` only runs leave-one-subject-out inside a single dataset, so a
    cross-dataset driver cannot call into it; the objective itself is unchanged.
    """
    batch_size = y.size()[0]
    y_logit, feats, proj = model.update(x)

    y_logit_c, feats_c, y_coral = (y_logit.chunk(ndomain, 0),
                                   feats.chunk(ndomain, 0), y.chunk(ndomain, 0))
    ce, penalty = 0, 0
    for i in range(ndomain):
        ce += F.cross_entropy(y_logit_c[i], y_coral[i])
        for j in range(i + 1, ndomain):
            penalty += coral_fn(feats_c[i], feats_c[j])
    ce /= ndomain
    penalty /= ndomain * (ndomain - 1) / 2

    lam = np.random.uniform(0.9, 1.0)
    sorted_y, indices = torch.sort(y)
    proj = proj[indices]
    intervals, ex = [], 0
    for idx, val in enumerate(sorted_y):
        if ex == val:
            continue
        intervals.append(idx)
        ex = val
    intervals.append(batch_size)

    mix1, mix2 = torch.zeros_like(proj), torch.zeros_like(proj)
    ex = 0
    for end in intervals:
        s1 = torch.randperm(end - ex) + ex
        s2 = torch.randperm(end - ex) + ex
        for k in range(end - ex):
            mix1[k + ex] = proj[s1[k]]
            mix2[k + ex] = proj[s2[k]]
        ex = end
    p1, p2 = lam * proj + (1 - lam) * mix1, lam * proj + (1 - lam) * mix2
    p = torch.cat([p1.unsqueeze(1), p2.unsqueeze(1)], dim=1)

    return ce + tradeOff * penalty + tradeOff2 * scl_fn(p, sorted_y, mask=None)


@torch.no_grad()
def evaluate(model, loader, device):
    model.network_sma.eval()
    correct = total = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits, _, _ = model.predict(xb)
        correct += (logits.argmax(1) == yb).sum().item()
        total += yb.numel()
    return correct / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--network', default='B7')
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--batch-size', type=int, default=36)     # 9 domains x 4
    ap.add_argument('--feature', type=int, default=32)
    ap.add_argument('--tradeoff', type=float, default=1.0)    # CORAL
    ap.add_argument('--tradeoff2', type=float, default=0.1)   # SCL
    ap.add_argument('--sma', type=int, default=100)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--seed', type=int, default=RAND_SEED)
    ap.add_argument('--device', default='mps')
    ap.add_argument('--out', default=os.path.join(HERE, 'cross_2a_2b_results.csv'))
    args = ap.parse_args()

    device = torch.device(args.device if (args.device != 'mps' or torch.backends.mps.is_available()) else 'cpu')
    d = np.load(DATA)
    Xs, ys, ss = d['Xs'], d['ys'].astype(np.int64), d['ss'].astype(np.int64)
    Xt, yt, st = d['Xt'], d['yt'].astype(np.int64), d['st'].astype(np.int64)
    n_classes = int(ys.max()) + 1

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    tr, va, cumulative = build_source(ys, ss, d['es'].astype(np.int64), n_classes, args.seed)
    ndomain = len(cumulative)

    sampler = RandomDomainSampler(cumulative, args.batch_size, ndomain)
    train_loader = DataLoader(ArrayDataset(Xs, ys, tr), batch_size=args.batch_size,
                              sampler=sampler, drop_last=True)
    val_loader = DataLoader(ArrayDataset(Xs, ys, va), batch_size=args.batch_size)

    print(f"device {device} | source {Xs.shape} ({ndomain} domains) -> target {Xt.shape}")
    print(f"network {args.network} | train {len(tr)} val {len(va)} | batch {args.batch_size}", flush=True)

    net = networks.__dict__[args.network](
        inputSize=(Xs.shape[3], Xs.shape[1], Xs.shape[2]), nClass=n_classes,
        m=args.feature, dropoutP=0.0, c=0.5, isProj=True).to(device)
    model = ERM_SMA(net, start=args.sma); model.network_sma.to(device)
    coral_fn = CorrelationAlignmentLoss().to(device)
    scl_fn = SupConLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    best_inacc, best_state, best_epoch = float('inf'), None, 0
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        net.train()
        running = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = smcldgn_loss(model, xb, yb, ndomain, coral_fn, scl_fn,
                                args.tradeoff, args.tradeoff2)
            loss.backward(); optimizer.step()
            running += loss.item()
        val_acc = evaluate(model, val_loader, device)
        if (1 - val_acc) < best_inacc:
            best_inacc, best_epoch = 1 - val_acc, epoch
            best_state = copy.deepcopy(model.network_sma.state_dict())
        print(f"  epoch {epoch:3d}/{args.epochs}  loss {running/len(train_loader):.4f}  "
              f"src-val {val_acc*100:.2f}%  (best {(1-best_inacc)*100:.2f} @ {best_epoch})  "
              f"{time.time()-t0:.0f}s", flush=True)

    model.network_sma.load_state_dict(best_state)
    per_subject = {}
    for s in sorted(np.unique(st)):
        m = st == s
        loader = DataLoader(ArrayDataset(Xt, yt, np.where(m)[0]), batch_size=args.batch_size)
        per_subject[int(s)] = evaluate(model, loader, device) * 100
    mean = float(np.mean(list(per_subject.values())))

    with open(args.out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['network', 'source_val', 'best_epoch', 'target_mean'] +
                   [f'B{i+1:02d}' for i in sorted(per_subject)])
        w.writerow([args.network, f"{(1-best_inacc)*100:.2f}", best_epoch, f"{mean:.2f}"] +
                   [f"{per_subject[i]:.2f}" for i in sorted(per_subject)])

    print(f"\n--> {args.network}: source-val {(1-best_inacc)*100:.2f}%  |  TARGET 2b {mean:.2f}%  "
          f"({(time.time()-t0)/60:.1f} min)")
    print("    " + "  ".join(f"B{k+1:02d} {v:.2f}" for k, v in sorted(per_subject.items())))


if __name__ == '__main__':
    main()
