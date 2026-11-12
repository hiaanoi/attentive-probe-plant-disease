import math
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


NORMALIZE_EPS = 1e-6
PROTOTYPES_PER_CLASS = 4
INITIAL_TAU = 0.07


class PrototypeHead(nn.Module):
    """Mean over K cosine similarities per class, divided by learned tau."""

    def __init__(self, centers):
        super().__init__()
        self.prototypes = nn.Parameter(torch.tensor(centers, dtype=torch.float32, device="cpu"))
        self.log_tau = nn.Parameter(torch.tensor(math.log(INITIAL_TAU), dtype=torch.float32, device="cpu"))

    def _normalized_logits(self, unit_features):
        unit_prototypes = F.normalize(self.prototypes, p=2, dim=-1, eps=NORMALIZE_EPS)
        similarities = torch.einsum("bd,ckd->bck", unit_features, unit_prototypes)
        return similarities.mean(dim=-1) / self.log_tau.exp()

    def forward(self, x):
        unit_features = F.normalize(x, p=2, dim=-1, eps=NORMALIZE_EPS)
        return self._normalized_logits(unit_features)


class HybridHead(nn.Module):
    """Equal linear/prototype logit mixture on L2-normalized features."""

    def __init__(self, feature_dim, num_classes, centers, class_means):
        super().__init__()
        self.linear = nn.Linear(feature_dim, num_classes, device="cpu", dtype=torch.float32)
        self.prototype = PrototypeHead(centers)
        with torch.no_grad():
            self.linear.weight.copy_(torch.as_tensor(class_means, dtype=torch.float32))
            self.linear.bias.zero_()

    def forward(self, x):
        unit_features = F.normalize(x, p=2, dim=-1, eps=NORMALIZE_EPS)
        # Linear weights remain unconstrained during training.
        return 0.5 * self.linear(unit_features) + 0.5 * self.prototype._normalized_logits(unit_features)


def _validate_train_arrays(feature_dim, num_classes, train_features_np, train_labels_np):
    if not isinstance(train_features_np, np.ndarray) or not isinstance(train_labels_np, np.ndarray):
        raise TypeError("TRAIN features and labels must be numpy ndarrays")
    if feature_dim < 1 or num_classes < 1:
        raise ValueError("feature_dim and num_classes must be positive")
    if train_features_np.ndim != 2 or train_features_np.shape[1] != feature_dim:
        raise ValueError("TRAIN features must have shape [N, feature_dim]")
    if train_labels_np.ndim != 1 or len(train_labels_np) != len(train_features_np):
        raise ValueError("TRAIN labels must have shape [N] matching TRAIN features")
    if not np.issubdtype(train_features_np.dtype, np.number) or np.iscomplexobj(train_features_np):
        raise TypeError("TRAIN features must be real numeric values")
    if not np.issubdtype(train_labels_np.dtype, np.integer):
        raise TypeError("TRAIN labels must be integer class indices")
    features = np.asarray(train_features_np, dtype=np.float32)
    labels = np.asarray(train_labels_np, dtype=np.int64)
    if not np.isfinite(features).all():
        raise ValueError("TRAIN features must be finite")
    if len(labels) == 0 or np.any(labels < 0) or np.any(labels >= num_classes):
        raise ValueError("TRAIN labels must lie in [0, num_classes)")
    counts = np.bincount(labels, minlength=num_classes)
    missing = np.flatnonzero(counts == 0)
    if len(missing):
        raise ValueError(f"Missing TRAIN classes: {missing.tolist()}")
    return features, labels


def _train_initialization(features, labels, num_classes, seed):
    from sklearn.cluster import KMeans

    # Identical epsilon convention to torch F.normalize; never use DEV rows.
    normalized = features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), NORMALIZE_EPS)
    centers = np.empty((num_classes, PROTOTYPES_PER_CLASS, features.shape[1]), dtype=np.float32)
    means = np.empty((num_classes, features.shape[1]), dtype=np.float32)
    metadata = []
    for class_index in range(num_classes):
        class_features = normalized[labels == class_index]
        unique_count = int(np.unique(class_features, axis=0).shape[0])
        effective_k = min(PROTOTYPES_PER_CLASS, unique_count)
        clustering = KMeans(n_clusters=effective_k, random_state=int(seed), n_init=10, max_iter=300)
        clustering.fit(class_features)
        # Preserve fitted-center order, repeat deterministically without noise.
        indices = np.arange(PROTOTYPES_PER_CLASS) % effective_k
        centers[class_index] = clustering.cluster_centers_[indices]
        means[class_index] = class_features.mean(axis=0)
        metadata.append({
            "class_index": class_index,
            "n": int(len(class_features)),
            "unique": unique_count,
            "init_clusters": effective_k,
            "fallback": effective_k < PROTOTYPES_PER_CLASS,
            "fallback_centers_added": PROTOTYPES_PER_CLASS - effective_k,
        })
    return centers, means, metadata


def make_head(kind, feature_dim, num_classes, train_features_np, train_labels_np, seed):
    """Construct one head on CPU from caller-provided TRAIN arrays only.

    kind: linear (raw features), prototype/hybrid (L2-normalized features), or
    mlp (raw features, D -> 512 -> C). The single-backbone caller fixes D=1024. PyTorch's
    ordinary Linear initialization uses the caller's torch.manual_seed(seed);
    k-means uses its explicit random_state. No shuffle generator is accepted
    or accessed, and no global NumPy random seed is changed.
    """
    started = time.perf_counter()
    if kind not in {"linear", "prototype", "hybrid", "mlp"}:
        raise ValueError(f"Unknown head kind: {kind!r}")
    features, labels = _validate_train_arrays(feature_dim, num_classes, train_features_np, train_labels_np)
    class_metadata = []
    if kind == "linear":
        head = nn.Linear(feature_dim, num_classes, device="cpu", dtype=torch.float32)
    elif kind == "mlp":
        head = nn.Sequential(
            nn.Linear(feature_dim, 512, device="cpu", dtype=torch.float32),
            nn.GELU(),
            nn.Linear(512, num_classes, device="cpu", dtype=torch.float32),
        )
    else:
        centers, means, class_metadata = _train_initialization(features, labels, num_classes, seed)
        if kind == "prototype":
            head = PrototypeHead(centers)
        else:
            head = HybridHead(feature_dim, num_classes, centers, means)
    head.init_metadata = {
        "kind": kind,
        "seed": int(seed),
        "feature_dim": int(feature_dim),
        "num_classes": int(num_classes),
        "initialization_source": "caller_supplied_TRAIN_only",
        "prototypes_per_class": PROTOTYPES_PER_CLASS if kind in {"prototype", "hybrid"} else 0,
        "normalize_eps": NORMALIZE_EPS,
        "kmeans": {"random_state": int(seed), "n_init": 10, "max_iter": 300}
                  if kind in {"prototype", "hybrid"} else None,
        "classes": class_metadata,
        "parameter_count": sum(p.numel() for p in head.parameters()),
        "initialization_seconds": time.perf_counter() - started,
    }
    return head


def focal_loss(logits, labels, gamma=2, smoothing=0.1):
    """Smoothed CE times focal factor using the unsmoothed true-label pt."""
    ce = F.cross_entropy(logits, labels, reduction="none", label_smoothing=smoothing)
    pt = logits.softmax(dim=-1).gather(1, labels[:, None]).squeeze(1)
    return ((1.0 - pt).pow(gamma) * ce).mean()


def learning_rate(step, total_steps=1350, base_lr=0.001):
    """One-based optimizer-step LR: ceil(5%) warmup, then cosine to zero."""
    if int(total_steps) != total_steps or total_steps < 1:
        raise ValueError("total_steps must be a positive integer")
    if int(step) != step or not 1 <= step <= total_steps:
        raise ValueError("step must be an integer in [1, total_steps]")
    warmup_steps = math.ceil(0.05 * total_steps)
    if step <= warmup_steps:
        return base_lr * step / warmup_steps
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))
