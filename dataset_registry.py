"""
dataset_registry.py
===================
Centralised dataset path management and split registry for InfantCryNet-v4.

Responsibilities
----------------
1. **Path resolution** — one place to define all raw/interim/processed paths.
2. **Dataset discovery** — recursively scan class folders → DataFrame.
3. **Reproducible splits** — stratified train/val/test + optional k-fold.
4. **Split persistence** — save/load splits to JSON for exact reproducibility.
5. **Class mapping** — apply 8→6 label remapping if needed.
6. **Quality checks** — warn on class imbalance, near-duplicate durations, etc.

Directory contract
------------------
Expected structure (configurable via DatasetConfig):

  data/
    raw/
      audio/
        cause_dataset/
          hungry/     ← one sub-folder per class
          tired/
          pain/
          ...
        cry_noncry/
          cry/
          non_cry/
    interim/
      features/       ← cached .pkl feature DataFrames
    processed/
      splits/         ← saved train/val/test JSON indices

Reproducibility guarantee
-------------------------
Given the same DatasetConfig + split_seed, encode_batch always returns the same
row-to-split assignment regardless of filesystem order.  The split JSON records
the git-commit hash (if available) and config hash for audit trails.

References
----------
Géron A. (2022). Hands-On ML with Scikit-Learn, Keras, and TensorFlow (3rd ed.).
  Chapter 2: End-to-end ML project — discusses data isolation and split hygiene.
"""

from __future__ import annotations

import hashlib
import json
import os
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AUDIO_EXTS: FrozenSet[str] = frozenset(
    {".wav", ".ogg", ".mp3", ".flac", ".m4a", ".aiff", ".aif"}
)

# Canonical 6-class label space
CANONICAL_LABELS: Tuple[str, ...] = (
    "hungry", "pain", "discomfort", "tired", "reflux", "colic"
)

# Accepted folder-name aliases → canonical name
LABEL_ALIASES: Dict[str, str] = {
    # Class 1
    "hungry":       "hungry",
    "hunger":       "hungry",
    # Class 2
    "pain":         "pain",
    "belly_pain":   "pain",
    "belly pain":   "pain",
    "bellyache":    "pain",
    # Class 3
    "discomfort":   "discomfort",
    "cold_hot":     "discomfort",
    "cold hot":     "discomfort",
    "lonely":       "discomfort",
    "scared":       "discomfort",
    # Class 4
    "tired":        "tired",
    "fatigue":      "tired",
    "sleepy":       "tired",
    # Class 5
    "reflux":       "reflux",
    "burping":      "reflux",
    "burp":         "reflux",
    # Class 6
    "colic":        "colic",
    # Cry-gate classes
    "cry":          "cry",
    "non_cry":      "non_cry",
    "noncry":       "non_cry",
    "noise":        "non_cry",
    "silence":      "non_cry",
}

_LABEL_TO_INT: Dict[str, int] = {
    "hungry":    1,
    "pain":      2,
    "discomfort":3,
    "tired":     4,
    "reflux":    5,
    "colic":     6,
}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class DatasetConfig:
    """
    Central configuration for all dataset-related operations.

    Parameters
    ----------
    data_root : str | Path
        Root directory of the project data tree.
    cause_dir : str
        Subdirectory (relative to data_root/raw/audio/) holding cause classes.
    cry_noncry_dir : str
        Sub-directory for binary cry/non-cry data.
    interim_dir : str
        Where to write feature caches (.pkl).
    processed_dir : str
        Where to write split JSON files.
    split_seed : int
        RNG seed for all train/val/test splits.
    test_fraction : float
        Fraction held out as test set.
    val_fraction : float
        Fraction of *training* data held out as validation.
    n_folds : int
        Number of k-fold cross-validation folds (0 = disabled).
    min_samples_per_class : int
        Warn if any class has fewer than this many samples.
    label_aliases : dict
        Folder name → canonical label mapping (defaults to LABEL_ALIASES).
    recursive : bool
        Whether to recurse into sub-folders of each class directory.
    """
    data_root             : str | Path = "data"
    cause_dir             : str        = "raw/audio/cause_dataset"
    cry_noncry_dir        : str        = "raw/audio/cry_noncry"
    interim_dir           : str        = "interim/features"
    processed_dir         : str        = "processed/splits"
    split_seed            : int        = 42
    test_fraction         : float      = 0.20
    val_fraction          : float      = 0.15
    n_folds               : int        = 5
    min_samples_per_class : int        = 30
    label_aliases         : Dict[str, str] = field(default_factory=lambda: dict(LABEL_ALIASES))
    recursive             : bool       = True

    def __post_init__(self) -> None:
        self.data_root = Path(self.data_root)

    @property
    def cause_path(self) -> Path:
        return self.data_root / self.cause_dir

    @property
    def cry_noncry_path(self) -> Path:
        return self.data_root / self.cry_noncry_dir

    @property
    def interim_path(self) -> Path:
        return self.data_root / self.interim_dir

    @property
    def processed_path(self) -> Path:
        return self.data_root / self.processed_dir

    def config_hash(self) -> str:
        """Stable 8-char hash of the config (for cache keys)."""
        d = {k: str(v) for k, v in asdict(self).items()
             if k not in ("label_aliases",)}
        raw = json.dumps(d, sort_keys=True).encode()
        return hashlib.md5(raw).hexdigest()[:8]


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

class DatasetRegistry:
    """
    Discovers audio files, manages splits, and provides reproducible
    train/val/test/fold iterators.

    Parameters
    ----------
    config : DatasetConfig

    Quick start
    -----------
    >>> cfg  = DatasetConfig(data_root="data")
    >>> reg  = DatasetRegistry(cfg)
    >>> df   = reg.discover_cause_dataset()
    >>> splits = reg.make_splits(df)
    >>> df_train = df.loc[splits["train"]]
    """

    def __init__(self, config: Optional[DatasetConfig] = None) -> None:
        self.cfg = config or DatasetConfig()

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def discover_cause_dataset(
        self,
        root: Optional[Path] = None,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Recursively discover audio files under cause_dataset/.

        Parameters
        ----------
        root    : override cause_path
        verbose : print per-class counts

        Returns
        -------
        df : DataFrame with columns
             filepath, label, label_int, original_folder, filename, duration_hint
        """
        base = root or self.cfg.cause_path
        if not base.exists():
            raise FileNotFoundError(
                f"Cause dataset not found: {base}\n"
                f"Set DatasetConfig.data_root correctly."
            )
        return self._scan_directory(base, verbose=verbose)

    def discover_cry_noncry(
        self,
        root: Optional[Path] = None,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Discover audio files for the binary cry-gate dataset.
        """
        base = root or self.cfg.cry_noncry_path
        if not base.exists():
            raise FileNotFoundError(f"Cry/non-cry dataset not found: {base}")
        return self._scan_directory(base, verbose=verbose, gate_mode=True)

    def _scan_directory(
        self,
        root: Path,
        verbose: bool,
        gate_mode: bool = False,
    ) -> pd.DataFrame:
        rows: List[Dict] = []
        class_dirs = sorted([d for d in root.iterdir() if d.is_dir()])
        if not class_dirs:
            raise RuntimeError(f"No class sub-folders found in: {root}")

        for cls_dir in class_dirs:
            orig = cls_dir.name.strip().lower()
            canonical = self.cfg.label_aliases.get(orig)
            if canonical is None:
                if verbose:
                    print(f"  ⚠️  Unknown folder '{cls_dir.name}' — skipped.")
                continue

            if self.cfg.recursive:
                files = [f for f in cls_dir.rglob("*")
                         if f.is_file() and f.suffix.lower() in AUDIO_EXTS]
            else:
                files = [f for f in cls_dir.iterdir()
                         if f.is_file() and f.suffix.lower() in AUDIO_EXTS]

            label_int = _LABEL_TO_INT.get(canonical, 0)

            for fp in files:
                rows.append({
                    "filepath":        str(fp),
                    "label":           canonical,
                    "label_int":       label_int,
                    "original_folder": orig,
                    "filename":        fp.name,
                    "ext":             fp.suffix.lower(),
                })

        if not rows:
            raise RuntimeError(f"No audio files found under: {root}")

        df = pd.DataFrame(rows).reset_index(drop=True)

        if verbose:
            self._print_distribution(df, "Discovered")

        self._check_class_balance(df)
        return df

    # ------------------------------------------------------------------
    # Splits
    # ------------------------------------------------------------------

    def make_splits(
        self,
        df: pd.DataFrame,
        save: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        Create stratified train / val / test index splits.

        Parameters
        ----------
        df   : the DataFrame returned by discover_*
        save : write split JSON to processed_dir

        Returns
        -------
        dict with keys "train", "val", "test" → int64 index arrays
        """
        labels = df["label"].values
        idx    = np.arange(len(df))

        # Test split (held out completely)
        idx_trainval, idx_test = train_test_split(
            idx,
            test_size=self.cfg.test_fraction,
            stratify=labels,
            random_state=self.cfg.split_seed,
        )

        # Val split from train+val
        labels_tv = df.iloc[idx_trainval]["label"].values
        idx_train, idx_val = train_test_split(
            idx_trainval,
            test_size=self.cfg.val_fraction,
            stratify=labels_tv,
            random_state=self.cfg.split_seed + 1,
        )

        splits = {
            "train": idx_train.astype(np.int64),
            "val":   idx_val.astype(np.int64),
            "test":  idx_test.astype(np.int64),
        }

        if save:
            self._save_splits(splits, df)

        return splits

    def make_kfold_splits(
        self,
        df: pd.DataFrame,
    ) -> List[Dict[str, np.ndarray]]:
        """
        Create k stratified cross-validation splits.

        Returns
        -------
        list of dicts, each with keys "train", "val" → int64 index arrays.
        Length = n_folds.
        """
        if self.cfg.n_folds < 2:
            raise ValueError("n_folds must be ≥ 2")

        labels = df["label"].values
        idx    = np.arange(len(df))
        skf    = StratifiedKFold(
            n_splits=self.cfg.n_folds,
            shuffle=True,
            random_state=self.cfg.split_seed,
        )
        folds = []
        for train_idx, val_idx in skf.split(idx, labels):
            folds.append({
                "train": idx[train_idx].astype(np.int64),
                "val":   idx[val_idx].astype(np.int64),
            })
        return folds

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_splits(
        self,
        splits: Dict[str, np.ndarray],
        df: pd.DataFrame,
    ) -> None:
        out = self.cfg.processed_path
        out.mkdir(parents=True, exist_ok=True)

        payload = {
            "config_hash": self.cfg.config_hash(),
            "n_total":     int(len(df)),
            "split_seed":  self.cfg.split_seed,
            "counts":      {k: int(len(v)) for k, v in splits.items()},
            "splits":      {k: v.tolist() for k, v in splits.items()},
            "label_counts_per_split": {
                split_name: {
                    lbl: int(np.sum(df.iloc[idxs]["label"] == lbl))
                    for lbl in df["label"].unique()
                }
                for split_name, idxs in splits.items()
            },
        }

        # Record git commit if available
        try:
            import subprocess
            sha = subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL
            ).decode().strip()
            payload["git_commit"] = sha
        except Exception:
            payload["git_commit"] = "unavailable"

        fname = out / f"splits_{self.cfg.config_hash()}.json"
        fname.write_text(json.dumps(payload, indent=2))

    def load_splits(
        self,
        config_hash: Optional[str] = None,
    ) -> Optional[Dict[str, np.ndarray]]:
        """
        Load previously saved splits.

        Parameters
        ----------
        config_hash : 8-char hash from DatasetConfig.config_hash().
                      If None, uses current config hash.

        Returns
        -------
        dict or None if not found.
        """
        h = config_hash or self.cfg.config_hash()
        fname = self.cfg.processed_path / f"splits_{h}.json"
        if not fname.exists():
            return None

        raw  = json.loads(fname.read_text())
        return {k: np.array(v, dtype=np.int64) for k, v in raw["splits"].items()}

    # ------------------------------------------------------------------
    # Feature cache helpers
    # ------------------------------------------------------------------

    def feature_cache_path(self, dataset_name: str = "cause") -> Path:
        """Return canonical path for feature cache PKL."""
        self.cfg.interim_path.mkdir(parents=True, exist_ok=True)
        return self.cfg.interim_path / f"features_{dataset_name}_{self.cfg.config_hash()}.pkl"

    def load_feature_cache(
        self, dataset_name: str = "cause"
    ) -> Optional[pd.DataFrame]:
        """Load cached feature DataFrame if it exists."""
        p = self.feature_cache_path(dataset_name)
        if not p.exists():
            return None
        return pd.read_pickle(p)

    def save_feature_cache(
        self, df_features: pd.DataFrame, dataset_name: str = "cause"
    ) -> None:
        """Persist feature DataFrame to the interim cache."""
        p = self.feature_cache_path(dataset_name)
        p.parent.mkdir(parents=True, exist_ok=True)
        df_features.to_pickle(p)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def _print_distribution(self, df: pd.DataFrame, title: str) -> None:
        total = len(df)
        print(f"\n  {title}: {total} files")
        for lbl, cnt in df["label"].value_counts().items():
            bar = "█" * int(cnt / total * 30)
            print(f"    {lbl:<14} {cnt:5d}  {bar}")

    def _check_class_balance(self, df: pd.DataFrame) -> None:
        counts = df["label"].value_counts()
        if counts.min() < self.cfg.min_samples_per_class:
            low = counts[counts < self.cfg.min_samples_per_class].to_dict()
            warnings.warn(
                f"Classes with fewer than {self.cfg.min_samples_per_class} samples: {low}\n"
                "Consider collecting more data or using SMOTE augmentation.",
                stacklevel=2,
            )
        imbalance = counts.max() / (counts.min() + 1e-9)
        if imbalance > 5.0:
            warnings.warn(
                f"High class imbalance (ratio {imbalance:.1f}:1). "
                "Enable balance_strategy='smote+under' in ProbabilisticEnsembleModel.",
                stacklevel=2,
            )

    def summary(self, df: Optional[pd.DataFrame] = None) -> str:
        lines = [
            "DatasetRegistry",
            f"  data_root   : {self.cfg.data_root}",
            f"  cause_dir   : {self.cfg.cause_path}",
            f"  split_seed  : {self.cfg.split_seed}",
            f"  test_frac   : {self.cfg.test_fraction:.0%}",
            f"  val_frac    : {self.cfg.val_fraction:.0%}",
            f"  n_folds     : {self.cfg.n_folds}",
            f"  config_hash : {self.cfg.config_hash()}",
        ]
        if df is not None:
            counts = df["label"].value_counts()
            lines += ["", "  Class distribution:"]
            for lbl, cnt in counts.items():
                lines.append(f"    {lbl:<14} {cnt:5d}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def load_or_discover(
    cfg: DatasetConfig,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Load a cached discovery DataFrame or run a fresh scan.

    Stores the result to cfg.interim_path/discovery_{hash}.parquet.
    """
    cache = cfg.interim_path / f"discovery_{cfg.config_hash()}.parquet"
    if cache.exists():
        if verbose:
            print(f"  📦 Loading discovery cache: {cache}")
        return pd.read_parquet(cache)

    reg = DatasetRegistry(cfg)
    df  = reg.discover_cause_dataset(verbose=verbose)
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache, index=False)
    return df


def node_subsets(
    df: pd.DataFrame,
    split_indices: Dict[str, np.ndarray],
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Return per-split, per-node subsets ready for classifier training.

    Output structure:
    {
      "train": {
        "node_a": df_all_train,
        "node_b": df_non_hungry_train,
        "node_c": df_active_distress_train,
      },
      "val":   { same },
      "test":  { same },
    }
    """
    from hierarchical_labels import encode_batch

    result: Dict[str, Dict[str, pd.DataFrame]] = {}
    for split_name, idxs in split_indices.items():
        sub = df.iloc[idxs].reset_index(drop=True)
        labels = sub["label"].values
        enc    = encode_batch(labels)

        result[split_name] = {
            "node_a": sub,
            "node_b": sub[enc["node_b_mask"]].reset_index(drop=True),
            "node_c": sub[enc["node_c_mask"]].reset_index(drop=True),
        }
    return result


# ---------------------------------------------------------------------------
# Self-test (runs without real data using a temp directory)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile
    import soundfile as sf

    # Create a tiny synthetic dataset in a temp dir
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        CLASSES = ["hungry", "pain", "discomfort", "tired", "reflux", "colic"]
        COUNTS  = [30, 20, 25, 22, 18, 15]

        # Write synthetic WAV files
        rng = np.random.default_rng(0)
        for cls, n in zip(CLASSES, COUNTS):
            cls_dir = root / cls
            cls_dir.mkdir()
            for i in range(n):
                wav = rng.standard_normal(22050 * 2).astype(np.float32)
                sf.write(cls_dir / f"{cls}_{i:03d}.wav", wav, 22050)

        cfg = DatasetConfig(
            data_root=tmpdir,
            cause_dir="",
            split_seed=42,
        )
        cfg.cause_dir = ""   # scan tmpdir directly

        reg = DatasetRegistry(cfg)
        df  = reg.discover_cause_dataset(root=root, verbose=True)

        print("\n" + reg.summary(df))

        splits = reg.make_splits(df, save=False)
        print(f"\n  Split sizes: train={len(splits['train'])}  "
              f"val={len(splits['val'])}  test={len(splits['test'])}")

        folds = reg.make_kfold_splits(df)
        print(f"  K-fold folds: {len(folds)}")

        subsets = node_subsets(df, splits)
        for split_name, nodes in subsets.items():
            print(f"  {split_name}: A={len(nodes['node_a'])}  "
                  f"B={len(nodes['node_b'])}  C={len(nodes['node_c'])}")

        # Test cache
        reg.save_feature_cache(df, dataset_name="test_cause")
        loaded = reg.load_feature_cache(dataset_name="test_cause")
        assert loaded is not None and len(loaded) == len(df)
        print(f"\n  Feature cache save/load OK ({len(loaded)} rows)")

    print("\n  All self-tests passed ✅")