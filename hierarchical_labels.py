"""
hierarchical_labels.py
======================
Hierarchical label encoder for InfantCryNet-v4.

Architecture (decision-tree, not flat):
  Stage 0 : Cry Gate       — cry (1) vs. not-cry (0)
  Node A  : Hunger Gate    — hungry (1) vs. non-hungry (0)
  Node B  : Fatigue Gate   — tired (1) vs. active-distress (0)
  Node C  : Distress Type  — pain (0) / reflux (1) / discomfort (2)

Flat label space (6 classes):
  1 = hungry   2 = pain   3 = discomfort
  4 = tired    5 = reflux  6 = colic (mapped → pain by default)

Mathematical Foundations
------------------------
Let Y ∈ {1,…,K} be the flat label. We define indicator projections:

  Node A :  Y_A = 1[Y = 1],              domain = {1,…,K}
  Node B :  Y_B = 1[Y = 4],              domain = {2,3,4,5,6}
  Node C :  Y_C = {2→0, 5→1, 3→2, 6→0}, domain = {2,3,5,6}

Conditional independence structure (factored likelihood):
  P(Y) = P(Y_A) · P(Y_B | Y_A=0) · P(Y_C | Y_A=0, Y_B=0)

References
----------
Fürnkranz J., Hüllermeier E. (2010). Preference Learning. Springer.
Dembczyński K. et al. (2012). Label Powerset. JMLR Workshop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Canonical string names used throughout the project
CLASS_NAMES: Dict[int, str] = {
    1: "hungry",
    2: "pain",
    3: "discomfort",
    4: "tired",
    5: "reflux",
    6: "colic",      # treated as pain by default (see COLIC_MAPPING)
}

NAME_TO_INT: Dict[str, int] = {v: k for k, v in CLASS_NAMES.items()}

# Alternative string aliases accepted on input
STRING_ALIASES: Dict[str, int] = {
    **NAME_TO_INT,
    "hunger":       1,
    "hungry":       1,
    "belly pain":   2,
    "belly_pain":   2,
    "burping":      5,
    "reflux":       5,
    "burp":         5,
    "fatigue":      4,
    "sleepy":       4,
    "cold_hot":     3,   # "discomfort" umbrella
    "cold hot":     3,
    "lonely":       3,
    "scared":       3,
}

# Node C active classes (subset of non-hungry, non-tired)
NODE_C_CLASSES: Tuple[int, ...] = (2, 3, 5, 6)
NODE_C_MAP: Dict[int, int] = {2: 0, 3: 2, 5: 1, 6: 0}   # pain→0, reflux→1, discomfort→2
NODE_C_NAMES: Dict[int, str] = {0: "pain", 1: "reflux", 2: "discomfort"}

# Colic is architecturally mapped to pain (same clinical pathway)
COLIC_MAPPING: int = 2

# ---------------------------------------------------------------------------
# Input normalisation
# ---------------------------------------------------------------------------

def _to_int(label: int | str) -> int:
    """Convert any label representation to canonical int in {1,…,6}."""
    if isinstance(label, (int, np.integer)):
        v = int(label)
        if v not in CLASS_NAMES:
            raise ValueError(f"Unknown int label {v}. Valid: {sorted(CLASS_NAMES)}")
        return v
    if isinstance(label, str):
        key = label.strip().lower()
        if key not in STRING_ALIASES:
            raise ValueError(
                f"Unknown string label '{label}'. "
                f"Valid aliases: {sorted(STRING_ALIASES)}"
            )
        return STRING_ALIASES[key]
    raise TypeError(f"Expected int or str, got {type(label).__name__}")


# ---------------------------------------------------------------------------
# Dataclass: hierarchical target vector for a single sample
# ---------------------------------------------------------------------------

@dataclass
class HierarchicalTarget:
    """
    All node targets for one sample.

    Attributes
    ----------
    flat_label : int
        Original class in {1,…,6}.
    node_a : int
        Binary: 1=hungry, 0=non-hungry.
    node_b : Optional[int]
        Binary: 1=tired, 0=active-distress.
        None when node_a==1 (sample is hungry, never reaches Node B).
    node_c : Optional[int]
        3-class: 0=pain, 1=reflux, 2=discomfort.
        None when node_a==1 or node_b==1.
    class_name : str
        Human-readable name of flat_label.
    path : str
        Decision path taken, e.g. "A=0 → B=0 → C=1 (reflux)".
    """
    flat_label : int
    node_a     : int
    node_b     : Optional[int]
    node_c     : Optional[int]
    class_name : str = field(init=False)
    path       : str = field(init=False)

    def __post_init__(self) -> None:
        self.class_name = CLASS_NAMES.get(self.flat_label, "unknown")
        parts = [f"A={self.node_a}"]
        if self.node_b is not None:
            parts.append(f"B={self.node_b}")
        if self.node_c is not None:
            parts.append(f"C={self.node_c} ({NODE_C_NAMES[self.node_c]})")
        self.path = " → ".join(parts)

    # Convenience ----------------------------------------------------------------
    @property
    def reaches_node_b(self) -> bool:
        """True if this sample passes Node A (non-hungry)."""
        return self.node_a == 0

    @property
    def reaches_node_c(self) -> bool:
        """True if this sample passes both Node A and Node B."""
        return self.node_a == 0 and self.node_b == 0


# ---------------------------------------------------------------------------
# Core encoder
# ---------------------------------------------------------------------------

class HierarchicalLabelEncoder:
    """
    Encode flat 6-class labels into per-node targets for the InfantCryNet
    decision-tree architecture.

    Parameters
    ----------
    colic_mapping : int
        Which class colic (6) maps to.  Default: 2 (pain).
        Clinical rationale: colic manifests as paroxysmal abdominal pain;
        the same Node C branch handles both.

    Examples
    --------
    >>> enc = HierarchicalLabelEncoder()
    >>> t = enc.encode(1)          # hungry
    >>> t.node_a, t.node_b, t.node_c
    (1, None, None)
    >>> t = enc.encode(5)          # reflux
    >>> t.node_a, t.node_b, t.node_c
    (0, 0, 1)
    >>> t = enc.encode("tired")
    >>> t.node_a, t.node_b, t.node_c
    (0, 1, None)
    """

    def __init__(self, colic_mapping: int = COLIC_MAPPING) -> None:
        self.colic_mapping = colic_mapping
        # Rebuild NODE_C_MAP with current colic_mapping
        self._node_c_map: Dict[int, int] = {
            2: 0,               # pain → 0
            3: 2,               # discomfort → 2
            5: 1,               # reflux → 1
            6: NODE_C_MAP[6],   # colic → same as pain by default
        }
        # Override colic mapping
        self._node_c_map[6] = NODE_C_MAP.get(colic_mapping, 0)

    # ------------------------------------------------------------------
    # Single-sample API
    # ------------------------------------------------------------------

    def encode(self, label: int | str) -> HierarchicalTarget:
        """
        Encode one flat label to a HierarchicalTarget.

        Parameters
        ----------
        label : int | str
            Flat class label or string alias.

        Returns
        -------
        HierarchicalTarget
        """
        flat = _to_int(label)
        node_a = 1 if flat == 1 else 0

        if node_a == 1:
            # Hungry: never reaches B or C
            return HierarchicalTarget(
                flat_label=flat,
                node_a=node_a,
                node_b=None,
                node_c=None,
            )

        node_b = 1 if flat == 4 else 0

        if node_b == 1:
            # Tired: never reaches C
            return HierarchicalTarget(
                flat_label=flat,
                node_a=node_a,
                node_b=node_b,
                node_c=None,
            )

        # Active distress → Node C
        node_c = self._node_c_map[flat]
        return HierarchicalTarget(
            flat_label=flat,
            node_a=node_a,
            node_b=node_b,
            node_c=node_c,
        )

    # ------------------------------------------------------------------
    # Batch API  (returns dict of per-node arrays, mask-aware)
    # ------------------------------------------------------------------

    def encode_batch(
        self,
        labels: List[int | str] | np.ndarray | pd.Series,
    ) -> Dict[str, np.ndarray]:
        """
        Encode a sequence of flat labels into per-node arrays.

        Samples that do not reach a node receive NaN in float arrays and
        -1 in int arrays (mask value). Use the boolean masks to filter
        before training each node classifier.

        Parameters
        ----------
        labels : array-like of int or str
            Flat labels, length N.

        Returns
        -------
        dict with keys:
            "flat"          : int array (N,)  — original labels
            "node_a"        : int array (N,)  — 0/1
            "node_a_mask"   : bool array (N,) — always True (all samples)
            "node_b"        : int array (N,)  — 0/1 or -1 if not reached
            "node_b_mask"   : bool array (N,) — True where node_b valid
            "node_c"        : int array (N,)  — 0/1/2 or -1 if not reached
            "node_c_mask"   : bool array (N,) — True where node_c valid
        """
        targets = [self.encode(lbl) for lbl in labels]
        N = len(targets)

        flat     = np.array([t.flat_label for t in targets], dtype=np.int8)
        node_a   = np.array([t.node_a for t in targets],     dtype=np.int8)
        node_a_m = np.ones(N, dtype=bool)

        node_b   = np.full(N, -1, dtype=np.int8)
        node_b_m = np.zeros(N, dtype=bool)
        node_c   = np.full(N, -1, dtype=np.int8)
        node_c_m = np.zeros(N, dtype=bool)

        for i, t in enumerate(targets):
            if t.node_b is not None:
                node_b[i]   = t.node_b
                node_b_m[i] = True
            if t.node_c is not None:
                node_c[i]   = t.node_c
                node_c_m[i] = True

        return {
            "flat":        flat,
            "node_a":      node_a,
            "node_a_mask": node_a_m,
            "node_b":      node_b,
            "node_b_mask": node_b_m,
            "node_c":      node_c,
            "node_c_mask": node_c_m,
        }

    # ------------------------------------------------------------------
    # Inverse: reconstruct flat distribution from cascaded node probas
    # ------------------------------------------------------------------

    def reconstruct_flat_proba(
        self,
        p_node_a: np.ndarray,               # (N, 2) or (2,)
        p_node_b: np.ndarray,               # (N, 2) or (2,)
        p_node_c: np.ndarray,               # (N, 3) or (3,)
    ) -> np.ndarray:
        """
        Reconstruct the full 6-class posterior from cascaded node probabilities.

        Factored posterior (chain rule of the hierarchy):
          P(Y=1)  = P(A=1)
          P(Y=4)  = P(A=0) · P(B=1)
          P(Y=2)  = P(A=0) · P(B=0) · P(C=0)   [pain]
          P(Y=5)  = P(A=0) · P(B=0) · P(C=1)   [reflux]
          P(Y=3)  = P(A=0) · P(B=0) · P(C=2)   [discomfort]
          P(Y=6)  = 0  (colic absorbed into pain)

        Parameters
        ----------
        p_node_a : (N,2) array  — [:, 1] = P(hungry)
        p_node_b : (N,2) array  — [:, 1] = P(tired | non-hungry)
        p_node_c : (N,3) array  — [:, 0]=P(pain), [:, 1]=P(reflux), [:, 2]=P(discomfort)

        Returns
        -------
        proba : (N, 6) float64 array  — columns ordered 1,2,3,4,5,6
        """
        p_a = np.atleast_2d(p_node_a)   # (N,2)
        p_b = np.atleast_2d(p_node_b)
        p_c = np.atleast_2d(p_node_c)

        N = p_a.shape[0]
        proba = np.zeros((N, 6), dtype=np.float64)

        p_hungry    = p_a[:, 1]
        p_non_hungry = p_a[:, 0]
        p_tired     = p_b[:, 1]
        p_active    = p_b[:, 0]

        proba[:, 0] = p_hungry                            # class 1
        proba[:, 3] = p_non_hungry * p_tired              # class 4
        proba[:, 1] = p_non_hungry * p_active * p_c[:, 0] # class 2 (pain)
        proba[:, 4] = p_non_hungry * p_active * p_c[:, 1] # class 5 (reflux)
        proba[:, 2] = p_non_hungry * p_active * p_c[:, 2] # class 3 (discomfort)
        # class 6 (colic) = 0; already absorbed

        # Renormalise for floating-point safety
        row_sums = proba.sum(axis=1, keepdims=True)
        proba /= np.where(row_sums > 0, row_sums, 1.0)
        return proba

    # ------------------------------------------------------------------
    # Statistics / diagnostics
    # ------------------------------------------------------------------

    def node_statistics(
        self, labels: List[int | str] | np.ndarray
    ) -> pd.DataFrame:
        """
        Return per-node class balance statistics for a label array.

        Useful before training: check that each node has balanced classes
        and a sufficient number of training samples.
        """
        enc = self.encode_batch(labels)
        rows = []

        # Node A
        a_vals, a_cnts = np.unique(enc["node_a"], return_counts=True)
        for v, c in zip(a_vals, a_cnts):
            rows.append({
                "node": "A",
                "class_int": int(v),
                "class_name": "hungry" if v == 1 else "non-hungry",
                "count": int(c),
                "fraction": float(c / len(labels)),
            })

        # Node B
        b_enc = enc["node_b"][enc["node_b_mask"]]
        b_vals, b_cnts = np.unique(b_enc, return_counts=True)
        for v, c in zip(b_vals, b_cnts):
            rows.append({
                "node": "B",
                "class_int": int(v),
                "class_name": "tired" if v == 1 else "active-distress",
                "count": int(c),
                "fraction": float(c / len(b_enc)),
            })

        # Node C
        c_enc = enc["node_c"][enc["node_c_mask"]]
        c_vals, c_cnts = np.unique(c_enc, return_counts=True)
        for v, c in zip(c_vals, c_cnts):
            rows.append({
                "node": "C",
                "class_int": int(v),
                "class_name": NODE_C_NAMES.get(int(v), "?"),
                "count": int(c),
                "fraction": float(c / len(c_enc)),
            })

        df = pd.DataFrame(rows)
        df["imbalance_ratio"] = df.groupby("node")["count"].transform(
            lambda g: g.max() / g.min() if g.min() > 0 else np.inf
        )
        return df

    # ------------------------------------------------------------------
    # Serialisation (for reproducibility / caching)
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict:
        return {"colic_mapping": self.colic_mapping}

    @classmethod
    def from_dict(cls, d: Dict) -> "HierarchicalLabelEncoder":
        return cls(colic_mapping=d.get("colic_mapping", COLIC_MAPPING))


# ---------------------------------------------------------------------------
# Convenience functions (module-level)
# ---------------------------------------------------------------------------

_default_encoder = HierarchicalLabelEncoder()


def encode_label(label: int | str) -> HierarchicalTarget:
    """Module-level shorthand for the default encoder."""
    return _default_encoder.encode(label)


def encode_batch(
    labels: List[int | str] | np.ndarray,
) -> Dict[str, np.ndarray]:
    """Module-level shorthand for batch encoding."""
    return _default_encoder.encode_batch(labels)


def get_node_masks(labels: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Convenience: return boolean masks for training each node.

    Returns
    -------
    dict with keys "node_a", "node_b", "node_c" (boolean arrays of length N)
    """
    enc = encode_batch(labels)
    return {
        "node_a": enc["node_a_mask"],
        "node_b": enc["node_b_mask"],
        "node_c": enc["node_c_mask"],
    }


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    enc = HierarchicalLabelEncoder()

    print("=== Single-sample encoding ===")
    for lbl in [1, 2, 3, 4, 5, 6, "hungry", "tired", "reflux", "belly pain"]:
        t = enc.encode(lbl)
        print(f"  {str(lbl):>12} → flat={t.flat_label}  A={t.node_a}  "
              f"B={str(t.node_b):>4}  C={str(t.node_c):>4}  | {t.path}")

    print("\n=== Batch encoding ===")
    labels = [1, 2, 3, 4, 5, 6, 1, 4, 5, 2, 3]
    batch  = enc.encode_batch(labels)
    print(f"  node_a      : {batch['node_a']}")
    print(f"  node_b      : {batch['node_b']}  (mask: {batch['node_b_mask'].astype(int)})")
    print(f"  node_c      : {batch['node_c']}  (mask: {batch['node_c_mask'].astype(int)})")

    print("\n=== Node statistics ===")
    stats = enc.node_statistics(labels)
    print(stats.to_string(index=False))

    print("\n=== Reconstruct flat proba ===")
    N = 3
    rng = np.random.default_rng(42)
    def rand_proba(k):
        x = rng.dirichlet(np.ones(k), size=N)
        return x
    p_a = rand_proba(2)
    p_b = rand_proba(2)
    p_c = rand_proba(3)
    flat_p = enc.reconstruct_flat_proba(p_a, p_b, p_c)
    print(f"  Shape: {flat_p.shape}")
    print(f"  Row sums: {flat_p.sum(axis=1).round(6)}")
    print(f"  Columns (classes 1–6):\n{flat_p.round(4)}")