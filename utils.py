"""
utils.py

General utilities for measuring lexical, syntactic, and semantic
variability and human/model uncertainty for NLG tasks with
multi-reference datasets (e.g., ASSET) and multi-sample model outputs.

Assumptions:
- Hugging Face dataset has a test (or given) split, with a field that
  contains a list of human references per example (e.g. "simplifications").
- CSV contains at least: id, strategy, output.
"""

from __future__ import annotations

import itertools
import random
import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.stats import wasserstein_distance
import spacy
from sentence_transformers import SentenceTransformer, util as st_util


# ---------------------------------------------------------------------
# Tokenization & n-grams
# ---------------------------------------------------------------------

def tokenize_words(s: str) -> List[str]:
    """
    Simple word tokenizer consistent with paper:
    - lowercase
    - keeps alphanumerics and contractions
    """
    if not isinstance(s, str):
        s = "" if s is None else str(s)
    return re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?", s.lower())


def ngrams(tokens: Sequence[str], n: int) -> List[Tuple[str, ...]]:
    """Return list of n-grams as tuples."""
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


# ---------------------------------------------------------------------
# Distance probes
# ---------------------------------------------------------------------

@dataclass
class DistanceProbes:
    """
    Encapsulates the three probes used in the paper:

    - lexical: unigram distance
    - syntactic: POS bigram distance
    - semantic: SBERT cosine distance
    """
    nlp: spacy.language.Language
    sbert: SentenceTransformer

    def lexical_distance(self, s1: str, s2: str) -> float:
        """Unigram Jaccard-like distance between word sets."""
        t1 = tokenize_words(s1)
        t2 = tokenize_words(s2)

        set1 = set(t1)
        set2 = set(t2)

        union = set1 | set2
        if not union:
            return 0.0

        non_matching = len(union - (set1 & set2))
        return non_matching / len(union)

    def syntactic_distance(self, s1: str, s2: str) -> float:
        """POS bigram distance (Jaccard-like)."""
        doc1 = self.nlp(s1)
        doc2 = self.nlp(s2)

        pos1 = [t.pos_ for t in doc1]
        pos2 = [t.pos_ for t in doc2]

        bigrams1 = set(ngrams(pos1, 2))
        bigrams2 = set(ngrams(pos2, 2))

        union = bigrams1 | bigrams2
        if not union:
            return 0.0

        non_matching = len(union - (bigrams1 & bigrams2))
        return non_matching / len(union)

    def semantic_distance(self, s1: str, s2: str) -> float:
        """Cosine distance between SBERT sentence embeddings."""
        # Using convert_to_tensor to allow GPU use
        emb1 = self.sbert.encode(s1, convert_to_tensor=True)
        emb2 = self.sbert.encode(s2, convert_to_tensor=True)
        cosine_sim = st_util.cos_sim(emb1, emb2).item()
        return 1.0 - cosine_sim


def load_default_probes(
    spacy_model: str = "en_core_web_sm",
    sbert_model: str = "all-distilroberta-v1",
    device: Optional[str] = None,
) -> DistanceProbes:
    """
    Load spaCy and SentenceTransformer models and wrap into DistanceProbes.

    device:
        - "cuda" or "cuda:0" to force GPU
        - "cpu"
        - None -> auto: "cuda:0" if available, else "cpu"
    """
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    nlp = spacy.load(spacy_model)
    sbert = SentenceTransformer(sbert_model, device=device)

    return DistanceProbes(nlp=nlp, sbert=sbert)


# ---------------------------------------------------------------------
# H(x): human–human distances
# ---------------------------------------------------------------------

def compute_H_for_id(
    idx: int,
    dataset_split,
    refs_field: str,
    probes: DistanceProbes,
) -> Dict:
    """
    Compute H(x) for a single example index (id):

    H_lexical, H_syntactic, H_semantic
    = pairwise distances between all distinct human references.
    """
    refs: List[str] = dataset_split[idx][refs_field]
    pairs = list(itertools.combinations(refs, 2))

    lexical_vals = []
    syntactic_vals = []
    semantic_vals = []

    for s1, s2 in pairs:
        lexical_vals.append(probes.lexical_distance(s1, s2))
        syntactic_vals.append(probes.syntactic_distance(s1, s2))
        semantic_vals.append(probes.semantic_distance(s1, s2))

    return {
        "id": int(idx),
        "H_lexical": lexical_vals,
        "H_syntactic": syntactic_vals,
        "H_semantic": semantic_vals,
    }


def compute_all_H(
    dataset_split,
    refs_field: str,
    probes: DistanceProbes,
    max_examples: Optional[int] = None,
    progress: bool = True,
) -> List[Dict]:
    """
    Compute H(x) for all examples (or first max_examples) in a dataset split.

    Returns: list of dicts with keys:
        - id
        - H_lexical
        - H_syntactic
        - H_semantic
    """
    n = len(dataset_split) if max_examples is None else min(len(dataset_split), max_examples)
    all_H: List[Dict] = []

    for i in range(n):
        if progress and i % 50 == 0:
            print(f"[H] Processing example {i}/{n}")
        all_H.append(compute_H_for_id(i, dataset_split, refs_field, probes))

    return all_H


# ---------------------------------------------------------------------
# M(x): model–model distances
# ---------------------------------------------------------------------

def compute_M_for_group(
    outputs: Sequence[str],
    probes: DistanceProbes,
) -> Dict[str, List[float]]:
    """
    Given 10 model outputs for a given (id, strategy),
    compute M_lexical, M_syntactic, M_semantic
    as pairwise distances between model outputs.
    """
    pairs = list(itertools.combinations(outputs, 2))

    M_lex = []
    M_syn = []
    M_sem = []

    for s1, s2 in pairs:
        M_lex.append(probes.lexical_distance(s1, s2))
        M_syn.append(probes.syntactic_distance(s1, s2))
        M_sem.append(probes.semantic_distance(s1, s2))

    return {
        "M_lexical": M_lex,
        "M_syntactic": M_syn,
        "M_semantic": M_sem,
    }


def compute_all_M(
    df: pd.DataFrame,
    probes: DistanceProbes,
    id_col: str = "id",
    strategy_col: str = "strategy",
    output_col: str = "output",
    progress: bool = True,
) -> List[Dict]:
    """
    Compute M(x) for all (id, strategy) groups in a model-output DataFrame.

    df is expected to contain multiple samples per (id, strategy),
    e.g. 10 generations per decoding strategy.
    """
    # Ensure outputs are strings
    df = df.copy()
    df[output_col] = df[output_col].fillna("").astype(str)

    all_M: List[Dict] = []

    for (id_val, strat), group in df.groupby([id_col, strategy_col]):
        idx = int(id_val)
        if progress and idx % 50 == 0:
            print(f"[M] Processing id={idx}, strategy={strat}")
        outputs = group[output_col].tolist()
        M_dict = compute_M_for_group(outputs, probes)
        all_M.append({
            "id": idx,
            "strategy": strat,
            **M_dict,
        })

    return all_M


# ---------------------------------------------------------------------
# C(x): human–model distances
# ---------------------------------------------------------------------

def compute_all_C(
    df: pd.DataFrame,
    dataset_split,
    refs_field: str,
    probes: DistanceProbes,
    id_col: str = "id",
    strategy_col: str = "strategy",
    output_col: str = "output",
    progress: bool = True,
) -> List[Dict]:
    """
    Compute C(x) for all (id, strategy) groups:

    C_lexical, C_syntactic, C_semantic
    = distances between all human references and all model outputs.
    """
    df = df.copy()
    df[output_col] = df[output_col].fillna("").astype(str)

    all_C: List[Dict] = []

    for (id_val, strat), group in df.groupby([id_col, strategy_col]):
        idx = int(id_val)
        if progress and idx % 50 == 0:
            print(f"[C] Processing id={idx}, strategy={strat}")

        human_refs: List[str] = dataset_split[idx][refs_field]
        model_outputs = group[output_col].tolist()

        C_lex: List[float] = []
        C_syn: List[float] = []
        C_sem: List[float] = []

        for h in human_refs:
            for m in model_outputs:
                C_lex.append(probes.lexical_distance(h, m))
                C_syn.append(probes.syntactic_distance(h, m))
                C_sem.append(probes.semantic_distance(h, m))

        all_C.append({
            "id": idx,
            "strategy": strat,
            "C_lexical": C_lex,
            "C_syntactic": C_syn,
            "C_semantic": C_sem,
        })

    return all_C


# ---------------------------------------------------------------------
# Human baseline: random 5–5 splits
# ---------------------------------------------------------------------

def compute_cross_human_distances(
    refs: Sequence[str],
    distance_fn: Callable[[str, str], float],
    rng: Optional[random.Random] = None,
) -> List[float]:
    """
    refs: list of 10 human references
    distance_fn: lexical_distance, syntactic_distance, semantic_distance
    rng: optional random.Random instance for reproducible splits

    Returns: list of 25 distances between groupA and groupB (5×5).
    """
    if rng is None:
        rng = random

    idxs = list(range(len(refs)))
    rng.shuffle(idxs)

    groupA_idxs = idxs[:5]
    groupB_idxs = idxs[5:]

    groupA = [refs[i] for i in groupA_idxs]
    groupB = [refs[i] for i in groupB_idxs]

    distances: List[float] = []
    for s1 in groupA:
        for s2 in groupB:
            distances.append(distance_fn(s1, s2))

    return distances


def compute_human_baseline_W(
    all_H: List[Dict],
    dataset_split,
    refs_field: str,
    probes: DistanceProbes,
    seed: Optional[int] = 42,
    progress: bool = True,
) -> List[Dict]:
    """
    For each example, compute a human baseline Wasserstein distance:

        W_lexical = W1(H_split_lex(x), H_full_lex(x))
        W_syntactic = W1(H_split_syn(x), H_full_syn(x))
        W_semantic = W1(H_split_sem(x), H_full_sem(x))

    H_split is based on a random 5–5 split of human references.
    Returns list of dicts with per-id W distances.
    """
    if seed is not None:
        base_rng = random.Random(seed)
    else:
        base_rng = random

    human_W: List[Dict] = []

    for i, h in enumerate(all_H):
        id_val = int(h["id"])
        if progress and i % 50 == 0:
            print(f"[Human W] Processing example {i}/{len(all_H)} (id={id_val})")

        refs: List[str] = dataset_split[id_val][refs_field]
        # fresh RNG per id for reproducibility but independence
        rng = random.Random(base_rng.randint(0, 10**9))

        Hsplit_lex = compute_cross_human_distances(refs, probes.lexical_distance, rng)
        Hsplit_syn = compute_cross_human_distances(refs, probes.syntactic_distance, rng)
        Hsplit_sem = compute_cross_human_distances(refs, probes.semantic_distance, rng)

        Hfull_lex = h["H_lexical"]
        Hfull_syn = h["H_syntactic"]
        Hfull_sem = h["H_semantic"]

        w_lex = wasserstein_distance(Hsplit_lex, Hfull_lex)
        w_syn = wasserstein_distance(Hsplit_syn, Hfull_syn)
        w_sem = wasserstein_distance(Hsplit_sem, Hfull_sem)

        human_W.append({
            "id": id_val,
            "W_lexical": w_lex,
            "W_syntactic": w_syn,
            "W_semantic": w_sem,
        })

    return human_W


# ---------------------------------------------------------------------
# Wasserstein: C vs H
# ---------------------------------------------------------------------

def compute_W_C_vs_H(
    all_C: List[Dict],
    all_H: List[Dict],
) -> List[Dict]:
    """
    For each (id, strategy) in all_C, compute:

        W_lexical = W1(C_lexical(x), H_lexical(x))
        W_syntactic = W1(C_syntactic(x), H_syntactic(x))
        W_semantic = W1(C_semantic(x), H_semantic(x))

    Returns a list of dicts with keys:
        id, strategy, W_lexical, W_syntactic, W_semantic
    """
    H_lookup = {h["id"]: h for h in all_H}
    all_W: List[Dict] = []

    for c in all_C:
        id_val = int(c["id"])
        strat = c["strategy"]

        h = H_lookup[id_val]

        w_lex = wasserstein_distance(c["C_lexical"], h["H_lexical"])
        w_syn = wasserstein_distance(c["C_syntactic"], h["H_syntactic"])
        w_sem = wasserstein_distance(c["C_semantic"], h["H_semantic"])

        all_W.append({
            "id": id_val,
            "strategy": strat,
            "W_lexical": w_lex,
            "W_syntactic": w_syn,
            "W_semantic": w_sem,
        })

    return all_W


def summarize_mean_W_by_strategy(
    all_W: List[Dict],
) -> pd.DataFrame:
    """
    Convert list of W results to a DataFrame and compute mean W per strategy.
    """
    df_W = pd.DataFrame(all_W)
    grouped = df_W.groupby("strategy").agg({
        "W_lexical": "mean",
        "W_syntactic": "mean",
        "W_semantic": "mean",
    }).reset_index()
    return grouped
