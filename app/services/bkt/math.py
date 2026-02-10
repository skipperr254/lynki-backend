from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Dict


def _clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


@dataclass(frozen=True)
class BKTParams:
    """Standard BKT parameters.
    p_learn: P(T) - probability of learning between opportunities
    p_guess: P(G) - probability of correct when not mastered
    p_slip:  P(S) - probability of incorrect when mastered
    """
    p_learn: float
    p_guess: float
    p_slip: float


def soft_evidence_update(p_mastery: float, q: float, params: BKTParams) -> Tuple[float, Dict[str, float]]:
    """Soft-evidence Bayesian Knowledge Tracing update.

    Inputs:
        p_mastery: Prior mastery probability P(K=1) before seeing this response.
        q: Claude score normalized to [0,1], interpreted as P(correct).
        params: BKTParams with (T, G, S).

    Returns:
        p_next: Updated mastery probability after Bayes update + learning transition.
        debug: Intermediate values for diagnostics/tests.
    """
    p = _clamp01(float(p_mastery))
    q = _clamp01(float(q))
    T = _clamp01(float(params.p_learn))
    G = _clamp01(float(params.p_guess))
    S = _clamp01(float(params.p_slip))

    # Likelihoods (soft observation):
    # P(obs | K=1) = q*(1-S) + (1-q)*S
    # P(obs | K=0) = q*G + (1-q)*(1-G)
    like_k1 = q * (1.0 - S) + (1.0 - q) * S
    like_k0 = q * G + (1.0 - q) * (1.0 - G)

    denom = p * like_k1 + (1.0 - p) * like_k0
    if denom <= 1e-12:
        p_post = p  # fallback: do not update on degenerate evidence
    else:
        p_post = (p * like_k1) / denom

    # Learning transition
    p_next = _clamp01(p_post + (1.0 - p_post) * T)

    debug = {
        "p_prior": p,
        "q": q,
        "T": T,
        "G": G,
        "S": S,
        "like_k1": like_k1,
        "like_k0": like_k0,
        "denom": denom,
        "p_post": p_post,
        "p_next": p_next,
    }
    return p_next, debug
