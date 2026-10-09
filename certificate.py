"""The certificate: from the eight log-probabilities of a quartet to LR, F, B, K and R.

A quartet is two true facts of one relation, (s1, o1) and (s2, o2). A term
"<a>_given_<b>" is the conditional probability p(a | b), so "o1_given_s1" is
p(o1 | s1). If the conditionals come from one joint distribution that gives the
four pairs positive probability, then

    p(s1|o1) p(o1|s2) p(s2|o2) p(o2|s1)
    -----------------------------------  =  1 ,
    p(o1|s1) p(s1|o2) p(o2|s2) p(s2|o1)

so its logarithm LR (in nats) is 0. Grouped by prompt, the terms form four
margins, each comparing the two candidates under one prompt and positive when
the true partner wins:

    F1 = lp(o1|s1) - lp(o2|s1)      F2 = lp(o2|s2) - lp(o1|s2)     forward, object given subject
    B1 = lp(s1|o1) - lp(s2|o1)      B2 = lp(s2|o2) - lp(s1|o2)     backward, subject given object

and LR = (B1 + B2) - (F1 + F2) = 2 (B - F) with F = (F1 + F2)/2 and B = (B1 + B2)/2.
LR < 0 means the forward margin is larger.

    K = number of the four margins that are > 0 (a tie counts as wrong), 0 to 4
    R = |LR| / (2|F| + 2|B|) = |B - F| / (|F| + |B|), 0 to 1, and 0 when F = B = 0
"""

from __future__ import annotations

import math

TERMS = ["o1_given_s1", "o2_given_s1", "o1_given_s2", "o2_given_s2",   # object given subject
         "s1_given_o1", "s2_given_o1", "s1_given_o2", "s2_given_o2"]   # subject given object
NUMERATOR = ["s1_given_o1", "o1_given_s2", "s2_given_o2", "o2_given_s1"]
DENOMINATOR = ["o1_given_s1", "s1_given_o2", "o2_given_s2", "s2_given_o1"]


def log_ratio(lp: dict[str, float]) -> float:
    return sum(lp[t] for t in NUMERATOR) - sum(lp[t] for t in DENOMINATOR)


def derive(lp: dict[str, float]) -> dict:
    """All statistics of one quartet from its eight log-probabilities (nats)."""
    missing = [t for t in TERMS if t not in lp]
    if missing:
        raise KeyError(f"missing terms {missing}")
    if any(not math.isfinite(lp[t]) for t in TERMS):
        raise ValueError(f"non-finite log-probability in {lp}")
    f1 = lp["o1_given_s1"] - lp["o2_given_s1"]
    f2 = lp["o2_given_s2"] - lp["o1_given_s2"]
    b1 = lp["s1_given_o1"] - lp["s2_given_o1"]
    b2 = lp["s2_given_o2"] - lp["s1_given_o2"]
    F, B = (f1 + f2) / 2, (b1 + b2) / 2
    LR = log_ratio(lp)
    if not math.isclose(LR, 2 * (B - F), rel_tol=1e-9, abs_tol=1e-6):
        raise ArithmeticError(f"LR = 2(B - F) violated: {LR} vs {2 * (B - F)}")
    forward_correct, backward_correct = int(f1 > 0) + int(f2 > 0), int(b1 > 0) + int(b2 > 0)
    scale = abs(F) + abs(B)
    return {
        "log_ratio": LR,
        "abs_log_ratio": abs(LR),
        "forward_margin": F,
        "backward_margin": B,
        "relative_incoherence": abs(B - F) / scale if scale > 0 else 0.0,   # = |LR| / (2 scale), never above 1 in floats
        "knowledge_score": forward_correct + backward_correct,
        "forward_correct": forward_correct,
        "backward_correct": backward_correct,
    }
