"""
engine.py
---------
Core solver logic for Letroso.

Feedback states per position (base-5 encoded):
  BLACK          (0) — letter not in the answer
  YELLOW         (1) — letter in the answer, wrong position
  GREEN          (2) — exact position match (isolated — no adjacent GREEN)
  GREEN_ADJACENT (3) — exact match AND at least one neighbour is also GREEN
                        (the game draws a connection line between them)
  GREEN_BORDER   (4) — exact match AND this position is the first or last
                        position of the ANSWER word (rounded corners)

Priority when multiple states apply:
    GREEN_ADJACENT > GREEN_BORDER > GREEN

Cross-length:
  guess and answer may differ in length.
  GREEN is only assigned at positions i < min(len(guess), len(answer)).
  The pattern always has len(guess) positions.
  GREEN_BORDER at position i means i == 0 (answer start) or i == len(answer)-1
  (answer end), which lets the solver deduce the answer's length.
"""

import math
from collections import defaultdict
from functools import lru_cache
from typing import Optional

# ── Feedback state constants ──────────────────────────────────────────────────
BLACK          = 0
YELLOW         = 1
GREEN          = 2
GREEN_ADJACENT = 3   # connected to neighbouring green
GREEN_BORDER   = 4   # at first/last position of the answer

_BASE = 5   # states per position


# ──────────────────────────────────────────────────────────────────────────────
# Pattern computation
# ──────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=8_000_000)
def get_pattern(guess: str, answer: str) -> int:
    """
    Return the Letroso feedback pattern for (guess, answer) as a base-5 int.
    Works for any combination of word lengths.
    """
    L1 = len(guess)
    L2 = len(answer)
    min_L = min(L1, L2)

    pattern    = [BLACK] * L1
    used_ans   = [False] * L2
    used_guess = [False] * L1

    # ── Phase 1: greens (only at overlapping positions) ──────────────────────
    for i in range(min_L):
        if guess[i] == answer[i]:
            pattern[i]    = GREEN
            used_ans[i]   = True
            used_guess[i] = True

    # ── Phase 2: yellows ─────────────────────────────────────────────────────
    available: dict[str, int] = {}
    for j in range(L2):
        if not used_ans[j]:
            available[answer[j]] = available.get(answer[j], 0) + 1

    avail = dict(available)
    for i in range(L1):
        if used_guess[i]:
            continue
        ch = guess[i]
        if avail.get(ch, 0) > 0:
            pattern[i] = YELLOW
            avail[ch] -= 1

    # ── Phase 3: upgrade GREEN → ADJACENT or BORDER ──────────────────────────
    is_green = [pattern[i] == GREEN for i in range(L1)]

    for i in range(L1):
        if not is_green[i]:
            continue

        has_adj = (
            (i > 0     and is_green[i - 1]) or
            (i < L1 - 1 and is_green[i + 1])
        )
        at_border = (i == 0) or (i == L2 - 1 and i < L1)

        if has_adj:
            pattern[i] = GREEN_ADJACENT
        elif at_border:
            pattern[i] = GREEN_BORDER

    # ── Encode base-6 ────────────────────────────────────────────────────────
    result = 0
    for i, v in enumerate(pattern):
        result += v * (_BASE ** i)
    return result


def is_win_pattern(pattern_int: int, length: int) -> bool:
    """True if every position is a GREEN variant (≥ 2)."""
    for _ in range(length):
        if pattern_int % _BASE < GREEN:
            return False
        pattern_int //= _BASE
    return True


# ──────────────────────────────────────────────────────────────────────────────
# Filtering
# ──────────────────────────────────────────────────────────────────────────────

def filter_candidates(
    candidates: list[str],
    guess: str,
    pattern_int: int,
) -> list[str]:
    """Keep only words whose get_pattern(guess, word) equals pattern_int."""
    return [w for w in candidates if get_pattern(guess, w) == pattern_int]


# ──────────────────────────────────────────────────────────────────────────────
# Entropy
# ──────────────────────────────────────────────────────────────────────────────

def compute_entropy(
    guess: str,
    candidates: list[str],
    weights: dict[str, float],
) -> float:
    total = sum(weights[w] for w in candidates)
    if total == 0:
        return 0.0

    bucket_prob: dict[int, float] = defaultdict(float)
    for answer in candidates:
        bucket_prob[get_pattern(guess, answer)] += weights[answer] / total

    return -sum(p * math.log2(p) for p in bucket_prob.values() if p > 0)


def rank_guesses(
    candidates: list[str],
    vocab: list[str],
    weights: dict[str, float],
    top_n: int = 10,
    progress_fn=None,
) -> list[tuple[str, float]]:
    """
    Return top-n guesses by descending entropy.
    When ≤ 20 candidates remain, only rank those (O(20²) instead of O(V·20)).

    progress_fn(done, total) is called periodically if provided.
    """
    candidate_set = set(candidates)
    search_space  = candidates if len(candidates) <= 20 else vocab
    total = len(search_space)
    results: list[tuple[str, float]] = []
    for i, g in enumerate(search_space):
        results.append((g, compute_entropy(g, candidates, weights)))
        if progress_fn and (i + 1) % 500 == 0:
            progress_fn(i + 1, total)
    if progress_fn:
        progress_fn(total, total)
    results.sort(key=lambda x: (x[1], x[0] in candidate_set), reverse=True)
    return results[:top_n]


# ──────────────────────────────────────────────────────────────────────────────
# Pattern parsing (user / API input)
# ──────────────────────────────────────────────────────────────────────────────

_CHAR_MAP = {
    # Letters
    "B": BLACK,          
    "Y": YELLOW,          
    "G": GREEN,           
    "A": GREEN_ADJACENT,
    "P": GREEN_BORDER,    
    # Digits
    "0": BLACK,
    "1": YELLOW,
    "2": GREEN,
    "3": GREEN_ADJACENT,
    "4": GREEN_BORDER,
}


def parse_pattern(feedback: str, length: int) -> Optional[int]:
    """
    Parse a feedback string into a base-5 pattern int.

    Accepted formats (case-insensitive, spaces optional):
      "GAPB"      — letter codes (B=black Y=yellow G=green A=adjacent P=border)
      "2 3 4 0"   — numeric spaced
      "2340"      — numeric compact
    Returns None if the input is invalid.
    """
    feedback = feedback.strip().upper()
    tokens   = feedback.split() if " " in feedback else list(feedback)

    if len(tokens) != length:
        return None

    result = 0
    for i, tok in enumerate(tokens):
        if tok not in _CHAR_MAP:
            return None
        result += _CHAR_MAP[tok] * (_BASE ** i)
    return result
