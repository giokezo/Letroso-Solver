import math
from collections import defaultdict
from functools import lru_cache
from typing import Optional
from collections import Counter

import multiprocessing as mp
import os

BLACK  = 0
YELLOW = 1
GREEN  = 2
BORDER = 3

_BASE = 8   # state*2 + concat_right  →  values 0–7

@lru_cache(maxsize=8_000_000)
def get_pattern(guess: str, answer: str) -> int:
    """
    Compute feedback pattern to match the site's JS algorithm exactly.

    The site uses edit-distance DP (cost=0 for match, inf for mismatch, 1 for deletion),
    equivalent to LCS. Enumerates ALL LCS alignments, scores each by
    boundary_bonus + 3*consecutive_pairs, breaks ties by pattern.join() descending
    (JS pattern values: 0=absent, 1=present/yellow, 2=head of green run, 3=tail).
    After selecting the best alignment, applies LCS-min-1 and BORDER rules.
    """
    L1, L2 = len(guess), len(answer)

    # LCS DP (bottom-right to top-left)
    dp = [[0] * (L2 + 1) for _ in range(L1 + 1)]
    for i in range(L1 - 1, -1, -1):
        for j in range(L2 - 1, -1, -1):
            if guess[i] == answer[j]:
                dp[i][j] = dp[i + 1][j + 1] + 1
            else:
                dp[i][j] = max(dp[i + 1][j], dp[i][j + 1])
    lcs_len = dp[0][0]

    # Enumerate all LCS alignments; find best by (score DESC, js_pattern_str DESC)
    best_score = -1
    best_pat_str = ""
    best_align: list[tuple[int, int]] = []

    def dfs(k: int, i: int, j: int, current: list[tuple[int, int]]) -> None:
        nonlocal best_score, best_pat_str, best_align
        if k == 0:
            # Score: boundary bonuses + 3 per consecutive pair
            score = 0
            if current and current[0] == (0, 0):
                score += 1
            if current and current[-1] == (L1 - 1, L2 - 1):
                score += 1
            for m in range(len(current) - 1):
                gi, gj = current[m]
                ni, nj = current[m + 1]
                if ni == gi + 1 and nj == gj + 1:
                    score += 3

            # Build JS-style pattern [0=B,1=Y,2=head,3=tail] for lex comparison
            green_d: dict[int, int] = {gi: gj for gi, gj in current}
            pat = [0] * L1
            for gi in green_d:
                pat[gi] = 2  # head initially
            gl = sorted(green_d)
            for m in range(len(gl) - 1):
                gi2, ni2 = gl[m], gl[m + 1]
                if ni2 == gi2 + 1 and green_d[ni2] == green_d[gi2] + 1:
                    pat[ni2] = 3  # tail

            # Yellow assignment: build available pool after subtracting greens
            avail_y: dict[str, int] = {}
            for ch in answer:
                avail_y[ch] = avail_y.get(ch, 0) + 1
            for gj in green_d.values():
                avail_y[answer[gj]] -= 1
            used_y = [False] * L1
            for j2 in range(L2):
                if j2 in green_d.values():
                    continue
                ch = answer[j2]
                if avail_y.get(ch, 0) <= 0:
                    continue
                for i2 in range(L1):
                    if pat[i2] >= 2 or used_y[i2]:
                        continue
                    if guess[i2] == ch:
                        pat[i2] = 1
                        used_y[i2] = True
                        avail_y[ch] -= 1
                        break

            pat_str = "".join(str(x) for x in pat)
            if score > best_score or (score == best_score and pat_str > best_pat_str):
                best_score = score
                best_pat_str = pat_str
                best_align = list(current)
            return

        # Enumerate next match position (i2, j2)
        for i2 in range(i, L1 - k + 1):
            for j2 in range(j, L2 - k + 1):
                if guess[i2] != answer[j2]:
                    continue
                # Must be on an LCS path
                if dp[i2 + 1][j2 + 1] + 1 != dp[i2][j2]:
                    continue
                current.append((i2, j2))
                dfs(k - 1, i2 + 1, j2 + 1, current)
                current.pop()

    dfs(lcs_len, 0, 0, [])

    align = best_align
    green_d2 = {gi: gj for gi, gj in align}
    n_greens = len(green_d2)

    # Yellow assignment
    ans_used = [False] * L2
    for gj in green_d2.values():
        ans_used[gj] = True
    state = [BLACK] * L1
    for gi in green_d2:
        state[gi] = GREEN
    used_y = [False] * L1
    for j in range(L2):
        if ans_used[j]:
            continue
        ch = answer[j]
        for i in range(L1):
            if state[i] >= GREEN or used_y[i]:
                continue
            if guess[i] == ch:
                state[i] = YELLOW
                used_y[i] = True
                break
    
    total_matches = sum(
    min(guess.count(c), answer.count(c))
    for c in set(guess))

    # LCS-min-1 rule: exactly 1 match and guess is longer than answer and not at a real boundary.
    # When L1 > L2, a lone interior match is shown as yellow by the site.
    # When L1 == L2, the match stays green (prevents false yellows on equal-length pairs).
    if n_greens == 1 and L1 > L2 and total_matches > 1:
        gi, gj = align[0]
        if not (gi == 0 and gj == 0) and not (gi == L1 - 1 and gj == L2 - 1):
            state[gi] = YELLOW

    # Recalculate green dict after LCS-min-1
    green_d3 = {gi: gj for gi, gj in align if state[gi] >= GREEN}
    n_greens2 = len(green_d3)

    # BORDER upgrade: any green matched at answer boundary → BORDER
    for gi, gj in green_d3.items():
        at_end = gi == L1 - 1 and gj == L2 - 1
        at_start = gi == 0 and gj == 0
        if at_end and n_greens2 >= 1:
            state[gi] = BORDER
        if at_start and n_greens2 >= 1:
            state[gi] = BORDER

    # Concat: adjacent greens consecutive in both guess and answer
    green_list = sorted(green_d3.keys())
    concat = [False] * L1
    for m in range(len(green_list) - 1):
        gi = green_list[m]
        ni = green_list[m + 1]
        if ni == gi + 1 and green_d3[ni] == green_d3[gi] + 1:
            concat[gi] = True

    # Encode
    result = 0
    for i in range(L1):
        v = state[i] * 2 + (1 if concat[i] else 0)
        result += v * (_BASE ** i)
    return result

def decode_pattern(pattern_int: int, length: int) -> tuple[list[int], list[bool]]:
    """Decode a pattern int into (states, concat_rights) for debugging."""
    states = []
    concats = []
    for _ in range(length):
        v = pattern_int % _BASE
        states.append(v // 2)
        concats.append(bool(v % 2))
        pattern_int //= _BASE
    return states, concats


def pattern_to_str(pattern_int: int, length: int) -> str:
    """Convert a pattern int to human-readable feedback string."""
    states, concats = decode_pattern(pattern_int, length)
    _MAP = {BLACK: "B", YELLOW: "Y", GREEN: "G", BORDER: "P"}
    parts = []
    for i in range(length):
        parts.append(_MAP[states[i]])
        if concats[i] and i < length - 1:
            parts.append("O")
    return "".join(parts)


def is_win_pattern(pattern_int: int, length: int) -> bool:
    """
    True only for the exact win pattern  PO(GO)^N P:
      - position 0        : BORDER, concat_right = True
      - positions 1..L-2  : GREEN,  concat_right = True
      - position L-1      : BORDER, concat_right = False

    This means the guessed word spans the full target consecutively
    from left border to right border (i.e. the guess IS the target).
    Single-letter edge case: just BORDER with concat_right = False.
    """
    if length == 1:
        v = pattern_int % _BASE
        return v // 2 == BORDER and not bool(v % 2)

    for i in range(length):
        v      = pattern_int % _BASE
        state  = v // 2
        concat = bool(v % 2)
        if i == 0:
            if state != BORDER or not concat:
                return False
        elif i == length - 1:
            if state != BORDER or concat:
                return False
        else:
            if state != GREEN or not concat:
                return False
        pattern_int //= _BASE
    return True


def filter_candidates(
    candidates: list[str],
    guess: str,
    pattern_int: int,
) -> list[str]:
    """Keep only words whose get_pattern(guess, word) equals pattern_int."""
    return [w for w in candidates if get_pattern(guess, w) == pattern_int]
    # states, concats = decode_pattern(pattern_int, len(guess))
    # return [w for w in candidates if matches_feedback(guess, w, states, concats)]

def compute_entropy(
    guess: str,
    candidates: list[str],
    weights: dict[str, float],
) -> float:
    total = sum(weights[w] for w in candidates)
    if total == 0:
        return 0.0

    bucket: dict[int, float] = defaultdict(float)
    for answer in candidates:
        bucket[get_pattern(guess, answer)] += weights[answer] / total

    return -sum(p * math.log2(p) for p in bucket.values() if p > 0)


# Per-worker globals set by the pool initializer (avoids re-serializing
# candidates and weights for every chunk).
_worker_candidates: list[str] = []
_worker_weights:    dict[str, float] = {}
_worker_total:      float = 0.0


def _worker_init(candidates: list[str], weights: dict[str, float]) -> None:
    global _worker_candidates, _worker_weights, _worker_total
    _worker_candidates = candidates
    _worker_weights    = weights
    _worker_total      = sum(weights[w] for w in candidates)


def _entropy_worker(chunk: list[str]) -> list[tuple[str, float]]:
    """Compute entropy for each guess in chunk against the shared candidate set."""
    results = []
    total = _worker_total
    if total == 0:
        return [(g, 0.0) for g in chunk]
    for g in chunk:
        bucket: dict[int, float] = defaultdict(float)
        for answer in _worker_candidates:
            bucket[get_pattern(g, answer)] += _worker_weights[answer] / total
        ent = -sum(p * math.log2(p) for p in bucket.values() if p > 0)
        results.append((g, ent))
    return results


def rank_guesses(
    candidates: list[str],
    vocab: list[str],
    weights: dict[str, float],
    top_n: int = 10,
    progress_fn=None,
) -> list[tuple[str, float]]:
    """
    Return top-n guesses by descending entropy.
    When ≤ 500 candidates remain, only rank candidates (much faster).
    Uses multiprocessing; candidates+weights are sent once per worker
    via an initializer instead of being re-pickled for every chunk.
    """

    candidate_set = set(candidates)
    # Always guess from the remaining candidates — any of them could be the answer,
    # and guessing from outside that set rarely helps enough to justify the cost.
    search_space = candidates
    total        = len(search_space)

    # # Single-threaded for small search spaces
    # if total <= 200:
    #     results: list[tuple[str, float]] = []
    #     for i, g in enumerate(search_space):
    #         results.append((g, compute_entropy(g, candidates, weights)))
    #         if progress_fn and (i + 1) % 50 == 0:
    #             progress_fn(i + 1, total)
    #     if progress_fn:
    #         progress_fn(total, total)
    #     results.sort(key=lambda x: (x[1], x[0] in candidate_set), reverse=True)
    #     return results[:top_n]

    # Split into many small chunks for frequent progress updates.
    # Each chunk is just a list[str] — candidates/weights go via initializer.
    n_workers  = os.cpu_count() or 4
    chunk_size = max(1, total // (n_workers * 10))
    chunks     = [search_space[i:i + chunk_size] for i in range(0, total, chunk_size)]

    if progress_fn:
        progress_fn(0, total)

    results = []
    with mp.Pool(
        n_workers,
        initializer=_worker_init,
        initargs=(candidates, weights),
    ) as pool:
        for chunk_results in pool.imap_unordered(_entropy_worker, chunks):
            results.extend(chunk_results)
            if progress_fn:
                progress_fn(min(len(results), total), total)

    results.sort(key=lambda x: (x[1], x[0] in candidate_set), reverse=True)
    return results[:top_n]


_CHAR_MAP = {
    "B": BLACK,   "0": BLACK,
    "Y": YELLOW,  "1": YELLOW,
    "G": GREEN,   "2": GREEN,
    "P": BORDER,  "3": BORDER,
}


def parse_pattern(feedback: str, length: int) -> Optional[int]:
    """
    Parse a feedback string into a base-8 pattern int.

    Accepted tokens:
      B/0 = black,  Y/1 = yellow,  G/2 = green,  P/3 = border,  O = concat
    Formats: "BGOGOGB", "BGYGB", "0 2 O 2 O 2 0", etc.
    Returns None if the input is invalid.
    """
    feedback = feedback.strip().upper()
    tokens = list(feedback.replace(" ", ""))

    states: list[int] = []
    concat: list[bool] = []

    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok == "O":
            if not states:
                return None      # O at start
            if states[-1] < GREEN:
                return None      # O after non-green
            concat[-1] = True
            i += 1
            continue
        if tok in _CHAR_MAP:
            # If previous position had concat, verify this is green-type
            if concat and concat[-1] and _CHAR_MAP[tok] < GREEN:
                return None      # O before non-green
            states.append(_CHAR_MAP[tok])
            concat.append(False)
            i += 1
        else:
            return None

    if len(states) != length:
        return None

    result = 0
    for i in range(length):
        v = states[i] * 2 + (1 if concat[i] else 0)
        result += v * (_BASE ** i)
    return result
