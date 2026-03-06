import math
from collections import defaultdict
from functools import lru_cache
from typing import Optional

BLACK  = 0
YELLOW = 1
GREEN  = 2
BORDER = 3

_BASE = 8   # state*2 + concat_right  →  values 0–7

@lru_cache(maxsize=8_000_000)
def get_pattern(guess: str, answer: str) -> int:
    L1, L2 = len(guess), len(answer)

    available = {}
    for ch in answer:
        available[ch] = available.get(ch, 0) + 1

    state = [BLACK] * L1
    ans_pos = {}  # guess index → answer index for green matches
    ans_used = [False] * L2

    # ── Step 1: find the GREEN subsequence ───────────────────────────────────
    # Priority order for choosing among multiple LCS of equal length:
    #   1. Most consecutive pairs (adjacent in BOTH guess and answer simultaneously)
    #   2. Earliest start in the sequence
    #
    # dp[i][j] = (L, C): best (LCS-length, consecutive-pairs) from position (i,j) onward.
    # is_m[i][j]: True if the optimal alignment at (i,j) starts by matching guess[i]→answer[j].
    # Boundary: dp[L1][*] = dp[*][L2] = (0,0), is_m boundary = False.

    dp   = [[(0, 0)] * (L2 + 1) for _ in range(L1 + 1)]
    is_m = [[False]   * (L2 + 1) for _ in range(L1 + 1)]

    for i in range(L1 - 1, -1, -1):
        for j in range(L2 - 1, -1, -1):
            skip_i = dp[i + 1][j]   # skip guess[i]
            skip_j = dp[i][j + 1]   # skip answer[j]
            # When skip_i == skip_j, prefer skip_j: keeps guess[i] available
            # for an earlier match, satisfying the "starts earlier" tiebreak.
            best_skip = skip_i if skip_i > skip_j else skip_j

            if guess[i] == answer[j]:
                nL, nC = dp[i + 1][j + 1]
                # +1 consecutive bonus if the very next optimal step is also
                # an immediate (i+1 → j+1) match.
                bonus     = 1 if is_m[i + 1][j + 1] else 0
                match_val = (1 + nL, nC + bonus)
                # Prefer match when tied with skip (achieves earliest start).
                if match_val >= best_skip:
                    dp[i][j]   = match_val
                    is_m[i][j] = True
                else:
                    dp[i][j]   = best_skip
            else:
                dp[i][j] = best_skip

    # Traceback: follow is_m flags, tie-break toward earlier start.
    i = j = 0
    while i < L1 and j < L2:
        if is_m[i][j]:
            state[i]   = GREEN
            ans_pos[i] = j
            ans_used[j] = True
            available[guess[i]] -= 1
            i += 1
            j += 1
        else:
            # When equal, prefer skip_j (same tiebreak as DP fill).
            if dp[i + 1][j] > dp[i][j + 1]:
                i += 1
            else:
                j += 1

    # ── Step 2: non-green positions → YELLOW or BLACK (right-to-left) ────────
    # Rightmost non-green positions get YELLOW first; leftmost excess → BLACK.
    # ("excess copies get BLACK from the left")
    for i in range(L1 - 1, -1, -1):
        if state[i] == GREEN:
            continue
        ch = guess[i]
        if available.get(ch, 0) > 0:
            state[i] = YELLOW
            available[ch] -= 1

    # ── Step 3: upgrade GREEN → BORDER at answer boundaries ──────────────────
    for i, j in ans_pos.items():
        if j == 0 or j == L2 - 1:
            state[i] = BORDER

    # ── Step 4: concatenation between adjacent GREEN/BORDER pairs ─────────────
    concat     = [False] * L1
    green_list = sorted(ans_pos.keys())
    for k in range(len(green_list) - 1):
        gi = green_list[k]
        gj = green_list[k + 1]
        if gj == gi + 1 and ans_pos[gj] == ans_pos[gi] + 1:
            concat[gi] = True

    # ── Encode ────────────────────────────────────────────────────────────────
    result = 0
    for i in range(L1):
        v = state[i] * 2 + (1 if concat[i] else 0)
        result += v * (_BASE ** i)
    return result

def matches_feedback(
    guess: str,
    candidate: str,
    states: list[int],
    concats: list[bool],
) -> bool:
    """
    Return True if candidate is compatible with the given feedback.

    Steps (in order, each consuming from the available letter pool):
      1. GREEN/BORDER letters must appear as a left-to-right subsequence in
         candidate.  BORDER letters must land at position 0 or len-1 of
         candidate; CONCAT between two adjacent greens means their candidate
         positions must be consecutive.
      2. YELLOW letters must be present in the remaining pool (after greens
         consumed their share).  In Letroso, yellow only means "present but
         out of order relative to the green subsequence" — no position check.
      3. BLACK letters must leave zero remaining copies in the pool (after
         both greens and yellows have consumed theirs).  This correctly caps
         the repetition count.
    """
    L2 = len(candidate)

    # Available letter counts in candidate
    available: dict[str, int] = {}
    for ch in candidate:
        available[ch] = available.get(ch, 0) + 1

    # ── Step 1: GREEN / BORDER – forward subsequence scan ─────────────────────
    green_cand_pos: list[int] = []   # candidate index for each green, in order
    green_guess_idx: list[int] = []  # corresponding guess index
    ptr = 0

    for i, s in enumerate(states):
        if s < GREEN:
            continue
        ch = guess[i]
        # find next occurrence of ch in candidate from ptr
        j = ptr
        while j < L2 and candidate[j] != ch:
            j += 1
        if j == L2:
            return False   # required letter not found in subsequence order
        green_cand_pos.append(j)
        green_guess_idx.append(i)
        available[ch] -= 1
        ptr = j + 1

    # Verify BORDER positions
    for k, i in enumerate(green_guess_idx):
        pos = green_cand_pos[k]
        if states[i] == BORDER:
            if pos != 0 and pos != L2 - 1:
                return False

    # Verify CONCAT: concats[i] means guess[i] (green) and the next green
    # in the guess must occupy consecutive positions in candidate.
    for k in range(len(green_guess_idx) - 1):
        gi = green_guess_idx[k]
        if concats[gi]:
            if green_cand_pos[k + 1] != green_cand_pos[k] + 1:
                return False

    # ── Step 2: YELLOW – letter must still be in the pool ─────────────────────
    for i, s in enumerate(states):
        if s != YELLOW:
            continue
        ch = guess[i]
        if available.get(ch, 0) <= 0:
            return False
        available[ch] -= 1

    # ── Step 3: BLACK – no remaining copies allowed ────────────────────────────
    # Iterating over every BLACK position is fine: if a letter appears multiple
    # times as BLACK and the pool is already 0, each check passes harmlessly.
    for i, s in enumerate(states):
        if s != BLACK:
            continue
        if available.get(guess[i], 0) > 0:
            return False

    return True

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
    True only for the exact win pattern  PO(GO)^NP:
      - position 0        : BORDER, concat = True
      - positions 1..L-2  : GREEN,  concat = True
      - position L-1      : BORDER, concat = True  (both borders part of the chain)

    The trailing O on the last BORDER is not shown in the string representation
    (pattern_to_str suppresses it), so the string still looks like "POGOP".
    Single-letter edge case: BORDER with concat = True.
    """
    for i in range(length):
        v      = pattern_int % _BASE
        state  = v // 2
        concat = bool(v % 2)
        if i == 0:
            if state != BORDER or not concat:
                return False
        elif i == length - 1:
            if state != BORDER:   # concat not checked: parse_pattern won't set it
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
    # return [w for w in candidates if get_pattern(guess, w) == pattern_int]
    states, concats = decode_pattern(pattern_int, len(guess))
    return [w for w in candidates if matches_feedback(guess, w, states, concats)]

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


def rank_guesses(
    candidates: list[str],
    vocab: list[str],
    weights: dict[str, float],
    top_n: int = 10,
    progress_fn=None,
) -> list[tuple[str, float]]:
    """
    Return top-n guesses by descending entropy.
    When ≤ 20 candidates remain, only rank those.
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
