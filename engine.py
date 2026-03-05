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
    ans_pos = {}
    ans_used = [False] * L2

    # ── LCS computation ─────────────────────────────
    dp = [[0]*(L2+1) for _ in range(L1+1)]

    for i in range(L1-1, -1, -1):
        for j in range(L2-1, -1, -1):
            if guess[i] == answer[j]:
                dp[i][j] = 1 + dp[i+1][j+1]
            else:
                dp[i][j] = max(dp[i+1][j], dp[i][j+1])

    i = j = 0
    while i < L1 and j < L2:
        if guess[i] == answer[j]:
            state[i] = GREEN
            ans_pos[i] = j
            ans_used[j] = True
            available[guess[i]] -= 1
            i += 1
            j += 1
        elif dp[i+1][j] >= dp[i][j+1]:
            i += 1
        else:
            j += 1

    # ── Step 3: right-to-left yellow assignment ─────
    for i in range(L1-1, -1, -1):
        if state[i] == GREEN:
            continue
        ch = guess[i]
        if available.get(ch, 0) > 0:
            state[i] = YELLOW
            available[ch] -= 1

    # ── Step 4: border upgrade ──────────────────────
    for i, j in ans_pos.items():
        if j == 0 or j == L2-1:
            state[i] = BORDER

    # ── Step 5: concatenation ───────────────────────
    concat = [False] * L1
    greens = sorted(ans_pos.keys())

    for k in range(len(greens)-1):
        gi = greens[k]
        gj = greens[k+1]

        if gj == gi + 1:
            if ans_pos[gj] == ans_pos[gi] + 1:
                concat[gi] = True

    # ── Encode pattern ──────────────────────────────
    result = 0
    for i in range(L1):
        v = state[i]*2 + (1 if concat[i] else 0)
        result += v * (_BASE ** i)

    return result

# def matches_feedback(
#     guess: str,
#     candidate: str,
#     states: list[int],
#     concats: list[bool],
# ) -> bool:
#     """
#     Return True if candidate word is compatible with manual feedback (states + concats).
#     Does NOT require exact pattern_int match from get_pattern.
#     """
#     # Step 0: sanity
#     if len(guess) != len(candidate) or len(states) != len(guess):
#         return False

#     # Count available letters in candidate
#     available = {}
#     for ch in candidate:
#         available[ch] = available.get(ch, 0) + 1

#     # Step 1: check greens / borders
#     green_indices = []
#     for i, s in enumerate(states):
#         gch = guess[i]
#         if s >= GREEN:
#             # Must appear somewhere in candidate, respecting order for green
#             found = False
#             for j, cch in enumerate(candidate):
#                 if cch == gch and (j not in green_indices):
#                     green_indices.append(j)
#                     available[cch] -= 1
#                     found = True
#                     break
#             if not found:
#                 return False

#     # Step 2: check concats (adjacent greens)
#     for i, flag in enumerate(concats[:-1]):  # skip last
#         if flag:
#             g1 = guess[i]
#             g2 = guess[i+1]
#             # positions in candidate
#             pos1 = next((p for idx, p in enumerate(green_indices) if guess[idx]==g1), None)
#             pos2 = next((p for idx, p in enumerate(green_indices) if guess[idx]==g2), None)
#             if pos1 is None or pos2 is None:
#                 return False
#             if abs(pos2 - pos1) != 1:
#                 return False

#     # Step 3: check yellows
#     for i, s in enumerate(states):
#         if s == YELLOW:
#             if gch := guess[i]:
#                 if available.get(gch, 0) <= 0:
#                     return False
#                 # Cannot appear at same position as guess[i] if green
#                 if candidate[i] == gch:
#                     return False
#                 available[gch] -= 1

#     # Step 4: check blacks
#     for i, s in enumerate(states):
#         if s == BLACK:
#             bch = guess[i]
#             if available.get(bch, 0) > 0:
#                 return False

#     return True

# def decode_pattern(pattern_int: int, length: int) -> tuple[list[int], list[bool]]:
#     """Decode a pattern int into (states, concat_rights) for debugging."""
#     states = []
#     concats = []
#     for _ in range(length):
#         v = pattern_int % _BASE
#         states.append(v // 2)
#         concats.append(bool(v % 2))
#         pattern_int //= _BASE
#     return states, concats


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
    """True if every position is GREEN or BORDER (state >= 2)."""
    for _ in range(length):
        v = pattern_int % _BASE
        if v // 2 < GREEN:
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
