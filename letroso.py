"""
Usage
-----
  python letroso.py                        # run built-in tests
  python letroso.py <answer> <guess>       # print feedback string
-----------
  B  –  Letter is NOT in the answer at all.
  Y  –  Letter IS in the answer, but its relative position (vs. other
         confirmed letters) is wrong.
  G  –  Letter is in the answer and in the correct relative position.
  P  –  Same as G, but this letter is the FIRST or LAST letter of the
         answer AND the first or last letter of the guess.
  O  –  "Concat" connector: inserted AFTER a G or P token to signal
         that the preceding green letter and the next green letter are
         directly adjacent in BOTH the guess and the answer (no gap
         between them in either word).
Algorithm
---------
1.  Find the matching (subsequence) between guess and answer that
    (a) maximises the number of matched letters  (longest common
        subsequence), and then
    (b) among all equally long matchings, maximises the number of
        adjacent matched pairs (pairs that are consecutive in both
        the guess and the answer simultaneously).
    This is solved with a DP over (guess_idx, answer_idx,
    prev_answer_idx, prev_guess_idx).

2.  For every guess letter:
      •  Matched → G or P (see P rule above).
      •  Unmatched, letter still available in answer → Y.
      •  Otherwise → B.

3.  After each G/P token: append O if the current matched letter and
    the NEXT matched letter are adjacent in BOTH words.
"""

from __future__ import annotations
from functools import lru_cache
from collections import Counter


def score(answer: str, guess: str) -> str:
    """Return the Letroso feedback string for *guess* against *answer*.

    Parameters
    ----------
    answer : str   The correct target word.
    guess  : str   The candidate word being evaluated.

    Returns
    -------
    str  Feedback string composed of B, Y, G, P, O characters.
    """
    answer = answer.lower()
    guess  = guess.lower()
    na, ng = len(answer), len(guess)

    # ------------------------------------------------------------------
    # Step 1 – DP to find best (matches, adjacent_pairs) achievable.
    #
    # dp(gi, ai, prev_ai, prev_gi)  →  (max_matches, max_adj_pairs)
    #   gi      : current index into guess
    #   ai      : current index into answer (must only use answer[ai:])
    #   prev_ai : answer index of the last matched pair  (-2 = none yet)
    #   prev_gi : guess  index of the last matched pair  (-2 = none yet)
    # ------------------------------------------------------------------
    @lru_cache(maxsize=None)
    def dp(gi: int, ai: int, prev_ai: int, prev_gi: int):
        if gi == ng or ai == na:
            return (0, 0)

        # Option A: skip guess[gi]
        best = dp(gi + 1, ai, prev_ai, prev_gi)

        # Option B: match guess[gi] to some answer[ai2] (ai2 >= ai)
        for ai2 in range(ai, na):
            if answer[ai2] == guess[gi]:
                is_adj = (
                    prev_ai >= 0
                    and ai2 == prev_ai + 1
                    and gi  == prev_gi + 1
                )
                rest      = dp(gi + 1, ai2 + 1, ai2, gi)
                candidate = (1 + rest[0], (1 if is_adj else 0) + rest[1])
                if candidate > best:
                    best = candidate

        return best

    # ------------------------------------------------------------------
    # Step 2 – Greedy backtrack to recover the actual matched pairs.
    # ------------------------------------------------------------------
    target_m, target_a = dp(0, 0, -2, -2)
    matched_pairs: list[tuple[int, int]] = []   # (answer_idx, guess_idx)

    gi, ai  = 0, 0
    prev_ai = -2
    prev_gi = -2
    tm, ta  = target_m, target_a

    while gi < ng and ai < na and tm > 0:
        matched = False
        for ai2 in range(ai, na):
            if answer[ai2] == guess[gi]:
                is_adj = (
                    prev_ai >= 0
                    and ai2 == prev_ai + 1
                    and gi  == prev_gi + 1
                )
                rest = dp(gi + 1, ai2 + 1, ai2, gi)
                m    = 1 + rest[0]
                a    = (1 if is_adj else 0) + rest[1]
                if (m, a) == (tm, ta):
                    matched_pairs.append((ai2, gi))
                    prev_ai, prev_gi = ai2, gi
                    ai  = ai2 + 1
                    tm, ta = rest
                    matched = True
                    break
        if not matched:
            gi += 1
            continue
        gi += 1

    dp.cache_clear()

    # ------------------------------------------------------------------
    # Step 3 – Build lookup structures.
    # ------------------------------------------------------------------
    matched_gi = {gi  for _, gi  in matched_pairs}
    matched_ai = {ai  for ai, _  in matched_pairs}
    gi_to_ai   = {gi: ai for ai, gi in matched_pairs}

    # Sorted by answer position (they already are, but be explicit)
    ap          = sorted(matched_pairs)
    next_ai_map = {ap[i][0]: ap[i + 1][0] for i in range(len(ap) - 1)}
    next_gi_map = {ap[i][1]: ap[i + 1][1] for i in range(len(ap) - 1)}

    # Unmatched answer letters available for Y tokens
    avail = Counter(answer[ai] for ai in range(na) if ai not in matched_ai)

    # ------------------------------------------------------------------
    # Step 4 – Emit tokens.
    # ------------------------------------------------------------------
    result: list[str] = []

    for gi, ch in enumerate(guess):
        if gi in matched_gi:
            ai = gi_to_ai[gi]

            # P: letter is at an answer boundary AND a guess boundary
            at_ans_boundary = (ai == 0 or ai == na - 1)
            at_gue_boundary = (gi == 0 or gi == ng - 1)
            tok = "P" if (at_ans_boundary and at_gue_boundary) else "G"
            result.append(tok)

            # O: next matched pair is adjacent in BOTH words
            if (
                ai in next_ai_map
                and next_ai_map[ai] == ai + 1
                and next_gi_map.get(gi) == gi + 1
            ):
                result.append("O")

        else:
            if avail.get(ch, 0) > 0:
                avail[ch] -= 1
                result.append("Y")
            else:
                result.append("B")

    return "".join(result)

if __name__ == "__main__":
    import sys

    if len(sys.argv) == 1:
        print("Running built-in tests …\n")
        run_tests()

    elif len(sys.argv) == 3:
        fb = score(sys.argv[1], sys.argv[2])
        print(fb)

    else:
        print("Usage:")
        print("  python letroso.py                   # run tests")
        print("  python letroso.py <answer> <guess>  # score one guess")
        sys.exit(1)