"""
solver.py
---------
Interactive CLI solver for Letroso.

Usage:
    python solver.py

Workflow:
    1. Solver suggests a word
    2. You enter it in Letroso and type the feedback (e.g. "1224")
    3. Solver narrows candidates and suggests the next word
    4. Repeat until solved

Feedback codes:
    0 / B  = grey   (letter not in word)
    1 / Y  = yellow (letter in word, wrong position)
    2 / G  = green  (correct position, isolated)
    3 / A  = green  (connected — adjacent to another green)
    4 / P  = green  (border — rounded corners, first/last letter of answer)

The solver does not know the answer's length — it works across the entire
3–10 letter vocabulary and deduces the length from feedback.
"""

import json
import os
import sys
import time

from engine import (
    filter_candidates,
    is_win_pattern,
    parse_pattern,
    rank_guesses,
)

BASE_DIR           = os.path.dirname(__file__)
DATA_FILE          = os.path.join(BASE_DIR, "data", "words.json")
FIRST_GUESSES_FILE = os.path.join(BASE_DIR, "data", "first_guesses.json")


def load_data() -> tuple[list[str], dict[str, float]]:
    if not os.path.exists(DATA_FILE):
        print(f"ERROR: {DATA_FILE} not found.  Run download_words.py first.")
        sys.exit(1)

    with open(DATA_FILE, encoding="utf-8") as fh:
        dataset: dict[str, float] = json.load(fh)

    total = sum(dataset.values())
    weights = {w: f / total for w, f in dataset.items()}
    vocab   = list(weights.keys())
    return vocab, weights


def load_first_guess() -> tuple[str, float] | None:
    if os.path.exists(FIRST_GUESSES_FILE):
        with open(FIRST_GUESSES_FILE, encoding="utf-8") as fh:
            data = json.load(fh)
        return data["word"], data["entropy"]
    return None


def _progress(done: int, total: int) -> None:
    print(f"\r  Computing best guess... ({done:,}/{total:,})", end="", flush=True)


def main() -> None:
    print("Loading vocabulary...")
    vocab, weights = load_data()
    first = load_first_guess()

    candidates = list(vocab)
    turn = 0

    print(f"Vocabulary: {len(vocab):,} words (lengths 3–10)")
    print()
    print("Feedback codes:")
    print("  0/B=grey  1/Y=yellow  2/G=green  3/A=connected  4/P=border")
    print()

    while True:
        turn += 1

        # ── Pick guess ───────────────────────────────────────────────────────
        if turn == 1 and first:
            guess, entropy = first
            print(f"Turn {turn}  |  candidates: {len(candidates):,}")
            print(f"  Guess: {guess}  (precomputed, {entropy:.4f} bits)")
        elif len(candidates) == 1:
            guess = candidates[0]
            print(f"Turn {turn}  |  candidates: 1")
            print(f"  Answer: {guess}")
            break
        elif len(candidates) == 0:
            print("No candidates remain — feedback may have been incorrect.")
            break
        else:
            print(f"Turn {turn}  |  candidates: {len(candidates):,}")
            t0 = time.time()
            ranked = rank_guesses(candidates, vocab, weights, top_n=5,
                                  progress_fn=_progress)
            elapsed = time.time() - t0
            guess, entropy = ranked[0]
            print(f"\r  Guess: {guess}  ({entropy:.4f} bits)  [{elapsed:.1f}s]")
            if len(ranked) > 1:
                alts = ", ".join(f"{w} ({e:.2f})" for w, e in ranked[1:])
                print(f"  Also good: {alts}")

        # ── Get feedback ─────────────────────────────────────────────────────
        while True:
            try:
                raw = input(f"\n  Feedback for '{guess}': ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye!")
                return

            if raw.lower() in ("q", "quit", "exit"):
                print("Bye!")
                return

            pattern_int = parse_pattern(raw, len(guess))
            if pattern_int is None:
                print(f"  Invalid.  Enter {len(guess)} codes: 0-4 or B/Y/G/A/P")
                continue

            # ── Win check ────────────────────────────────────────────────────
            if is_win_pattern(pattern_int, len(guess)):
                print(f"\n  Solved in {turn} turn{'s' if turn > 1 else ''}!  Answer: {guess}")
                return

            # ── Filter ───────────────────────────────────────────────────────
            old_count = len(candidates)
            new_candidates = filter_candidates(candidates, guess, pattern_int)

            if not new_candidates:
                print(f"  No candidates match that feedback.  Please re-enter.")
                continue

            candidates = new_candidates
            print(f"  {old_count:,} → {len(candidates):,} candidates")
            break


if __name__ == "__main__":
    main()
