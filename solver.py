import json
import os
import sys
import time

from engine import (
    decode_pattern,
    filter_candidates,
    is_win_pattern,
    parse_pattern,
    rank_guesses,
    BLACK, YELLOW, GREEN, BORDER,
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


def _validate_feedback(
    guess: str,
    states: list[int],
    known_absent: set[str],
    known_present: set[str],
) -> str | None:
    """
    Validate feedback against cross-turn knowledge.
    Returns an error message, or None if valid.
    """
    for i, s in enumerate(states):
        ch = guess[i]
        if s >= GREEN and ch in known_absent:
            return (
                f"  '{ch}' was BLACK in a previous turn but marked as "
                f"{'green' if s == GREEN else 'border'} now.  Please re-enter."
            )
        if s == BLACK and ch in known_present:
            # This is OK — could be an excess duplicate.  Don't block.
            pass
    return None


def _play_game(vocab: list[str], weights: dict[str, float], first) -> bool:
    """Play one game. Returns False if the user wants to quit."""
    candidates = list(vocab)
    turn = 0

    # Cross-turn memory
    known_absent:  set[str] = set()
    known_present: set[str] = set()
    letter_ever_nonblack: set[str] = set()

    while True:
        turn += 1

        # Pick guess
        if turn == 1 and first:
            guess, entropy = first
            print(f"Turn {turn}  |  candidates: {len(candidates):,}")
            print(f"  Guess: {guess}  (precomputed, {entropy:.4f} bits)")
        elif len(candidates) == 1:
            guess = candidates[0]
            print(f"Turn {turn}  |  candidates: 1")
            print(f"  Answer: {guess}")
            return True
        elif len(candidates) == 0:
            print("No candidates remain — feedback may have been incorrect.")
            return True
        else:
            print(f"Turn {turn}  |  candidates: {len(candidates):,}")
            t0 = time.time()
            ranked = rank_guesses(candidates, vocab, weights, top_n=5,
                                  progress_fn=_progress)
            elapsed = time.time() - t0
            guess, entropy = ranked[0]
            print(f"\r  Guess: {guess}  ({entropy:.4f} bits)  [{elapsed:.1f}s]")
            if len(ranked) > 1:
                alts = ", ".join(f"{w} ({e:.4f})" for w, e in ranked[1:])
                print(f"  Also good: {alts}")

        # Get feedback
        while True:
            try:
                raw = input(f"\n  Feedback for '{guess}': ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye!")
                return False

            if raw.lower() in ("q", "quit", "exit"):
                print("Bye!")
                return False

            pattern_int = parse_pattern(raw, len(guess))
            if pattern_int is None:
                print(f"  Invalid format.  Use B/Y/G/P (with O between greens).")
                continue

            # Decode for validation
            states, concats = decode_pattern(pattern_int, len(guess))

            # Cross-turn validation
            err = _validate_feedback(guess, states, known_absent, known_present)
            if err:
                print(err)
                continue

            # Win check
            if is_win_pattern(pattern_int, len(guess)):
                print(f"\n  Solved in {turn} turn{'s' if turn > 1 else ''}!  Answer: {guess}")
                return True

            # Filter candidates
            old_count = len(candidates)
            new_candidates = filter_candidates(candidates, guess, pattern_int)

            if not new_candidates:
                print(f"  No candidates match that feedback.  Please re-enter.")
                continue

            candidates = new_candidates
            print(f"  {old_count:,} → {len(candidates):,} candidates")

            # Update cross-turn memory
            for i, s in enumerate(states):
                ch = guess[i]
                if s >= GREEN or s == YELLOW:
                    letter_ever_nonblack.add(ch)
                    known_present.add(ch)

            # A letter is confirmed absent only if EVERY occurrence across
            # ALL turns was BLACK (and it was never non-black)
            for i, s in enumerate(states):
                ch = guess[i]
                if s == BLACK and ch not in letter_ever_nonblack:
                    known_absent.add(ch)

            break


def main() -> None:
    print("Loading vocabulary...")
    vocab, weights = load_data()
    first = load_first_guess()

    print(f"Vocabulary: {len(vocab):,} words (lengths 3–10)")
    print()
    print("Feedback: B=black Y=yellow G=green P=border O=concat")
    print("  e.g. BGYGB or BGOGOGB  |  q to quit")
    print()

    game = 0
    while True:
        game += 1
        print(f"{'─'*40}")
        print(f"Game {game}")
        print(f"{'─'*40}")
        keep_going = _play_game(vocab, weights, first)
        if not keep_going:
            break
        print()
        try:
            again = input("Play again? [Y/n]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if again in ("n", "no", "q", "quit"):
            print("Bye!")
            break
        print()


if __name__ == "__main__":
    main()
