import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

# Limit numpy internal threading BEFORE importing numpy
_CPU_COUNT   = os.cpu_count() or 1
_MAX_WORKERS = max(1, int(_CPU_COUNT * 0.70))

for _env in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_env] = "1"

import numpy as np

try:
    os.nice(10)
except (AttributeError, PermissionError):
    pass

DATA_FILE  = os.path.join(os.path.dirname(__file__), "..","data", "words.json")
OUT_FILE   = os.path.join(os.path.dirname(__file__), "..","data", "first_guesses.json")
CHUNK_SIZE = 300

def _chunk_worker(args: tuple) -> tuple[str, float, int]:
    """
    Score a chunk of guess words against ALL answer words.
    Returns (best_word, best_entropy, n_words_in_chunk).

    Uses engine.get_pattern directly (subsequence-based matching).
    We bypass the lru_cache wrapper since every (guess, answer) pair
    is unique in precompute and caching wastes memory.
    """
    chunk_words, all_answers, all_weights_arr = args

    # Import inside worker so each process gets its own copy
    from engine import get_pattern

    # Use the unwrapped function to avoid lru_cache overhead
    _get_pattern = get_pattern.__wrapped__

    C = len(chunk_words)
    A = len(all_answers)

    best_word = chunk_words[0]
    best_ent  = -1.0

    # Pre-allocate pattern array for bincount
    pats = np.empty(A, dtype=np.int32)

    for gi, guess in enumerate(chunk_words):
        # Compute patterns for this guess against all answers
        for ai, answer in enumerate(all_answers):
            pats[ai] = _get_pattern(guess, answer)

        # Entropy via bincount
        max_pat = int(pats.max()) + 1
        bucket  = np.bincount(pats, weights=all_weights_arr, minlength=max_pat)
        probs   = bucket[bucket > 0]
        total   = probs.sum()
        if total == 0:
            continue
        probs = probs / total
        ent   = -float(np.sum(probs * np.log2(probs)))
        if ent > best_ent:
            best_ent  = ent
            best_word = guess

    return best_word, best_ent, C

def main() -> None:
    if not os.path.exists(DATA_FILE):
        print(f"ERROR: {DATA_FILE} not found.  Run download_words.py first.")
        raise SystemExit(1)

    with open(DATA_FILE, encoding="utf-8") as fh:
        dataset: dict[str, float] = json.load(fh)

    all_words  = list(dataset.keys())
    all_freqs  = list(dataset.values())
    total_freq = sum(all_freqs)
    norm_w     = np.array([f / total_freq for f in all_freqs], dtype=np.float64)

    total_words = len(all_words)
    by_length: dict[int, int] = {}
    for w in all_words:
        by_length[len(w)] = by_length.get(len(w), 0) + 1
    lengths = sorted(by_length.keys())

    print(f"Vocabulary : {total_words:,} words  |  lengths: {lengths}")
    print(f"Workers    : {_MAX_WORKERS} / {_CPU_COUNT} cores (70 %)  |  chunk: {CHUNK_SIZE}\n")

    # Build tasks: one per chunk of guess words
    tasks: list[tuple] = []
    for chunk_start in range(0, total_words, CHUNK_SIZE):
        chunk_end   = min(chunk_start + CHUNK_SIZE, total_words)
        chunk_words = all_words[chunk_start:chunk_end]
        tasks.append((chunk_words, all_words, norm_w))

    n_tasks = len(tasks)
    print(f"Tasks      : {n_tasks} chunks to process\n")

    global_best_word = all_words[0]
    global_best_ent  = -1.0
    processed        = 0
    t0               = time.time()

    with ProcessPoolExecutor(max_workers=_MAX_WORKERS) as pool:
        futures = [pool.submit(_chunk_worker, task) for task in tasks]

        for future in as_completed(futures):
            word, ent, n = future.result()
            processed += n

            if ent > global_best_ent:
                global_best_ent  = ent
                global_best_word = word

            pct     = processed / total_words * 100
            elapsed = time.time() - t0
            print(
                f"\r  {pct:5.1f}%  ({processed:,}/{total_words:,})  "
                f"best: {global_best_word!r} ({global_best_ent:.4f} bits)  "
                f"[{elapsed:.0f}s]",
                end="", flush=True,
            )

        print()  # newline after progress

    elapsed = time.time() - t0
    print(f"\n  100%  done in {elapsed:.1f}s")
    print(f"  Global best opener: {global_best_word!r}  entropy = {global_best_ent:.4f} bits")

    result = {"word": global_best_word, "entropy": round(global_best_ent, 6)}
    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    with open(OUT_FILE, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
    print(f"  Saved → {OUT_FILE}")


if __name__ == "__main__":
    main()
