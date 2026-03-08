import json
import math
import os
import sys
import time
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from concurrent.futures import ProcessPoolExecutor, as_completed
from engine import get_pattern

# Limit numpy internal threading BEFORE importing numpy
_CPU_COUNT   = os.cpu_count() or 1
_MAX_WORKERS = max(1, int(_CPU_COUNT * 0.70))

for _env in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_env] = "1"

try:
    os.nice(10)
except (AttributeError, PermissionError):
    pass

DATA_FILE  = os.path.join(os.path.dirname(__file__), "..","data", "words.json")
OUT_FILE   = os.path.join(os.path.dirname(__file__), "..","data", "first_guesses.json")
CHUNK_SIZE = 50

# Per-worker globals — set once via initializer instead of pickling into every task
_worker_answers: list[str] = []
_worker_weights: np.ndarray = np.empty(0)
_worker_get_pattern = None


def _worker_init(all_answers: list[str], all_weights_arr: np.ndarray) -> None:
    global _worker_answers, _worker_weights, _worker_get_pattern
    _worker_answers = all_answers
    _worker_weights = all_weights_arr
    _worker_get_pattern = get_pattern.__wrapped__


def _chunk_worker(chunk_words: list[str]) -> tuple[str, float, int]:
    """Score a chunk of guess words against ALL answer words (shared via initializer)."""
    all_answers     = _worker_answers
    all_weights_arr = _worker_weights
    _get_pattern    = _worker_get_pattern

    A         = len(all_answers)
    best_word = chunk_words[0]
    best_ent  = -1.0
    pats      = np.empty(A, dtype=np.int32)

    for guess in chunk_words:
        for ai, answer in enumerate(all_answers):
            pats[ai] = _get_pattern(guess, answer)

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

    return best_word, best_ent, len(chunk_words)


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

    guess_words = [w for w in all_words if len(w) == 10]
    n_guesses   = len(guess_words)
    n_tasks     = math.ceil(n_guesses / CHUNK_SIZE)
    chunks      = [guess_words[i:i + CHUNK_SIZE] for i in range(0, n_guesses, CHUNK_SIZE)]

    print(f"Vocabulary : {total_words:,} words  |  lengths: {lengths}")
    print(f"Guesses    : {n_guesses:,} length-10 words")
    print(f"Workers    : {_MAX_WORKERS} / {_CPU_COUNT} cores |  chunk: {CHUNK_SIZE}  |  tasks: {n_tasks}")
    print(f"Initialising workers (sending {total_words:,} words once each)...", flush=True)

    global_best_word = guess_words[0]
    global_best_ent  = -1.0
    processed        = 0
    t0               = time.time()

    with ProcessPoolExecutor(
        max_workers=_MAX_WORKERS,
        initializer=_worker_init,
        initargs=(all_words, norm_w),
    ) as pool:
        print(f"Workers ready — submitting {n_tasks} chunks...\n", flush=True)

        futures = [pool.submit(_chunk_worker, chunk) for chunk in chunks]

        for future in as_completed(futures):
            word, ent, n = future.result()
            processed += n

            if ent > global_best_ent:
                global_best_ent  = ent
                global_best_word = word

            pct     = processed / n_guesses * 100
            elapsed = time.time() - t0
            eta     = (elapsed / processed * (n_guesses - processed)) if processed else 0
            print(
                f"  {pct:5.1f}%  ({processed:,}/{n_guesses:,})  "
                f"best: {global_best_word!r} ({global_best_ent:.4f} bits)  "
                f"[{elapsed:.0f}s  ETA {eta:.0f}s]",
                flush=True,
            )

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
