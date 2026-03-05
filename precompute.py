"""
precompute.py
-------------
Offline one-shot script: find the single globally best opening guess across
the ENTIRE vocabulary (all word lengths combined), then save it to
data/first_guesses.json.

Run once before starting the solver:
    python precompute.py

CPU/RAM budget
~~~~~~~~~~~~~~
  _MAX_WORKERS = floor(cpu_count × 0.70)  parallel processes
  numpy threads = 1 per worker
  Each task holds at most CHUNK_SIZE × |answer_block| × 4 bytes in RAM.
"""

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

DATA_FILE  = os.path.join(os.path.dirname(__file__), "data", "words.json")
OUT_FILE   = os.path.join(os.path.dirname(__file__), "data", "first_guesses.json")
CHUNK_SIZE = 300


# ──────────────────────────────────────────────────────────────────────────────
# Vectorised helpers
# ──────────────────────────────────────────────────────────────────────────────

def words_to_array(words: list[str], pad_to: int) -> np.ndarray:
    """(V, pad_to) int8 array, right-padded with 0."""
    arr = np.zeros((len(words), pad_to), dtype=np.int8)
    for i, w in enumerate(words):
        enc = w.encode()
        arr[i, :len(enc)] = np.frombuffer(enc, dtype=np.int8)
    return arr


def _patterns_chunk(
    guesses_chunk: np.ndarray,   # (C, L1) int8
    answers_arr:   np.ndarray,   # (A, L2) int8
    L1: int,
    L2: int,
) -> np.ndarray:                 # (C, A) int32  base-5 encoded
    """
    Vectorised Letroso pattern computation.
    States 3 (ADJACENT) and 4 (BORDER) are GREEN refinements.
    """
    C    = guesses_chunk.shape[0]
    A    = answers_arr.shape[0]
    min_L = min(L1, L2)

    # ── GREEN ────────────────────────────────────────────────────────────────
    green = np.zeros((C, A, L1), dtype=bool)
    if min_L > 0:
        green[:, :, :min_L] = (
            guesses_chunk[:, np.newaxis, :min_L] ==
            answers_arr  [np.newaxis, :,  :min_L]
        )

    colour = np.zeros((C, A, L1), dtype=np.int8)
    colour[green] = 2  # GREEN

    # ── YELLOW ───────────────────────────────────────────────────────────────
    for ch_val in np.unique(answers_arr[:, :L2]):
        if ch_val == 0:
            continue
        g_mask = (~green) & (guesses_chunk[:, np.newaxis, :] == ch_val)
        a_mask_ng = (answers_arr[np.newaxis, :, :L2] == ch_val)
        if L2 <= L1:
            a_mask_ng = a_mask_ng & ~green[:, :, :L2]
        avail = a_mask_ng.view(np.uint8).cumsum(axis=2)[:, :, -1:]
        used  = g_mask.view(np.uint8).cumsum(axis=2)
        colour[g_mask & (used <= avail)] = 1  # YELLOW

    # ── Upgrade GREEN → ADJACENT (3) or BORDER (4) ──────────────────────────
    is_green = (colour == 2)  # raw GREEN positions

    # Adjacency: any GREEN with an adjacent GREEN gets ADJACENT (3)
    has_adj = np.zeros_like(is_green)
    if L1 > 1:
        has_adj[:, :, :-1] |= is_green[:, :, 1:]   # right neighbour green
        has_adj[:, :, 1:]  |= is_green[:, :, :-1]   # left  neighbour green
    adj_mask = is_green & has_adj
    colour[adj_mask] = 3

    # Border: GREEN at position 0 (answer start) or L2-1 (answer end)
    border_mask = is_green & ~has_adj  # only if not already ADJACENT
    # Position 0 is always a border of the answer
    if L1 > 0:
        colour[:, :, 0] = np.where(border_mask[:, :, 0], 4, colour[:, :, 0])
    # Position L2-1 is the end of the answer (if within guess bounds)
    if 0 < L2 - 1 < L1:
        colour[:, :, L2 - 1] = np.where(
            border_mask[:, :, L2 - 1], 4, colour[:, :, L2 - 1]
        )

    # ── Encode base-5 ────────────────────────────────────────────────────────
    powers = np.array([5 ** l for l in range(L1)], dtype=np.int32)
    return (colour.astype(np.int32) * powers).sum(axis=2)


# ──────────────────────────────────────────────────────────────────────────────
# Chunk worker
# ──────────────────────────────────────────────────────────────────────────────

def _chunk_worker(args: tuple) -> tuple[str, float, int]:
    """
    Score a chunk of guess words against ALL answer blocks.
    Returns (best_word, best_entropy, n_words_in_chunk).
    """
    chunk_words, guess_arr, L_guess, answer_blocks = args
    C = len(chunk_words)

    # Accumulate weighted pattern buckets per guess using numpy
    # We collect all patterns + weights, then bincount per guess
    all_pats:    list[np.ndarray] = []
    all_weights: list[np.ndarray] = []

    for answers_arr, L_ans, w_block in answer_blocks:
        pats = _patterns_chunk(guess_arr, answers_arr, L_guess, L_ans)  # (C, A)
        all_pats.append(pats)
        all_weights.append(w_block)

    # Concatenate across all answer blocks: (C, total_A)
    full_pats = np.concatenate(all_pats, axis=1)     # (C, total_A) int32
    full_w    = np.concatenate(all_weights)           # (total_A,) float64
    max_pat   = int(full_pats.max()) + 1

    best_word = chunk_words[0]
    best_ent  = -1.0

    for gi, gword in enumerate(chunk_words):
        row    = full_pats[gi]
        bucket = np.bincount(row, weights=full_w, minlength=max_pat)
        probs  = bucket[bucket > 0]
        total  = probs.sum()
        if total == 0:
            continue
        probs  = probs / total
        ent    = -float(np.sum(probs * np.log2(probs)))
        if ent > best_ent:
            best_ent  = ent
            best_word = gword

    return best_word, best_ent, C


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    if not os.path.exists(DATA_FILE):
        print(f"ERROR: {DATA_FILE} not found.  Run download_words.py first.")
        raise SystemExit(1)

    with open(DATA_FILE, encoding="utf-8") as fh:
        dataset: dict[str, float] = json.load(fh)

    all_words  = list(dataset.keys())
    all_freqs  = list(dataset.values())
    total_freq = sum(all_freqs)
    norm_w     = {w: f / total_freq for w, f in zip(all_words, all_freqs)}

    by_length: dict[int, list[str]] = {}
    for word in all_words:
        by_length.setdefault(len(word), []).append(word)

    lengths = sorted(by_length.keys())

    # Pre-build answer blocks (grouped by length for vectorised computation)
    answer_blocks: list[tuple[np.ndarray, int, np.ndarray]] = []
    for L_ans in lengths:
        ans_words = by_length[L_ans]
        arr       = words_to_array(ans_words, L_ans)
        w_arr     = np.array([norm_w[w] for w in ans_words], dtype=np.float64)
        answer_blocks.append((arr, L_ans, w_arr))

    total_words = len(all_words)
    print(f"Vocabulary : {total_words:,} words  |  lengths: {lengths}")
    print(f"Workers    : {_MAX_WORKERS} / {_CPU_COUNT} cores (70 %)  |  chunk: {CHUNK_SIZE}\n")

    # Build tasks: one per chunk of guess words
    tasks: list[tuple] = []
    for L_guess in lengths:
        g_words    = by_length[L_guess]
        g_arr_full = words_to_array(g_words, L_guess)
        for chunk_start in range(0, len(g_words), CHUNK_SIZE):
            chunk_end   = min(chunk_start + CHUNK_SIZE, len(g_words))
            chunk_words = g_words[chunk_start:chunk_end]
            guess_arr   = g_arr_full[chunk_start:chunk_end]
            tasks.append((chunk_words, guess_arr, L_guess, answer_blocks))

    global_best_word = all_words[0]
    global_best_ent  = -1.0
    processed        = 0
    last_milestone   = -1
    t0               = time.time()

    n_tasks = len(tasks)
    print(f"Tasks      : {n_tasks} chunks to process\n")

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
