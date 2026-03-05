"""
precompute_gpu.py
-----------------
GPU-accelerated version of precompute.py using CuPy (CUDA).
Designed for Google Colab (T4 / A100) or any NVIDIA GPU with 8 GB+ VRAM.

Colab setup
~~~~~~~~~~~
    !pip install cupy-cuda12x       # match your CUDA version (11x / 12x)
    # upload data/words.json via the Files panel, then:
    !python precompute_gpu.py

If you hit an OOM error, lower CHUNK_SIZE_GPU.
"""

import json
import math
import os
import sys
import time

import numpy as np

try:
    import cupy as cp
    cp.cuda.runtime.getDeviceCount()
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "CuPy not installed. Run:  !pip install cupy-cuda12x\n"
        "(use cupy-cuda11x if your Colab runtime uses CUDA 11)"
    )
except Exception as exc:
    raise RuntimeError(
        f"CUDA not available or CuPy error: {exc}\n"
        "Make sure you selected a GPU runtime in Colab: Runtime → Change runtime type → T4 GPU"
    )

DATA_FILE = os.path.join(os.path.dirname(__file__), "data", "words.json")
OUT_FILE  = os.path.join(os.path.dirname(__file__), "data", "first_guesses.json")

CHUNK_SIZE_GPU = 2_000


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def words_to_array(words: list[str], pad_to: int) -> np.ndarray:
    arr = np.zeros((len(words), pad_to), dtype=np.int8)
    for i, w in enumerate(words):
        enc = w.encode()
        arr[i, :len(enc)] = np.frombuffer(enc, dtype=np.int8)
    return arr


def _patterns_gpu_chunk(
    guesses_chunk: "cp.ndarray",   # (C, L1) int8
    answers_arr:   "cp.ndarray",   # (A, L2) int8
    L1: int,
    L2: int,
) -> "cp.ndarray":                 # (C, A) int32  base-6 encoded
    C    = guesses_chunk.shape[0]
    A    = answers_arr.shape[0]
    min_L = min(L1, L2)

    # ── GREEN ────────────────────────────────────────────────────────────────
    green = cp.zeros((C, A, L1), dtype=bool)
    if min_L > 0:
        green[:, :, :min_L] = (
            guesses_chunk[:, cp.newaxis, :min_L] ==
            answers_arr  [cp.newaxis, :,  :min_L]
        )

    colour = cp.zeros((C, A, L1), dtype=cp.int8)
    colour[green] = 2

    # ── YELLOW ───────────────────────────────────────────────────────────────
    for ch_val in cp.unique(answers_arr[:, :L2]):
        if int(ch_val) == 0:
            continue
        g_mask = (~green) & (guesses_chunk[:, cp.newaxis, :] == ch_val)
        a_mask_ng = (answers_arr[cp.newaxis, :, :L2] == ch_val)
        if L2 <= L1:
            a_mask_ng = a_mask_ng & ~green[:, :, :L2]
        avail = a_mask_ng.view(cp.uint8).cumsum(axis=2)[:, :, -1:]
        used  = g_mask.view(cp.uint8).cumsum(axis=2)
        colour[g_mask & (used <= avail)] = 1

    # ── Upgrade GREEN → ADJACENT (3) or BORDER (4) ──────────────────────────
    is_green = (colour == 2)

    has_adj = cp.zeros_like(is_green)
    if L1 > 1:
        has_adj[:, :, :-1] |= is_green[:, :, 1:]
        has_adj[:, :, 1:]  |= is_green[:, :, :-1]
    adj_mask = is_green & has_adj
    colour[adj_mask] = 3

    border_mask = is_green & ~has_adj
    if L1 > 0:
        colour[:, :, 0] = cp.where(border_mask[:, :, 0], 4, colour[:, :, 0])
    if 0 < L2 - 1 < L1:
        colour[:, :, L2 - 1] = cp.where(
            border_mask[:, :, L2 - 1], 4, colour[:, :, L2 - 1]
        )

    powers = cp.array([5 ** l for l in range(L1)], dtype=cp.int32)
    return (colour.astype(cp.int32) * powers).sum(axis=2)


# ──────────────────────────────────────────────────────────────────────────────
# Per guess-length processing
# ──────────────────────────────────────────────────────────────────────────────

def _process_guess_length(
    L_guess:       int,
    guess_words:   list[str],
    answer_blocks: list[tuple[np.ndarray, int, np.ndarray]],
) -> tuple[str, float]:
    V    = len(guess_words)
    garr = words_to_array(guess_words, L_guess)

    bucket_totals: list[dict[int, float]] = [{} for _ in range(V)]

    for ans_arr_cpu, L_ans, w_block in answer_blocks:
        ans_gpu = cp.array(ans_arr_cpu)

        for chunk_start in range(0, V, CHUNK_SIZE_GPU):
            chunk_end  = min(chunk_start + CHUNK_SIZE_GPU, V)
            guess_gpu  = cp.array(garr[chunk_start:chunk_end])
            pats_gpu   = _patterns_gpu_chunk(guess_gpu, ans_gpu, L_guess, L_ans)
            pats_cpu   = cp.asnumpy(pats_gpu)

            del guess_gpu, pats_gpu
            cp.get_default_memory_pool().free_all_blocks()

            C_chunk = chunk_end - chunk_start
            A_block = ans_arr_cpu.shape[0]
            for gi in range(C_chunk):
                row = pats_cpu[gi]
                bt  = bucket_totals[chunk_start + gi]
                for ai in range(A_block):
                    pat = int(row[ai])
                    w   = float(w_block[ai])
                    bt[pat] = bt.get(pat, 0.0) + w

        del ans_gpu
        cp.get_default_memory_pool().free_all_blocks()

    best_word = guess_words[0]
    best_ent  = -1.0

    for gi, gword in enumerate(guess_words):
        bt    = bucket_totals[gi]
        total = sum(bt.values())
        if total == 0:
            continue
        ent = -sum((p / total) * math.log2(p / total) for p in bt.values())
        if ent > best_ent:
            best_ent  = ent
            best_word = gword

    return best_word, best_ent


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

    answer_blocks: list[tuple[np.ndarray, int, np.ndarray]] = []
    for L_ans in lengths:
        ans_words = by_length[L_ans]
        arr       = words_to_array(ans_words, L_ans)
        w_arr     = np.array([norm_w[w] for w in ans_words], dtype=np.float64)
        answer_blocks.append((arr, L_ans, w_arr))

    total_words = len(all_words)

    gpu_name = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
    vram_gb  = cp.cuda.runtime.getDeviceProperties(0)["totalGlobalMem"] / 1e9

    print(f"GPU        : {gpu_name}  ({vram_gb:.1f} GB VRAM)")
    print(f"Vocabulary : {total_words:,} words  |  lengths: {lengths}")
    print(f"Chunk/GPU  : {CHUNK_SIZE_GPU} guesses per kernel launch\n")

    global_best_word = all_words[0]
    global_best_ent  = -1.0
    processed        = 0
    last_milestone   = -1
    t0               = time.time()

    for L_guess in lengths:
        g_words = by_length[L_guess]

        word, ent = _process_guess_length(L_guess, g_words, answer_blocks)

        if ent > global_best_ent:
            global_best_ent  = ent
            global_best_word = word

        processed += len(g_words)
        pct       = processed / total_words * 100
        milestone = int(pct // 5) * 5

        if milestone > last_milestone or pct >= 99.9:
            last_milestone = milestone
            elapsed = time.time() - t0
            print(
                f"  {min(pct, 100):5.1f}%  best so far: {global_best_word!r}"
                f"  ({global_best_ent:.4f} bits)  [{elapsed:.0f}s]"
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
