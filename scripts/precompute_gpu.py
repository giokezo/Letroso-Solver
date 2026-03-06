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

from engine import get_pattern

DATA_FILE = os.path.join(os.path.dirname(__file__), "..","data", "words.json")
OUT_FILE  = os.path.join(os.path.dirname(__file__), "..","data", "first_guesses.json")

CHUNK_SIZE_GPU = 500


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
    norm_w_gpu = cp.array(norm_w)

    total_words = len(all_words)
    by_length: dict[int, int] = {}
    for w in all_words:
        by_length[len(w)] = by_length.get(len(w), 0) + 1
    lengths = sorted(by_length.keys())

    # Bypass lru_cache for precompute (every pair is unique)
    _get_pattern = get_pattern.__wrapped__

    gpu_name = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
    vram_gb  = cp.cuda.runtime.getDeviceProperties(0)["totalGlobalMem"] / 1e9

    print(f"GPU        : {gpu_name}  ({vram_gb:.1f} GB VRAM)")
    print(f"Vocabulary : {total_words:,} words  |  lengths: {lengths}")
    print(f"Chunk/GPU  : {CHUNK_SIZE_GPU} guesses per batch\n")

    global_best_word = all_words[0]
    global_best_ent  = -1.0
    processed        = 0
    t0               = time.time()

    A = total_words

    for chunk_start in range(0, total_words, CHUNK_SIZE_GPU):
        chunk_end   = min(chunk_start + CHUNK_SIZE_GPU, total_words)
        chunk_words = all_words[chunk_start:chunk_end]
        C           = len(chunk_words)

        # Compute patterns on CPU
        pats_cpu = np.empty((C, A), dtype=np.int32)
        for gi, guess in enumerate(chunk_words):
            for ai, answer in enumerate(all_words):
                pats_cpu[gi, ai] = _get_pattern(guess, answer)

        # Move to GPU for entropy computation
        pats_gpu = cp.array(pats_cpu)
        max_pat  = int(pats_gpu.max()) + 1

        for gi in range(C):
            row    = pats_gpu[gi]
            bucket = cp.bincount(row, weights=norm_w_gpu, minlength=max_pat)
            probs  = bucket[bucket > 0]
            total  = float(probs.sum())
            if total == 0:
                continue
            probs = probs / total
            ent   = -float(cp.sum(probs * cp.log2(probs)))
            if ent > global_best_ent:
                global_best_ent  = ent
                global_best_word = chunk_words[gi]

        del pats_gpu
        cp.get_default_memory_pool().free_all_blocks()

        processed += C
        pct     = processed / total_words * 100
        elapsed = time.time() - t0
        print(
            f"\r  {pct:5.1f}%  ({processed:,}/{total_words:,})  "
            f"best: {global_best_word!r} ({global_best_ent:.4f} bits)  "
            f"[{elapsed:.0f}s]",
            end="", flush=True,
        )

    elapsed = time.time() - t0
    print(f"\n\n  100%  done in {elapsed:.1f}s")
    print(f"  Global best opener: {global_best_word!r}  entropy = {global_best_ent:.4f} bits")

    result = {"word": global_best_word, "entropy": round(global_best_ent, 6)}
    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    with open(OUT_FILE, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
    print(f"  Saved → {OUT_FILE}")


if __name__ == "__main__":
    main()
