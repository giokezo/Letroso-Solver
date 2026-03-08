#!/usr/bin/env python3
"""Verify engine against data/patterns.csv. Report get_pattern mismatches and FAILED analysis."""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import (
    get_pattern,
    pattern_to_str,
    parse_pattern,
    decode_pattern,
    matches_feedback,
    filter_candidates,
)

CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "patterns.csv")
DATA_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "words.json")


def main():
    if not os.path.exists(CSV_PATH):
        print(f"Not found: {CSV_PATH}")
        return

    rows = []
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append((row["guess"].strip().lower(), row["target"].strip().lower(), row["pattern"].strip()))

    success = [(g, t, p) for g, t, p in rows if t != "failed"]
    failed = [(g, t, p) for g, t, p in rows if t == "failed"]

    get_pattern.cache_clear()

    # 1) Successful rows: get_pattern(guess, target) should match pattern
    mismatches = []
    for guess, target, pattern in success:
        L = len(guess)
        if len(target) != L and "failed" in target.lower():
            continue
        pint = parse_pattern(pattern, L)
        if pint is None:
            mismatches.append((guess, target, pattern, None, "parse_fail"))
            continue
        our = get_pattern(guess, target)
        our_str = pattern_to_str(our, L)
        if our_str != pattern:
            mismatches.append((guess, target, pattern, our_str, "mismatch"))

    print("=== Successful rows: get_pattern(guess, target) vs CSV pattern ===\n")
    print(f"Total successful: {len(success)}")
    print(f"Mismatches: {len(mismatches)}")
    for g, t, p, our, kind in mismatches[:25]:
        print(f"  {g} vs {t}: csv={p!r} ours={our!r}")

    # 2) matches_feedback: every successful (guess, target, pattern) must pass
    mf_fails = []
    for guess, target, pattern in success:
        L = len(guess)
        pint = parse_pattern(pattern, L)
        if pint is None:
            continue
        states, concats = decode_pattern(pint, L)
        if not matches_feedback(guess, target, states, concats):
            mf_fails.append((guess, target, pattern))
    print(f"\nmatches_feedback(guess, target, pattern) fails: {len(mf_fails)}")
    for g, t, p in mf_fails[:10]:
        print(f"  {g} vs {t} pattern={p!r}")

    # 3) FAILED rows: with constraint filter, how many candidates survive?
    if failed and os.path.exists(DATA_FILE):
        import json
        with open(DATA_FILE, encoding="utf-8") as f:
            vocab = list(json.load(f).keys())
        print(f"\n=== FAILED rows: candidates left if we use constraint filter ===\n")
        zero_count = 0
        for guess, _, pattern in failed[:30]:
            L = len(guess)
            pint = parse_pattern(pattern, L)
            if pint is None:
                print(f"  {guess} pattern={pattern!r} -> parse None")
                zero_count += 1
                continue
            candidates = [w for w in vocab if len(w) == L]
            if not candidates:
                candidates = vocab
            filtered = filter_candidates(candidates, guess, pint)
            if len(filtered) == 0:
                zero_count += 1
                print(f"  {guess} ({pattern!r}) -> 0 candidates")
            else:
                print(f"  {guess} ({pattern!r}) -> {len(filtered)} candidates")
        print(f"\nFAILED rows with 0 candidates (first 30): {zero_count}")
    return mismatches, mf_fails


if __name__ == "__main__":
    main()
