#!/usr/bin/env python3
"""
Verify engine.get_pattern(guess, answer) matches site feedback from solve_log.csv.
Report mismatches to reverse-engineer site rules; focus on FAILED games.
"""
import csv
import re
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import get_pattern, pattern_to_str, parse_pattern, filter_candidates

LOG_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "solve_log.csv")
DATA_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "words.json")


def parse_guess_list(guess_list_str: str) -> list[tuple[str, str]]:
    """Parse 'word1(fb1) -> word2(fb2)' into [(word1, fb1), (word2, fb2)]."""
    pairs = []
    # Match word(feedback) segments; feedback is [BGYPO]+
    for m in re.finditer(r"([a-zA-Z]+)\(([BGYPO]+)\)", guess_list_str):
        pairs.append((m.group(1).lower(), m.group(2)))
    return pairs


def main():
    if not os.path.exists(LOG_PATH):
        print(f"Log not found: {LOG_PATH}")
        return

    mismatches = []
    failed_entries = []
    success_count = 0
    total_checked = 0

    with open(LOG_PATH, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if len(row) < 4:
                continue
            ts, answer, turns, guess_list = row[0], row[1], row[2], row[3]
            pairs = parse_guess_list(guess_list)
            if not pairs:
                continue

            if answer == "FAILED":
                failed_entries.append((answer, turns, pairs))
                continue

            answer = answer.lower()
            success_count += 1
            for guess, site_fb in pairs:
                total_checked += 1
                our_pattern = get_pattern(guess, answer)
                our_fb = pattern_to_str(our_pattern, len(guess))
                if our_fb != site_fb:
                    mismatches.append({
                        "answer": answer,
                        "guess": guess,
                        "site_fb": site_fb,
                        "our_fb": our_fb,
                        "turn": len([p for p in pairs if pairs.index((guess, site_fb)) >= 0]),
                    })

    print("=== Verification: get_pattern(guess, answer) vs site feedback ===\n")
    print(f"Successful games: {success_count}")
    print(f"Total (guess, feedback) pairs checked: {total_checked}")
    print(f"Mismatches: {len(mismatches)}\n")

    if mismatches:
        print("--- Mismatches (our pattern != site) ---")
        for m in mismatches[:40]:
            print(f"  answer={m['answer']!r} guess={m['guess']!r}")
            print(f"    site: {m['site_fb']!r}")
            print(f"    ours: {m['our_fb']!r}")
        if len(mismatches) > 40:
            print(f"  ... and {len(mismatches) - 40} more")

    print("\n--- FAILED games (site feedback only; no answer to verify) ---")
    for answer, turns, pairs in failed_entries:
        print(f"  turns={turns} guesses={[w for w, _ in pairs]}")
        for guess, site_fb in pairs:
            print(f"    {guess}({site_fb})")

    # For each FAILED (guess, feedback): simulate filtering; see if we go to 0 candidates.
    if failed_entries and os.path.exists(DATA_FILE):
        import json
        with open(DATA_FILE, encoding="utf-8") as f:
            vocab = list(json.load(f).keys())
        print("\n--- FAILED: candidates left after each turn (using site feedback) ---")
        for _answer, turns, pairs in failed_entries:
            candidates = list(vocab)
            for guess, site_fb in pairs:
                L = len(guess)
                if not candidates:
                    print(f"  {guess}({site_fb}) -> 0 (already empty)")
                    continue
                pattern_int = parse_pattern(site_fb, L)
                if pattern_int is None:
                    print(f"  {guess}({site_fb}) -> parse_pattern returned None (len={L})")
                    continue
                before = len(candidates)
                candidates = filter_candidates(candidates, guess, pattern_int)
                print(f"  {guess}({site_fb}) -> {before} -> {len(candidates)} candidates")
                if len(candidates) <= 5 and candidates:
                    print(f"    remaining: {candidates}")

    return mismatches, failed_entries


if __name__ == "__main__":
    main()
