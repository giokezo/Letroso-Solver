"""
scripts/collect_patterns.py
---------------------------
Drives the Letroso browser game to collect ground-truth (guess, target, pattern)
triples. Uses the real game's feedback so patterns are always correct regardless
of any discrepancies in the local get_pattern implementation.

Each game:
  1. Pick a random word from the remaining candidates and type it.
  2. Read the real pattern from the DOM.
  3. Filter candidates using matches_feedback (constraint-based, correct).
  4. When the game is won (win pattern detected), the last guess IS the target.
     Flush all buffered rows for that game to CSV.
  5. Reload the page and repeat.

Output: scripts/patterns.csv  (appended to if it already exists)

Usage:
    python scripts/collect_patterns.py
    python scripts/collect_patterns.py --seed-word artes
"""

import argparse
import csv
import json
import os
import random
import sys
import time

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Make project root importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import parse_pattern, is_win_pattern, filter_candidates

DATA_FILE = os.path.join(os.path.dirname(__file__), "..","data", "words.json")
OUT_FILE  = os.path.join(os.path.dirname(__file__), "..","data","patterns.csv")


def open_browser() -> webdriver.Firefox:
    options = webdriver.FirefoxOptions()
    options.set_preference("privacy.trackingprotection.enabled", True)
    options.set_preference("privacy.trackingprotection.socialtracking.enabled", True)
    options.set_preference("network.cookie.cookieBehavior", 5)
    options.set_preference("dom.max_script_run_time", 10)
    driver = webdriver.Firefox(options=options)
    driver.set_page_load_timeout(30)
    ublock_path = os.path.expanduser(
        "~/.mozilla/firefox/abzqd7t7.default-release/extensions/uBlock0@raymondhill.net.xpi"
    )
    if os.path.exists(ublock_path):
        driver.install_addon(ublock_path, temporary=True)
        time.sleep(1)
    load_page(driver)
    return driver


def load_page(driver: webdriver.Firefox) -> None:
    try:
        driver.get("https://letroso.com/en/unlimited")
    except Exception:
        pass  # ad-related timeout; game usually still loads
    WebDriverWait(driver, 60).until(
        EC.element_to_be_clickable((By.ID, "key_a"))
    )
    time.sleep(0.5)


def type_word(driver: webdriver.Firefox, word: str) -> None:
    for ch in word.lower():
        driver.find_element(By.ID, f"key_{ch}").click()


def press_enter(driver: webdriver.Firefox) -> None:
    driver.find_element(By.ID, "key_enter").click()


def press_backspace(driver: webdriver.Firefox) -> None:
    driver.find_element(By.ID, "key_backspace").click()


def wait_for_feedback(driver: webdriver.Firefox, guess_index: int, timeout: float = 10) -> None:
    def check(_driver):
        board = _driver.find_element(By.CSS_SELECTOR, ".board")
        guesses = board.find_elements(By.CSS_SELECTOR, ".guess")
        if len(guesses) <= guess_index:
            return False
        letters = guesses[guess_index].find_elements(By.CSS_SELECTOR, ".letter")
        if not letters:
            return False
        for ltr in letters:
            cls = ltr.get_attribute("class") or ""
            feedback_classes = [c for c in cls.split() if c != "letter"]
            if not feedback_classes:
                return False
        return True
    WebDriverWait(driver, timeout).until(check)


def read_feedback(driver: webdriver.Firefox, guess_index: int) -> str:
    board = driver.find_element(By.CSS_SELECTOR, ".board")
    guesses = board.find_elements(By.CSS_SELECTOR, ".guess")
    letters = guesses[guess_index].find_elements(By.CSS_SELECTOR, ".letter")

    letter_info = []
    for ltr in letters:
        cls_str = ltr.get_attribute("class") or ""
        classes = set(cls_str.split())
        classes.discard("letter")
        letter_info.append(classes)

    parts = []
    for i, classes in enumerate(letter_info):
        if "absent" in classes:
            code = "B"
        elif "one-present" in classes:
            code = "G"
        elif "present" in classes:
            code = "Y"
        elif "start" in classes or "end" in classes:
            code = "P"
        else:
            code = "G"

        parts.append(code)

        if code in ("G", "P") and i + 1 < len(letter_info):
            if "tail" in letter_info[i + 1]:
                parts.append("O")

    return "".join(parts)


def play_one_game(
    driver: webdriver.Firefox,
    vocab: list[str],
    seed_word: str | None,
) -> list[tuple[str, str, str]] | None:
    """
    Play one game. Returns list of (guess, target, pattern_str) rows,
    or None if the game failed (invalid word, timeout, parse error).
    """
    candidates = list(vocab)
    buffered: list[tuple[str, str]] = []   # (guess, pattern_str) — target unknown yet
    turn = 0

    while True:
        turn += 1

        # Pick guess: seed word on turn 1 if provided, else random candidate
        if turn == 1 and seed_word and seed_word in candidates:
            guess = seed_word
        else:
            guess = random.choice(candidates)

        type_word(driver, guess)
        press_enter(driver)

        guess_idx = turn - 1
        try:
            wait_for_feedback(driver, guess_idx, timeout=8)
        except Exception:
            # Word rejected (not in game's dictionary) — backspace and remove from pool
            for _ in range(len(guess)):
                press_backspace(driver)
            candidates = [w for w in candidates if w != guess]
            turn -= 1
            continue

        feedback = read_feedback(driver, guess_idx)
        pattern_int = parse_pattern(feedback, len(guess))
        if pattern_int is None:
            return None  # shouldn't happen; bail out

        buffered.append((guess, feedback))

        if is_win_pattern(pattern_int, len(guess)):
            # guess == target; fill in target for all buffered rows
            target = guess
            return [(g, target, pat) for g, pat in buffered]

        # Filter candidates using real feedback
        candidates = filter_candidates(candidates, guess, pattern_int)
        if not candidates:
            return None  # no candidates left; feedback inconsistency


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect Letroso game patterns via browser")
    parser.add_argument("--seed-word", default=None,
                        help="Fixed first guess for every game (optional)")
    args = parser.parse_args()

    if not os.path.exists(DATA_FILE):
        print(f"ERROR: {DATA_FILE} not found. Run download_words.py first.")
        sys.exit(1)

    print("Loading vocabulary...")
    with open(DATA_FILE, encoding="utf-8") as f:
        vocab = list(json.load(f).keys())
    print(f"Loaded {len(vocab):,} words")

    write_header = not os.path.exists(OUT_FILE) or os.path.getsize(OUT_FILE) == 0

    total_rows  = 0
    total_games = 0
    failed      = 0
    t0          = time.time()

    print("Opening browser...")
    driver = open_browser()
    print(f"Writing to {OUT_FILE}")
    print("Ctrl-C to stop\n")

    try:
        with open(OUT_FILE, "a", newline="", encoding="utf-8") as csvfile:
            w = csv.writer(csvfile)
            if write_header:
                w.writerow(["guess", "target", "pattern"])

            while True:
                rows = play_one_game(driver, vocab, args.seed_word)

                if rows:
                    w.writerows(rows)
                    csvfile.flush()
                    total_rows  += len(rows)
                    total_games += 1
                else:
                    failed += 1

                elapsed = time.time() - t0
                print(
                    f"\r  games: {total_games:,}  rows: {total_rows:,}  "
                    f"failed: {failed}  [{elapsed:.0f}s]",
                    end="", flush=True,
                )

                # Reload for next game
                load_page(driver)

    except KeyboardInterrupt:
        print(f"\n\nStopped. {total_rows:,} rows from {total_games:,} games written to {OUT_FILE}")
    finally:
        driver.quit()


if __name__ == "__main__":
    main()
