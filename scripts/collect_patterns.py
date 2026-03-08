"""
scripts/collect_patterns.py
---------------------------
Drives the Letroso browser game to collect ground-truth (guess, target, pattern)
triples. Never fails: only filters by black letters, then guesses randomly.

Each game:
  1. Pick a random word from the remaining candidates and type it.
  2. Read the real pattern from the DOM.
  3. Filter candidates only by black: remove words that contain more of a letter
     than the feedback allows (greens + yellows for that letter). Never over-filter.
  4. When the game is won (all positions green/border), the last guess IS the target.
     Flush all buffered rows to CSV.
  5. Reload the page and repeat.

Output: data/patterns.csv  (appended to if it already exists)

The unlimited page defaults to 3-letter mode, so without --length all targets are length 3.
Use --length 5 (or 3–10) to set word length before each game; use --length random to vary.

Usage:
    python scripts/collect_patterns.py
    python scripts/collect_patterns.py --seed-word artes
    python scripts/collect_patterns.py --length 5
    python scripts/collect_patterns.py --length random
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
from engine import is_win_pattern, parse_pattern

DATA_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "words.json")
OUT_FILE  = os.path.join(os.path.dirname(__file__), "..", "data", "patterns.csv")

# State codes from feedback string (for black-only filter)
_B, _Y, _G, _P = 0, 1, 2, 3
_CODE = {"B": _B, "Y": _Y, "G": _G, "P": _P}


def feedback_to_states(feedback: str, length: int) -> list[int] | None:
    """Parse feedback string (e.g. 'BBGYOP') into list of state codes. O is concat, not a position."""
    states: list[int] = []
    for c in feedback.strip().upper():
        if c in _CODE:
            states.append(_CODE[c])
        elif c == "O":
            continue
        else:
            return None
    return states if len(states) == length else None


def filter_by_black_only(
    candidates: list[str],
    guess: str,
    feedback: str,
) -> list[str]:
    """
    Keep only words that are consistent with black feedback: for each letter L,
    the answer can have at most (number of non-black positions for L in the guess).
    So we never remove the true target and never run out of candidates from logic.
    """
    states = feedback_to_states(feedback, len(guess))
    if states is None:
        return candidates  # can't parse; don't filter

    # For each letter in the guess, max count in answer = non-black count (0 if only black)
    max_allowed: dict[str, int] = {c: 0 for c in guess}
    for i, c in enumerate(guess):
        if i < len(states) and states[i] != _B:
            max_allowed[c] += 1

    # Exclude words that contain too many of any letter, or any letter that was only black
    result = []
    for w in candidates:
        ok = True
        for c in set(w):
            if w.count(c) > max_allowed.get(c, 999):
                ok = False
                break
        if ok:
            result.append(w)

    return result if result else candidates  # never return empty; keep all if filter would empty


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


def set_word_length(driver: webdriver.Firefox, length: int) -> bool:
    """
    Try to set the game's word length (3–10). The unlimited page defaults to 3;
    without this, every target is length 3. Returns True if a length control was clicked.
    """
    if not (3 <= length <= 10):
        return False
    time.sleep(0.3)
    # Try data-length attribute (common in React)
    try:
        el = driver.find_element(By.CSS_SELECTOR, f"[data-length=\"{length}\"]")
        el.click()
        time.sleep(0.3)
        return True
    except Exception:
        pass
    # Try button/link with exact text (e.g. "5")
    try:
        el = driver.find_element(By.XPATH, f"//button[text()=\"{length}\"]")
        el.click()
        time.sleep(0.3)
        return True
    except Exception:
        pass
    try:
        el = driver.find_element(By.XPATH, f"//*[@role='button' and text()=\"{length}\"]")
        el.click()
        time.sleep(0.3)
        return True
    except Exception:
        pass
    return False


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


def _is_win_from_feedback(feedback: str, length: int) -> bool:
    """True if feedback indicates a win (all green/border). Works even when parse_pattern would fail."""
    states = feedback_to_states(feedback, length)
    if states is None or len(states) != length:
        return False
    return all(s in (_G, _P) for s in states)


def play_one_game(
    driver: webdriver.Firefox,
    vocab: list[str],
    seed_word: str | None,
    word_length: int | None = None,
) -> list[tuple[str, str, str]] | None:
    """
    Play one game. Returns list of (guess, target, pattern_str) rows,
    or None only on unrecoverable error (e.g. invalid word and no candidates left after removal).
    Never fails due to 0 candidates from filtering; uses black-only filter and random guess.
    If word_length is set (3–10), only guess words of that length (so the board accepts them).
    """
    if word_length is not None and 3 <= word_length <= 10:
        candidates = [w for w in vocab if len(w) == word_length]
        if not candidates:
            candidates = list(vocab)
    else:
        candidates = list(vocab)
    buffered: list[tuple[str, str]] = []   # (guess, pattern_str) — target unknown yet
    turn = 0

    while True:
        turn += 1

        # Pick guess: seed word on turn 1 if provided, else random from candidates (or full vocab fallback)
        if turn == 1 and seed_word and seed_word in candidates:
            guess = seed_word
        elif candidates:
            guess = random.choice(candidates)
        else:
            guess = random.choice(vocab)

        type_word(driver, guess)
        press_enter(driver)

        guess_idx = turn - 1
        try:
            wait_for_feedback(driver, guess_idx, timeout=8)
        except Exception:
            # Word rejected (not in game's dictionary) — backspace and remove from pool
            for _ in range(len(guess)):
                press_backspace(driver)
            candidates = [w for w in candidates if w != guess] if candidates else list(vocab)
            turn -= 1
            continue

        feedback = read_feedback(driver, guess_idx)
        buffered.append((guess, feedback))

        # Win: engine's is_win_pattern if we have pattern_int, else heuristic from feedback string
        pattern_int = parse_pattern(feedback, len(guess))
        if pattern_int is not None and is_win_pattern(pattern_int, len(guess)):
            target = guess
            return [(g, target, pat) for g, pat in buffered]
        if _is_win_from_feedback(feedback, len(guess)):
            target = guess
            return [(g, target, pat) for g, pat in buffered]

        # Black-only filter; then guess randomly next time. Never empty candidates.
        candidates = filter_by_black_only(candidates, guess, feedback)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect Letroso game patterns via browser")
    parser.add_argument("--seed-word", default=None,
                        help="Fixed first guess for every game (optional)")
    parser.add_argument("--length", default=None,
                        help="Word length 3–10, or 'random' to vary each game (default: page default, usually 3)")
    args = parser.parse_args()

    word_length_arg: int | None | str = None
    if args.length is not None:
        if args.length.strip().lower() == "random":
            word_length_arg = "random"
        else:
            try:
                n = int(args.length)
                if 3 <= n <= 10:
                    word_length_arg = n
                else:
                    print("ERROR: --length must be 3–10 or 'random'")
                    sys.exit(1)
            except ValueError:
                print("ERROR: --length must be an integer 3–10 or 'random'")
                sys.exit(1)

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
                # Set word length for this game (unlimited page defaults to 3)
                word_length = None
                if word_length_arg == "random":
                    word_length = random.randint(3, 10)
                    set_word_length(driver, word_length)
                elif isinstance(word_length_arg, int):
                    word_length = word_length_arg
                    set_word_length(driver, word_length)

                rows = play_one_game(driver, vocab, args.seed_word, word_length)

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
