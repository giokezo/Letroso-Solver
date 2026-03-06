import os
import time
import sys
from datetime import datetime

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from engine import (
    parse_pattern,
    decode_pattern,
    filter_candidates,
    is_win_pattern,
    rank_guesses,
    BLACK, YELLOW, GREEN, BORDER,
)
from solver import load_data, load_first_guess

BASE_DIR = os.path.dirname(__file__)
LOG_FILE = os.path.join(BASE_DIR, "solve_log.csv")


# ── Logging ───────────────────────────────────────────────────────────────────

def init_log():
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w") as f:
            f.write("timestamp,answer,turns,guesses\n")


def log_result(answer: str, turns: int, guesses: list[str]):
    with open(LOG_FILE, "a") as f:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        guess_list = " -> ".join(guesses)
        f.write(f"{ts},{answer},{turns},{guess_list}\n")


# ── Selenium helpers ──────────────────────────────────────────────────────────

def open_browser() -> webdriver.Firefox:
    options = webdriver.FirefoxOptions()
    options.set_preference("privacy.trackingprotection.enabled", True)
    options.set_preference("privacy.trackingprotection.socialtracking.enabled", True)
    options.set_preference("network.cookie.cookieBehavior", 5)
    options.set_preference("dom.max_script_run_time", 10)
    driver = webdriver.Firefox(options=options)
    driver.set_page_load_timeout(30)
    # Install uBlock Origin to kill ads
    ublock_path = os.path.expanduser(
        "~/.mozilla/firefox/abzqd7t7.default-release/extensions/uBlock0@raymondhill.net.xpi"
    )
    if os.path.exists(ublock_path):
        driver.install_addon(ublock_path, temporary=True)
        print("  uBlock Origin loaded")
        time.sleep(1)
    load_page(driver)
    return driver


def load_page(driver: webdriver.Firefox):
    print("  Waiting for page to load...")
    try:
        driver.get("https://letroso.com/en/unlimited")
    except Exception:
        print("  Page load timed out (ads), checking if game is ready...")
    WebDriverWait(driver, 60).until(
        EC.element_to_be_clickable((By.ID, "key_a"))
    )
    print("  Page ready!")
    time.sleep(1)


def type_word(driver: webdriver.Firefox, word: str) -> None:
    for ch in word.lower():
        btn = driver.find_element(By.ID, f"key_{ch}")
        btn.click()
        time.sleep(0.05)


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
        for l in letters:
            cls = l.get_attribute("class") or ""
            classes = cls.split()
            feedback_classes = [c for c in classes if c != "letter"]
            if not feedback_classes:
                return False
        return True

    WebDriverWait(driver, timeout).until(check)


def read_feedback(driver: webdriver.Firefox, guess_index: int) -> str:
    board = driver.find_element(By.CSS_SELECTOR, ".board")
    guesses = board.find_elements(By.CSS_SELECTOR, ".guess")
    letters = guesses[guess_index].find_elements(By.CSS_SELECTOR, ".letter")

    letter_info = []
    for l in letters:
        cls_str = l.get_attribute("class") or ""
        classes = set(cls_str.split())
        classes.discard("letter")
        letter_info.append(classes)

    parts = []
    for i, classes in enumerate(letter_info):
        if "absent" in classes:
            code = "B"
        elif "present" in classes or "one-present" in classes:
            code = "Y"
        elif "start" in classes or "end" in classes:
            code = "P"
        else:
            code = "G"

        parts.append(code)

        if code in ("G", "P") and i + 1 < len(letter_info):
            next_classes = letter_info[i + 1]
            if "tail" in next_classes:
                parts.append("O")

    return "".join(parts)


# ── Single game ───────────────────────────────────────────────────────────────

def solve_one(driver: webdriver.Firefox, vocab, weights, first) -> tuple[str | None, int, list[str]]:
    """
    Play one round. Returns (answer, turns, list_of_guesses).
    answer is None if it failed.
    """
    candidates = list(vocab)
    guesses = []
    turn = 0

    while True:
        turn += 1

        # Pick guess
        if turn == 1 and first:
            guess, entropy = first
            print(f"\nTurn {turn} | candidates: {len(candidates):,}")
            print(f"  Guess: {guess} (precomputed, {entropy:.4f} bits)")
        elif len(candidates) == 1:
            guess = candidates[0]
            print(f"\nTurn {turn} | candidates: 1")
            print(f"  Final answer: {guess}")
        elif len(candidates) == 0:
            print("\nNo candidates remain - feedback may have been incorrect.")
            return None, turn, guesses
        else:
            print(f"\nTurn {turn} | candidates: {len(candidates):,}")
            t0 = time.time()
            def _progress(done, total):
                print(f"\r  Computing best guess... ({done:,}/{total:,})", end="", flush=True)
            ranked = rank_guesses(candidates, vocab, weights, top_n=5,
                                  progress_fn=_progress)
            elapsed = time.time() - t0
            guess, entropy = ranked[0]
            print(f"\r  Guess: {guess} ({entropy:.4f} bits) [{elapsed:.1f}s]")
            if len(ranked) > 1:
                alts = ", ".join(f"{w} ({e:.2f})" for w, e in ranked[1:])
                print(f"  Also good: {alts}")

        guesses.append(guess)

        # Type the guess
        type_word(driver, guess)
        time.sleep(0.3)
        press_enter(driver)

        # Wait for feedback
        guess_idx = turn - 1
        try:
            wait_for_feedback(driver, guess_idx)
        except Exception:
            print("  Timeout waiting for feedback. Word may be invalid.")
            for _ in range(len(guess)):
                press_backspace(driver)
                time.sleep(0.05)
            candidates = [w for w in candidates if w != guess]
            guesses.pop()
            turn -= 1
            continue

        time.sleep(1)

        # Read feedback
        feedback = read_feedback(driver, guess_idx)
        print(f"  Feedback: {feedback}")

        pattern_int = parse_pattern(feedback, len(guess))
        if pattern_int is None:
            print(f"  ERROR: Could not parse feedback '{feedback}'")
            return None, turn, guesses

        # Check for win
        if is_win_pattern(pattern_int, len(guess)):
            print(f"\n  Solved in {turn} turn{'s' if turn > 1 else ''}! Answer: {guess}")
            return guess, turn, guesses

        # Filter candidates
        old_count = len(candidates)
        candidates = filter_candidates(candidates, guess, pattern_int)
        print(f"  {old_count:,} -> {len(candidates):,} candidates")

        if len(candidates) <= 10:
            print(f"  Remaining: {', '.join(candidates)}")


# ── Main loop ─────────────────────────────────────────────────────────────────

def main() -> None:
    init_log()
    print("Loading vocabulary...")
    vocab, weights = load_data()
    first = load_first_guess()

    print(f"Vocabulary: {len(vocab):,} words")
    print("Opening browser...")
    driver = open_browser()

    game_num = 0
    try:
        while True:
            game_num += 1
            print(f"\n{'='*50}")
            print(f"  GAME #{game_num}")
            print(f"{'='*50}")

            answer, turns, guesses = solve_one(driver, vocab, weights, first)

            if answer:
                log_result(answer, turns, guesses)
                print(f"  Logged: {answer} in {turns} turns")
            else:
                log_result("FAILED", turns, guesses)
                print(f"  Logged: FAILED after {turns} turns")

            # Start new game by reloading the page
            print("\n  Starting next game in 3s...")
            time.sleep(3)
            load_page(driver)

    except KeyboardInterrupt:
        print(f"\n\nStopped after {game_num} games. Results in {LOG_FILE}")
    finally:
        driver.quit()


if __name__ == "__main__":
    main()
