import os
import time

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from engine import (
    parse_pattern,
    filter_candidates,
    is_win_pattern,
    rank_guesses,
)
from solver import load_data, load_first_guess
from selenium_solver import open_browser, load_page, read_feedback, wait_for_feedback

OVERLAY_ID = "solver_overlay"


def inject_overlay(driver: webdriver.Firefox, lines: list[str], title: str = "") -> None:
    """Inject / update a floating recommendations panel on the page."""
    rows_html = "".join(
        f'<div style="padding:2px 0;font-size:13px;font-family:monospace">{line}</div>'
        for line in lines
    )
    title_html = f'<div style="font-weight:bold;margin-bottom:6px;font-size:14px">{title}</div>' if title else ""
    html = title_html + rows_html

    driver.execute_script(f"""
        var el = document.getElementById('{OVERLAY_ID}');
        if (!el) {{
            el = document.createElement('div');
            el.id = '{OVERLAY_ID}';
            el.style.cssText = [
                'position:fixed', 'top:12px', 'right:12px', 'z-index:99999',
                'background:rgba(20,20,20,0.93)', 'color:#eee',
                'border:1px solid #555', 'border-radius:8px',
                'padding:12px 16px', 'max-width:320px',
                'box-shadow:0 4px 20px rgba(0,0,0,0.5)',
                'pointer-events:none'
            ].join(';');
            document.body.appendChild(el);
        }}
        el.innerHTML = {repr(html)};
    """)


def remove_overlay(driver: webdriver.Firefox) -> None:
    driver.execute_script(f"""
        var el = document.getElementById('{OVERLAY_ID}');
        if (el) el.remove();
    """)


def read_guess_word(driver: webdriver.Firefox, guess_index: int) -> str:
    """Read the letters the player typed for a given guess row."""
    board = driver.find_element(By.CSS_SELECTOR, ".board")
    guesses = board.find_elements(By.CSS_SELECTOR, ".guess")
    letters = guesses[guess_index].find_elements(By.CSS_SELECTOR, ".letter")
    return "".join(l.text.strip().lower() for l in letters)


def count_submitted_guesses(driver: webdriver.Firefox) -> int:
    """Return how many guess rows already have feedback classes."""
    try:
        board = driver.find_element(By.CSS_SELECTOR, ".board")
        guesses = board.find_elements(By.CSS_SELECTOR, ".guess")
        count = 0
        for g in guesses:
            letters = g.find_elements(By.CSS_SELECTOR, ".letter")
            if not letters:
                break
            # A submitted guess has feedback classes beyond just "letter"
            first_cls = letters[0].get_attribute("class") or ""
            extra = set(first_cls.split()) - {"letter"}
            if extra:
                count += 1
            else:
                break
        return count
    except Exception:
        return 0


def wait_for_next_submission(driver: webdriver.Firefox, expected_index: int) -> bool:
    """Block until the player submits their next word (guess_index = expected_index)."""
    while True:
        submitted = count_submitted_guesses(driver)
        if submitted > expected_index:
            return True
        time.sleep(0.3)


def show_recommendations(
    driver: webdriver.Firefox,
    ranked: list[tuple[str, float]],
    candidates: list[str],
    turn: int,
) -> None:
    n = len(candidates)
    lines = []
    for i, (word, ent) in enumerate(ranked):
        marker = "►" if i == 0 else f"{i+1:2}."
        lines.append(f"{marker} {word:<14} {ent:.4f} bits")

    lines.append(f"── {n:,} candidate{'s' if n != 1 else ''} remaining ──")

    title = f"Turn {turn} — top {len(ranked)} suggestions"
    inject_overlay(driver, lines, title=title)

    # Also print to terminal
    print(f"\n  Top suggestions (turn {turn}, {n:,} candidates):")
    for i, (word, ent) in enumerate(ranked):
        marker = "►" if i == 0 else f"  "
        print(f"  {marker} {word:<14} {ent:.4f} bits")


def play(driver: webdriver.Firefox, vocab: list[str], weights: dict, first) -> bool:
    """Returns True if solved, False if failed (no candidates remain)."""
    candidates = list(vocab)
    turn = 0

    def _progress(done, total):
        pct = done / total * 100 if total else 0
        print(f"\r  Computing... {pct:.0f}%", end="", flush=True)

    while True:
        turn += 1

        if len(candidates) == 0:
            print("\nNo candidates remain — moving to next game.")
            return False

        if len(candidates) == 1:
            word = candidates[0]
            inject_overlay(driver, [f"► {word}   (only candidate)"], title=f"Turn {turn} — answer found!")
            print(f"\n  Only candidate: {word}")
            print(f"  Type '{word}' in the browser and press Enter.")

        if turn == 1:
            print(f"\n  Type your first word in the browser and press Enter...")
        elif len(candidates) > 1:
            print(f"\n  Type your word in the browser and press Enter...")

        wait_for_next_submission(driver, turn - 1)

        # Brief pause for animation to settle
        time.sleep(1.2)

        guess_idx = turn - 1
        try:
            wait_for_feedback(driver, guess_idx, timeout=5)
        except Exception:
            pass

        guess    = read_guess_word(driver, guess_idx)
        feedback = read_feedback(driver, guess_idx)
        print(f"  You played:  {guess}")
        print(f"  Feedback:    {feedback}")

        if not guess:
            print("  Could not read guess from browser.")
            return False

        pattern_int = parse_pattern(feedback, len(guess))
        if pattern_int is None:
            print(f"  ERROR: Could not parse feedback '{feedback}'")
            return False

        if is_win_pattern(pattern_int, len(guess)):
            remove_overlay(driver)
            print(f"\n  Solved in {turn} turn{'s' if turn > 1 else ''}!  Answer: {guess}")
            return True

        old_count  = len(candidates)
        candidates = filter_candidates(candidates, guess, pattern_int)
        print(f"  {old_count:,} → {len(candidates):,} candidates remaining")

        if 0 < len(candidates) <= 15:
            print(f"  Remaining: {', '.join(candidates)}")

        if len(candidates) == 0:
            continue  # overlay keeps last content; handled at top of next loop

        inject_overlay(driver, ["⏳ Computing recommendations..."], title=f"Turn {turn + 1}")

        t0 = time.time()
        ranked = rank_guesses(candidates, vocab, weights, top_n=10, progress_fn=_progress)
        print(f"\r  Done in {time.time()-t0:.1f}s            ")
        show_recommendations(driver, ranked, candidates, turn + 1)


def _wait_for_board_reset(driver: webdriver.Firefox, timeout: float = 300) -> None:
    """Wait until the board is empty again (user clicked play again on the site)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if count_submitted_guesses(driver) == 0:
            return
        time.sleep(0.5)


def main() -> None:
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
            print(f"  GAME #{game_num}  —  play in the browser, I'll guide you")
            print(f"{'='*50}")

            solved = play(driver, vocab, weights, first)

            if solved:
                print("\n  Click 'Play again' in the browser to start the next game...")
                _wait_for_board_reset(driver)
            else:
                print("\n  Starting next game in 3s...")
                time.sleep(3)
                load_page(driver)

    except KeyboardInterrupt:
        pass
    finally:
        print("\nClosing browser.")
        driver.quit()


if __name__ == "__main__":
    main()
