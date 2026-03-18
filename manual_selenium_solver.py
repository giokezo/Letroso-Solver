import os
import time
import json

from database import start_new_session, save_game_score
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
SESSION_OVERLAY_ID = "session_overlay"

def inject_overlay(driver: webdriver.Firefox, lines: list[str], title: str = "") -> None:
    """Original Top-Right suggestions panel."""
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

def inject_session_ui(driver: webdriver.Firefox, name: str = "") -> None:
    """Bottom-Left Login UI with flicker prevention."""
    target_state = "PLAYING" if name else "LOGIN"
    driver.execute_script(f"""
        var el = document.getElementById('{SESSION_OVERLAY_ID}');
        var targetState = "{target_state}";
        if (el && el.getAttribute('data-state') === targetState) return;
        if (!el) {{
            el = document.createElement('div');
            el.id = '{SESSION_OVERLAY_ID}';
            el.style.cssText = 'position:fixed;bottom:20px;left:20px;z-index:10000;background:rgba(30,30,30,0.95);color:#eee;border:2px solid #555;border-radius:10px;padding:15px;width:200px;box-shadow:0 0 15px rgba(0,0,0,0.5);pointer-events:auto;';
            document.body.appendChild(el);
        }}
        el.setAttribute('data-state', targetState);
        if (targetState === "LOGIN") {{
            el.innerHTML = `
                <div style="font-weight:bold;margin-bottom:8px;font-size:14px;">Letroso Solver</div>
                <input type="text" id="student_input" placeholder="Name to save scores" 
                    style="width:100%;padding:8px;background:#222;color:white;border:1px solid #777;border-radius:4px;margin-bottom:10px;box-sizing:border-box;">
                <button id="start_btn" style="width:100%;padding:10px;background:#28a745;color:white;border:none;border-radius:4px;cursor:pointer;font-weight:bold;">START GAME</button>
            `;
            var input = document.getElementById('student_input');
            input.onkeydown = function(e) {{ e.stopPropagation(); }};
            document.getElementById('start_btn').onclick = function() {{
                var val = input.value.trim();
                if(val !== "") window.current_user = val;
            }};
        }} else {{
            el.innerHTML = `
                <div style="font-weight:bold;color:#00ff88;font-size:14px;">{name} is playing👾</div>
                <div style="font-size:11px;color:#aaa;margin-bottom:10px;">Scores being saved.</div>
                <button id="end_btn" style="width:100%;padding:8px;background:#dc3545;color:white;border:none;border-radius:4px;cursor:pointer;font-weight:bold;">STOP GAME</button>
            `;
            document.getElementById('end_btn').onclick = function() {{
                window.current_user = "SIGNAL_EXIT_SESSION";
            }};
        }}
    """)

def remove_overlay(driver: webdriver.Firefox) -> None:
    driver.execute_script(f"var el = document.getElementById('{OVERLAY_ID}'); if (el) el.remove();")

def read_guess_word(driver: webdriver.Firefox, guess_index: int) -> str:
    board = driver.find_element(By.CSS_SELECTOR, ".board")
    guesses = board.find_elements(By.CSS_SELECTOR, ".guess")
    letters = guesses[guess_index].find_elements(By.CSS_SELECTOR, ".letter")
    return "".join(l.text.strip().lower() for l in letters)

def count_submitted_guesses(driver: webdriver.Firefox) -> int:
    try:
        board = driver.find_element(By.CSS_SELECTOR, ".board")
        guesses = board.find_elements(By.CSS_SELECTOR, ".guess")
        count = 0
        for g in guesses:
            letters = g.find_elements(By.CSS_SELECTOR, ".letter")
            if not letters: break
            first_cls = letters[0].get_attribute("class") or ""
            if set(first_cls.split()) - {"letter"}: count += 1
            else: break
        return count
    except: return 0

def wait_for_next_submission(driver: webdriver.Firefox, expected_index: int) -> bool:
    while True:
        if count_submitted_guesses(driver) > expected_index: return True
        time.sleep(0.3)

def show_recommendations(driver: webdriver.Firefox, ranked: list[tuple[str, float]], candidates: list[str], turn: int) -> None:
    n = len(candidates)
    lines = [f"{('►' if i == 0 else f'{i+1:2}.')} {w:<14} {e:.4f} bits" for i, (w, e) in enumerate(ranked)]
    lines.append(f"── {n:,} candidate{'s' if n != 1 else ''} remaining ──")
    inject_overlay(driver, lines, title=f"Turn {turn} — top {len(ranked)} suggestions")

def play(driver: webdriver.Firefox, vocab: list[str], weights: dict, first) -> tuple:
    """Original play logic modified only to return (opener, score)."""
    candidates = list(vocab)
    turn = 0
    opener = ""

    def _progress(done, total):
        pct = done / total * 100 if total else 0
        print(f"\r  Computing... {pct:.0f}%", end="", flush=True)

    while True:
        turn += 1
        if len(candidates) == 0:
            print("\nNo candidates remain.")
            return opener, 0

        if len(candidates) == 1:
            word = candidates[0]
            inject_overlay(driver, [f"► {word}   (only candidate)"], title=f"Turn {turn} — answer found!")
            print(f"\n  Only candidate: {word}")

        if turn == 1:
            print(f"\n  Type your first word in the browser and press Enter...")
        elif len(candidates) > 1:
            print(f"\n  Type your word in the browser and press Enter...")

        wait_for_next_submission(driver, turn - 1)
        time.sleep(1.2)

        guess = read_guess_word(driver, turn - 1)
        if turn == 1: opener = guess # Track first word

        feedback = read_feedback(driver, turn - 1)
        print(f"  You played:  {guess}")
        print(f"  Feedback:    {feedback}")

        pattern_int = parse_pattern(feedback, len(guess))
        if is_win_pattern(pattern_int, len(guess)):
            remove_overlay(driver)
            print(f"\n  Solved in {turn} turns! Answer: {guess}")
            return opener, turn

        candidates = filter_candidates(candidates, guess, pattern_int)
        print(f"  {len(candidates):,} candidates remaining")

        inject_overlay(driver, ["⏳ Computing recommendations..."], title=f"Turn {turn + 1}")

        t0 = time.time()

        ranked = rank_guesses(candidates, vocab, weights, top_n=10, progress_fn=_progress)
        print(f"\r  Done in {time.time()-t0:.1f}s            ")
        show_recommendations(driver, ranked, candidates, turn + 1)

def _wait_for_board_reset(driver: webdriver.Firefox) -> None:
    while count_submitted_guesses(driver) != 0:
        time.sleep(0.5)

def main() -> None:
    print("Loading vocabulary...")
    vocab, weights = load_data()
    first = load_first_guess()
    print("Opening browser...")
    driver = open_browser()

    current_session_id = None
    student_name = None

    try:
        while True:

            inject_session_ui(driver, name=student_name)
            
            while count_submitted_guesses(driver) == 0:
                user_signal = driver.execute_script("return window.current_user;")
                if user_signal == "SIGNAL_EXIT_SESSION":
                    student_name, current_session_id = None, None
                    driver.execute_script("window.current_user = null;")
                    inject_session_ui(driver, name=None)
                elif user_signal and user_signal != "SIGNAL_EXIT_SESSION" and not current_session_id:
                    student_name = user_signal
                    current_session_id = start_new_session(student_name)
                    print(f"Session {current_session_id} started: {student_name}")
                    inject_session_ui(driver, name=student_name)
                time.sleep(0.2)

            opener_word, score = play(driver, vocab, weights, first)


            if score > 0 and current_session_id:
                save_game_score(current_session_id, opener_word, score)
                print(f"Data saved for {student_name}")


            print("Waiting for next game reset...")
            while count_submitted_guesses(driver) > 0:
                user_signal = driver.execute_script("return window.current_user;")
                if user_signal == "SIGNAL_EXIT_SESSION":
                    student_name, current_session_id = None, None
                    driver.execute_script("window.current_user = null;")
                    inject_session_ui(driver, name=None)
                elif user_signal and not current_session_id:
                    student_name = user_signal
                    current_session_id = start_new_session(student_name)
                    inject_session_ui(driver, name=student_name)
                time.sleep(0.5)
            
            remove_overlay(driver)

    except KeyboardInterrupt:
        pass
    finally:
        driver.quit()

if __name__ == "__main__":
    main()

    