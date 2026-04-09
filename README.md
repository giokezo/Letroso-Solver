
# Letroso Solver

A solver for Letroso (letroso.com/en/unlimited), the Spanish Wordle variant that uses LCS-based pattern matching instead of exact positional matching. Words can be different lengths, and consecutive matching letters show as concatenated runs in the feedback.

Three modes are available: a terminal solver you interact with manually, a fully automated Selenium bot, and a semi-manual Selenium mode where you type words yourself but the solver reads feedback and shows recommendations.

---

## How the feedback works

Letroso uses LCS (Longest Common Subsequence) matching rather than exact position matching. Each letter in your guess gets one of four states:

- B (black): letter not in the answer at all
- Y (yellow): letter is in the answer but not part of the LCS alignment
- G (green): letter is part of the LCS alignment
- P (border): green letter that sits at the start or end of the answer
- O (concat): appears between two consecutive greens to show they are adjacent in both the guess and the answer

Example: guessing "kettle" against "beetle" could produce BGBGOGOP, meaning the second t, l, and e form a consecutive run (the OGO between them), with the final e at the right boundary (P).

---

## Setup

Clone the repo and install dependencies:

    pip install -r requirements.txt

You also need geckodriver installed and on your PATH for the Selenium modes.

Download the word list from Letroso and assign frequency scores:

    python scripts/download_words.py

This fetches the word list directly from Letroso's JS bundle and scores each word using the wordfreq library. The result is saved to data/words.json.

Optionally precompute the best opening guess (takes a while, uses most of your CPU cores):

    python scripts/precompute.py

This evaluates every 10-letter word as a potential first guess and saves the best one to data/first_guesses.json. The solver will use this automatically if the file exists. Without it the solver just asks you to enter the first word yourself.

---

## Usage

### Terminal solver

    python solver.py

The solver asks you to enter your guess and the feedback string after each turn. Feedback is entered as a sequence of B/G/Y/P characters with O between consecutive greens, for example BGBGOGOP. The solver then shows the top 10 candidates ranked by entropy.

After each game it asks if you want to play again.

### Automated Selenium bot

    python selenium_solver.py

Opens a Firefox window, navigates to Letroso, and plays the game fully automatically. It reads feedback from the DOM after each guess and types the next recommended word. Results are logged to data/solve_log.csv.

### Manual Selenium mode

    python manual_selenium_solver.py

Opens a Firefox window and navigates to Letroso. You type words yourself in the browser. After each submission, Selenium reads the feedback automatically and shows the top 10 recommendations both in the terminal and as a floating overlay panel on the page. You pick which word to type next.

If only one candidate remains, the overlay keeps showing it until you actually type it correctly, so a mistype does not clear the recommendation.

---

## Project structure

    engine.py                   core algorithm (pattern matching, entropy, filtering)
    solver.py                   terminal solver
    selenium_solver.py          fully automated bot
    manual_selenium_solver.py   manual play with Selenium-assisted feedback
    scripts/
        download_words.py       fetch word list from Letroso and score frequencies
        precompute.py           find best opening guess using multiprocessing
    data/
        words.json              word list with frequency scores
        first_guesses.json      precomputed best opening guess
        solve_log.csv           game history logged by the automated solver

---

## Engine

engine.py is the core module. You can import it directly if you want to build on top of it.

### Constants

    BLACK  = 0
    YELLOW = 1
    GREEN  = 2
    BORDER = 3

    FREQ_ALPHA: float = 1.0

FREQ_ALPHA controls how much word frequency influences the entropy calculation. At 1.0 frequencies are used as-is. Higher values make common words dominate more (2.0 is a reasonable starting point). Lower values flatten toward uniform. The solver also scales this up automatically as the candidate pool shrinks, so at endgame it effectively just picks the most common remaining word.

### Pattern encoding

Patterns are encoded as a base-8 integer. Each position contributes one digit:

    value = state * 2 + concat_right

Where state is 0-3 (BLACK/YELLOW/GREEN/BORDER) and concat_right is 1 if this position is concatenated with the next one, 0 otherwise. Position 0 is the least significant digit.

### Key functions

    get_pattern(guess: str, answer: str) -> int

Computes the pattern integer for a given guess against a given answer. This is the core LCS computation. Results are cached with lru_cache (up to 8 million entries). The cache is shared across calls within a process, so the first game is slower and subsequent games are faster as the cache warms up.

    parse_pattern(feedback: str, length: int) -> int | None

Parses a human-readable feedback string like "BGBGOGOP" into a pattern integer. Accepts B/G/Y/P/O as tokens. Returns None if the string is invalid.

    decode_pattern(pattern_int: int, length: int) -> list[tuple[int, bool]]

Decodes a pattern integer back into a list of (state, concat_right) tuples, one per position.

    is_win_pattern(pattern_int: int, length: int) -> bool

Returns True if the pattern represents a complete match (all positions green/border, fully concatenated from start to end).

    filter_candidates(candidates: list[str], guess: str, pattern_int: int) -> list[str]

Filters a list of candidate words, keeping only those where get_pattern(guess, word) equals the observed pattern. This is how the candidate pool is narrowed after each guess.

    rank_guesses(candidates, vocab, weights, top_n=10, progress_fn=None) -> list[tuple[str, float]]

Ranks candidate words by entropy and returns the top N. Uses multiprocessing with one pool per call. Each candidate is evaluated as a potential guess against the full candidate pool. Words are weighted by frequency (shaped by FREQ_ALPHA). Returns a list of (word, entropy_bits) tuples sorted descending.

    compute_entropy(guess: str, candidates: list[str], weights: dict) -> float

Computes the Shannon entropy of a single guess against the candidate pool. Used for single-threaded scenarios.

### Entropy calculation

For a given guess, the candidate pool is partitioned into buckets by feedback pattern. Each bucket has a probability equal to the sum of frequency weights of the candidates that fall into it. Entropy is:

    H = -sum(p * log2(p) for each bucket)

A higher entropy means the guess splits the candidate pool more evenly, giving more information. The solver always picks the guess with the highest entropy among the remaining candidates.

---

## Tuning

FREQ_ALPHA in engine.py is the main knob. The effect scales with the number of remaining candidates because _dynamic_alpha in engine.py bumps it up at smaller pool sizes:

    >= 1000 candidates: FREQ_ALPHA as set
    200-999 candidates: FREQ_ALPHA + 1
    50-199 candidates:  FREQ_ALPHA + 2
    < 50 candidates:    FREQ_ALPHA + 3

So with FREQ_ALPHA = 1.0, the endgame alpha is 4.0, meaning a word twice as frequent gets 2^4 = 16x the weight. With FREQ_ALPHA = 2.0 it would be 2^5 = 32x.
