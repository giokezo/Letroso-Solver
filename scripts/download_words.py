import json
import os
import re
import sys

import requests
from wordfreq import word_frequency

DATA_DIR    = os.path.join(os.path.dirname(__file__), "..","data")
OUTPUT_FILE = os.path.join(DATA_DIR, "words.json")

LETROSO_BASE = "https://letroso.com"
MAIN_JS_PATH = "/static/js/main.f2276900.js"

MIN_FREQUENCY = 1e-9
MIN_LENGTH    = 3
MAX_LENGTH    = 10

# Print a progress line every this many words during encoding
_PROGRESS_EVERY = 500


def fetch_main_bundle() -> str:
    url = LETROSO_BASE + MAIN_JS_PATH
    print(f"Fetching main bundle from {url} ...")
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.text


def find_word_chunk_url(bundle_text: str) -> str:
    chunk_map_match = re.search(
        r'\{(?:[0-9]+:"[0-9a-f]+"(?:,|(?=\})))+\}', bundle_text
    )
    if not chunk_map_match:
        raise RuntimeError("Could not locate chunk hash map in bundle.")

    raw_map   = re.sub(r'([{,])(\d+):', r'\1"\2":', chunk_map_match.group())
    chunk_map = json.loads(raw_map)

    en_chunk_match = re.search(r'"./all-en\.json":\[(\d+),\d+\]', bundle_text)
    if not en_chunk_match:
        raise RuntimeError("Could not find all-en.json module reference.")

    chunk_id   = en_chunk_match.group(1)
    chunk_hash = chunk_map.get(chunk_id)
    if chunk_hash is None:
        raise RuntimeError(f"No hash found for chunk ID {chunk_id}.")

    return f"{LETROSO_BASE}/static/js/{chunk_id}.{chunk_hash}.chunk.js"


def fetch_word_list(chunk_url: str) -> list[str]:
    print(f"Fetching word list from {chunk_url} ...")
    resp = requests.get(chunk_url, timeout=60)
    resp.raise_for_status()
    text = resp.text

    match = re.search(r'e\.exports=JSON\.parse\(\'(.*?)\'\)', text, re.DOTALL)
    if match:
        return json.loads(match.group(1))
    match = re.search(r'e\.exports=(\[".+?"\])', text, re.DOTALL)
    if not match:
        raise RuntimeError("Could not extract word array from chunk.")
    return json.loads(match.group(1))


def build_dataset(raw_words: list[str], lang: str = "en") -> dict[str, float]:
    """
    Frequency-encode and filter words, returning {word: frequency} sorted
    alphabetically.  Prints overall progress every _PROGRESS_EVERY words.
    """
    # Pre-filter by length so we don't waste wordfreq lookups
    candidates = [
        w for w in raw_words
        if MIN_LENGTH <= len(w) <= MAX_LENGTH
    ]
    total = len(candidates)
    print(f"Frequency-encoding {total:,} words ...")

    freq_map: dict[str, float] = {}
    floor = 1e-12

    for idx, word in enumerate(candidates, 1):
        freq = word_frequency(word, lang)
        f    = freq if freq > 0 else floor
        if f >= MIN_FREQUENCY:
            freq_map[word] = f

        if idx % _PROGRESS_EVERY == 0 or idx == total:
            pct = idx / total * 100
            print(f"  {pct:5.1f}%  ({idx:,}/{total:,})  kept: {len(freq_map):,}", flush=True)

    # Sort alphabetically
    return dict(sorted(freq_map.items()))


def main() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)

    try:
        bundle_text = fetch_main_bundle()
        chunk_url   = find_word_chunk_url(bundle_text)
        raw_words   = fetch_word_list(chunk_url)
    except Exception as exc:
        print(f"\n[WARNING] Could not fetch Letroso word list: {exc}")
        print("Falling back to NLTK words corpus ...\n")
        raw_words = _fallback_nltk_words()

    dataset = build_dataset(raw_words)

    print(f"\nSaving {len(dataset):,} words to {OUTPUT_FILE} ...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as fh:
        json.dump(dataset, fh, ensure_ascii=False)

    # Summary by length
    by_len: dict[int, int] = {}
    for w in dataset:
        by_len[len(w)] = by_len.get(len(w), 0) + 1
    print("\nDone! Words per length:")
    for length in sorted(by_len):
        print(f"  {length} letters: {by_len[length]:,}")


def _fallback_nltk_words() -> list[str]:
    try:
        import nltk  # type: ignore
        try:
            from nltk.corpus import words as nltk_words
            return [w.lower() for w in nltk_words.words() if w.isalpha()]
        except LookupError:
            nltk.download("words", quiet=True)
            from nltk.corpus import words as nltk_words
            return [w.lower() for w in nltk_words.words() if w.isalpha()]
    except ImportError:
        print("NLTK not available. Please install it: pip install nltk")
        sys.exit(1)


if __name__ == "__main__":
    main()
