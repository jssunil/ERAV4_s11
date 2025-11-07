"""Train a custom BPE tokenizer for Telugu text.

The script reads `data/telugu_corpus.txt`, learns a BPE model with a target
vocabulary larger than 5000 tokens, evaluates its compression ratio on the
training corpus, and saves useful artefacts for inspection.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


TARGET_VOCAB_SIZE = 6000
MAX_MERGES = 12000
CORPUS_PATH = Path(__file__).resolve().parents[1] / "data" / "telugu_corpus.txt"
MERGES_PATH = Path(__file__).resolve().parents[1] / "data" / "telugu_bpe_merges.txt"
VOCAB_PATH = Path(__file__).resolve().parents[1] / "data" / "telugu_bpe_vocab.txt"


Pair = Tuple[str, str]
Word = Tuple[str, ...]


def clean_and_tokenize(text: str) -> List[str]:
    """Return whitespace-delimited Telugu words from raw text."""

    # Keep Telugu letters and whitespace. Drop other characters (numbers, Latin).
    filtered = re.sub(r"[^\u0C00-\u0C7F\s]", " ", text)
    normalized = re.sub(r"\s+", " ", filtered).strip()
    if not normalized:
        return []
    return normalized.split()


def build_initial_vocab(words: Iterable[str]) -> Counter[Word]:
    """Map each word (as tuple of symbols) to its frequency."""

    vocab: Counter[Word] = Counter()
    for word, freq in Counter(words).items():
        symbols = tuple(list(word) + ["</w>"])
        vocab[symbols] = freq
    return vocab


def get_pair_stats(vocab: Counter[Word]) -> Dict[Pair, int]:
    """Compute frequency of symbol pairs across the vocabulary."""

    stats: Dict[Pair, int] = defaultdict(int)
    for symbols, freq in vocab.items():
        for idx in range(len(symbols) - 1):
            pair = (symbols[idx], symbols[idx + 1])
            stats[pair] += freq
    return stats


def merge_pair(pair: Pair, word: Word) -> Word:
    """Merge occurrences of `pair` inside `word`."""

    merged: List[str] = []
    i = 0
    while i < len(word):
        if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
            merged.append(word[i] + word[i + 1])
            i += 2
        else:
            merged.append(word[i])
            i += 1
    return tuple(merged)


@dataclass
class BPEModel:
    merges: List[Pair]
    vocab: List[str]

    def encode_word(self, word: str) -> List[str]:
        symbols: List[str] = list(word) + ["</w>"]
        if not symbols:
            return []

        merge_ranks = {pair: idx for idx, pair in enumerate(self.merges)}

        def get_pairs(sequence: Sequence[str]) -> set[Pair]:
            return {(sequence[i], sequence[i + 1]) for i in range(len(sequence) - 1)}

        while True:
            pairs = get_pairs(symbols)
            if not pairs:
                break

            ranked_pairs = {pair: merge_ranks.get(pair, math.inf) for pair in pairs}
            best_pair = min(ranked_pairs, key=ranked_pairs.get)
            if ranked_pairs[best_pair] is math.inf:
                break

            new_symbols: List[str] = []
            i = 0
            while i < len(symbols):
                if (
                    i < len(symbols) - 1
                    and symbols[i] == best_pair[0]
                    and symbols[i + 1] == best_pair[1]
                ):
                    new_symbols.append(symbols[i] + symbols[i + 1])
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            symbols = new_symbols

        if symbols and symbols[-1] == "</w>":
            symbols = symbols[:-1]
        return symbols


def learn_bpe(vocab: Counter[Word]) -> BPEModel:
    merges: List[Pair] = []
    symbol_set = {symbol for word in vocab for symbol in word}

    for iteration in range(MAX_MERGES):
        if len(symbol_set) >= TARGET_VOCAB_SIZE:
            break

        pair_stats = get_pair_stats(vocab)
        if not pair_stats:
            break

        best_pair, frequency = max(pair_stats.items(), key=lambda item: item[1])
        if frequency < 2 and len(symbol_set) >= TARGET_VOCAB_SIZE:
            break

        vocab = Counter({merge_pair(best_pair, word): freq for word, freq in vocab.items()})
        merges.append(best_pair)
        new_symbol = best_pair[0] + best_pair[1]
        symbol_set.add(new_symbol)

        if (iteration + 1) % 200 == 0:
            print(
                f"[Iteration {iteration + 1}] symbols={len(symbol_set)} last_pair={best_pair}"
            )

    print(f"Finished training after {len(merges)} merges; vocab size={len(symbol_set)}.")

    return BPEModel(merges=merges, vocab=sorted(symbol_set))


def evaluate_bpe(model: BPEModel, words: Sequence[str]) -> Tuple[int, int, float]:
    total_characters = sum(len(word) for word in words)
    encoded_token_count = 0

    for word in words:
        encoded = model.encode_word(word)
        encoded_token_count += len(encoded)

    if encoded_token_count == 0:
        raise RuntimeError("Encoding produced zero tokens.")

    compression_ratio = total_characters / encoded_token_count
    return total_characters, encoded_token_count, compression_ratio


def save_artifacts(model: BPEModel) -> None:
    MERGES_PATH.write_text(
        "\n".join([f"{left} {right}" for left, right in model.merges]), encoding="utf-8"
    )
    VOCAB_PATH.write_text("\n".join(model.vocab), encoding="utf-8")


def main() -> None:
    if not CORPUS_PATH.exists():
        raise FileNotFoundError(f"Cannot find corpus at {CORPUS_PATH}")

    raw_text = CORPUS_PATH.read_text(encoding="utf-8")
    words = clean_and_tokenize(raw_text)
    if not words:
        raise RuntimeError("Corpus is empty after cleaning.")

    vocab = build_initial_vocab(words)
    model = learn_bpe(vocab)
    save_artifacts(model)

    total_chars, tokens, ratio = evaluate_bpe(model, words)

    print(f"Unique tokens (including </w>): {len(model.vocab)}")
    print(f"Characters encoded: {total_chars}")
    print(f"Tokens produced: {tokens}")
    print(f"Compression ratio (chars per token): {ratio:.2f}")
    print("Top 20 merges:")
    for left, right in model.merges[:20]:
        print(f"  {left} + {right} -> {left + right}")


if __name__ == "__main__":
    main()

