import pickle
from collections.abc import Iterable, Iterator
import regex as re
from cs336_basics.constants import GPT2_REGEX_PAT

class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab: dict[int, bytes] = vocab
        self.reverse_vocab: dict[bytes, int] = {v: k for (k, v) in vocab.items()}
        self.merges: list[tuple[bytes, bytes]] = merges
        self.special_tokens: list[str] | None = sorted(special_tokens, key=len, reverse=True) if special_tokens else None

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)
        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)
        return cls(vocab, merges, special_tokens)
    
    def encode(self, text: str) -> list[int]:
        btext = text.encode("utf-8")
        special_token_idx = []
        if self.special_tokens is not None:
            split_special_token = b"|".join(re.escape(s).encode("utf-8") for s in self.special_tokens)
            found = re.finditer(split_special_token, btext)
            for fd in found:
                special_token_idx.append((fd.start(), fd.end()))
        special_token_idx.append((len(btext), len(btext)))
        start = 0
        ids = []
        for (special_token_start, special_token_end) in special_token_idx:
            end = special_token_start
            chunk = btext[start:end]
            print(f"text len: {len(btext)},  processing chunk: {start} to {end}")
            tokens = re.finditer(GPT2_REGEX_PAT, chunk)
            for token in tokens:
                token_list = list(bytes([t]) for t in token.group())
                while True:
                    merge_idx = []
                    for i in range(len(token_list) - 1):
                        pair = (token_list[i], token_list[i + 1])
                        if pair in self.merges:
                            merge_idx.append((i, self.merges.index(pair)))
                    if len(merge_idx) == 0:
                        break
                    i = min(merge_idx, key=lambda x: x[1])[0]
                    new_token = token_list[i] + token_list[i + 1]
                    token_list = token_list[:i] + [new_token] + token_list[i + 2:]
                for t in token_list:
                    ids.append(self.reverse_vocab[t])
            special_token = btext[special_token_start:special_token_end]
            if len(special_token) != 0:
                ids.append(self.reverse_vocab[special_token])
            start = special_token_end
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for it in iterable:
            for token in self.encode(it):
                yield token

    def decode(self, ids: list[int]) -> str:
        tokens = b"".join([(self.vocab[id] if id in self.vocab else "\uFFFD".encode("utf-8")) for id in ids])
        return tokens.decode("utf-8", errors="replace")

if __name__ == "__main__":
    # Test for Example (bpe_encoding)
    # vocab = {0: b' ', 1: b'a', 2: b'c', 3: b'e', 4: b'h', 5: b't', 6: b'th', 7: b' c', 8: b' a', 9: b'the', 10: b' at'}
    # merges = [(b't', b'h'), (b' ', b'c'), (b' ', b'a'), (b'th', b'e'), (b' a', b't')]
    # tokenizer = Tokenizer(vocab, merges)
    # ids = tokenizer.encode("the cat ate")
    # s = tokenizer.decode(ids)

    # Test for Problem (tokenizer_experiments)
    with open(r"results/bpe_tinystories.pkl", "rb") as f:
        data = pickle.load(f)
        tinystories_vocab, tinystories_merges = data["vocab"], data["merges"]
    with open(r"results/bpe_expts_owt.pkl", "rb") as f:
        data = pickle.load(f)
        owt_vocab, owt_merges = data["vocab"], data["merges"]
    tinystories_tokenizer = Tokenizer(tinystories_vocab, tinystories_merges, ['<|endoftext|>'])
    owt_tokenizer = Tokenizer(owt_vocab, owt_merges, ['<|endoftext|>'])
    s = ""
    # with open(r"tests/fixtures/tinystories_sample.txt", "r", encoding="utf-8") as f:
    #     s += f.read()
    # with open(r"tests/fixtures/expts_owt_sample.txt", "r", encoding="utf-8") as f:
    #     s += f.read()
    with open(r"tests/fixtures/tinystories_sample_5M.txt", "r", encoding="utf-8") as f:
        s += f.read()
    import time
    start_time = time.time()
    tinystories_ids = tinystories_tokenizer.encode(s)
    end_time = time.time()
    print(f"tinystories encoding time: {end_time - start_time} seconds")
    start_time = time.time()
    owt_ids = owt_tokenizer.encode(s)
    end_time = time.time()
    print(f"owt encoding time: {end_time - start_time} seconds")
    print()
    print(f"input bytes: {len(s.encode("utf-8"))}")
    print()
    print(f"tinystories_ids(len {len(tinystories_ids)})")
    print()
    print(f"owt_ids(len {len(owt_ids)})")

    # Encode train and valid datasets
    # with open(r"results/bpe_tinystories.pkl", "rb") as f:
    #     data = pickle.load(f)
    #     tinystories_vocab, tinystories_merges = data["vocab"], data["merges"]
    # with open(r"results/bpe_expts_owt.pkl", "rb") as f:
    #     data = pickle.load(f)
    #     owt_vocab, owt_merges = data["vocab"], data["merges"]
    # tinystories_tokenizer = Tokenizer(tinystories_vocab, tinystories_merges, ['<|endoftext|>'])
    # owt_tokenizer = Tokenizer(owt_vocab, owt_merges, ['<|endoftext|>'])
    # s = ""
    # import numpy as np
    # with open(r"data/TinyStories/TinyStories-train.txt", "r", encoding="utf-8") as f:
    #     s = f.read()
    #     ids = tinystories_tokenizer.encode(s)
    #     arr = np.array(ids, dtype=np.uint16)
    #     np.save(r"results/token_ids/TinyStories-train.npy", arr)
    # with open(r"data/TinyStories/TinyStories-valid.txt", "r", encoding="utf-8") as f:
    #     s = f.read()
    #     ids = tinystories_tokenizer.encode(s)
    #     arr = np.array(ids, dtype=np.uint16)
    #     np.save(r"results/token_ids/TinyStories-valid.npy", arr)