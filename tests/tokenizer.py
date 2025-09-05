import pickle
from collections.abc import Iterable, Iterator
import regex as re
from .constants import GPT2_REGEX_PAT

class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab: dict[int, bytes] = vocab
        self.reverse_vocab: dict[int, bytes] = {v: k for (k, v) in vocab.items()}
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
            tokens = re.finditer(GPT2_REGEX_PAT, chunk)
            for token in tokens:
                token_list = list(bytes([t]) for t in token.group())
                for merge in self.merges:
                    i = 0
                    while True:
                        if i >= len(token_list) - 1:
                            break
                        if token_list[i] == merge[0] and token_list[i+1] == merge[1]:
                            token_list[i] += token_list[i+1]
                            del token_list[i+1]
                        else:
                            i += 1
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
    vocab = {0: b' ', 1: b'a', 2: b'c', 3: b'e', 4: b'h', 5: b't', 6: b'th', 7: b' c', 8: b' a', 9: b'the', 10: b' at'}
    merges = [(b't', b'h'), (b' ', b'c'), (b' ', b'a'), (b'th', b'e'), (b' a', b't')]
    tokenizer = Tokenizer(vocab, merges)
    ids = tokenizer.encode("the cat ate")
    s = tokenizer.decode(ids)