# ============================ Third Party libs ============================
from typing import List
# =============================== My packages ==============================
from .indexer import Indexer


class TokenIndexer(Indexer):
    def __init__(self, vocabs: list = None, pad_index: int = 0, unk_index: int = 1):
        super().__init__(vocabs)
        self.pad_index = pad_index
        self.unk_index = unk_index

        self._vocab2idx = None
        self._idx2vocab = None

    def get_idx(self, token: str) -> int:
        if not self._vocab2idx:
            self._empty_vocab_handler()
        if token in self._vocab2idx:
            return self._vocab2idx[token]
        return self._vocab2idx["<UNK>"]

    def get_word(self, idx: int) -> str:
        if not self._idx2vocab:
            self._empty_vocab_handler()
        if idx in self._idx2vocab:
            return self._idx2vocab[idx]
        return self._idx2vocab[self.unk_index]

    def build_vocab2idx(self):
        self._vocab2idx = {"<PAD>": self.pad_index, "<UNK>": self.unk_index}
        for vocab in self.vocabs:
            self._vocab2idx[vocab] = len(self._vocab2idx)

    def build_idx2vocab(self):
        self._idx2vocab = {self.pad_index: "<PAD>", self.unk_index: "<UNK>"}
        for vocab in self.vocabs:
            self._idx2vocab[len(self._idx2vocab)] = vocab

    def convert_samples_to_token_indexes(self, tokenized_samples: List[list]) -> List[list]:
        for index, tokenized_sample in enumerate(tokenized_samples):
            token_indexes = []
            for token_index, token in enumerate(tokenized_sample):
                token_indexes.append(self.get_idx(token))
            tokenized_samples[index] = token_indexes
        return tokenized_samples
