# ============================ Third Party libs ============================
import os
from abc import abstractmethod
from typing import List

# =============================== My packages ==============================
from Language_Identification.data_loader import write_json, read_json


# ==========================================================================


class Indexer:
    def __init__(self, vocabs: list = None):
        self.vocabs = vocabs
        self._vocab2idx = None
        self._idx2vocab = None
        if self.vocabs:
            self._unitize_vocabs()  # build unique vocab

    def get_vocab2idx(self) -> dict:
        if not self._vocab2idx:
            self._empty_vocab_handler()
        return self._vocab2idx

    def get_idx2vocab(self) -> dict:
        if not self._idx2vocab:
            self._empty_vocab_handler()
        return self._idx2vocab

    @abstractmethod
    def get_idx(self, token: str) -> int:
        if not self._vocab2idx:
            self._empty_vocab_handler()
        if token in self._vocab2idx:
            return self._vocab2idx[token]
        print("error handler")
        raise Exception("target is not available")

    @abstractmethod
    def get_word(self, idx: int) -> str:
        if not self._idx2vocab:
            self._empty_vocab_handler()
        if idx in self._idx2vocab:
            return self._idx2vocab[idx]

        print("error handler")
        raise Exception("target is not available")

    @abstractmethod
    def build_vocab2idx(self) -> None:
        self._vocab2idx = {}
        for vocab in self.vocabs:
            self._vocab2idx[vocab] = len(self._vocab2idx)

    @abstractmethod
    def build_idx2vocab(self) -> None:
        self._idx2vocab = {}
        for vocab in self.vocabs:
            self._idx2vocab[len(self._idx2vocab)] = vocab

    def _empty_vocab_handler(self):
        if not self.vocabs:
            raise Exception("Vocabs is None")
        self.build_vocab2idx()
        self.build_idx2vocab()

    def convert_samples_to_indexes(self, tokenized_samples: List[list]) -> List[list]:
        for index, tokenized_sample in enumerate(tokenized_samples):
            for token_index, token in enumerate(tokenized_sample):
                tokenized_samples[index][token_index] = self.get_idx(token)
        return tokenized_samples

    def convert_indexes_to_samples(self, indexed_samples: List[list]) -> List[list]:
        for index, indexed_sample in enumerate(indexed_samples):
            for token_index, token in enumerate(indexed_sample):
                indexed_samples[index][token_index] = self.get_word(token)
        return indexed_samples

    def _unitize_vocabs(self) -> None:
        self.vocabs = list(set(self.vocabs))

    def load(self, vocab2idx_path: str, idx2vocab_path: str):
        self._vocab2idx = read_json(path=vocab2idx_path)
        self._idx2vocab = read_json(path=idx2vocab_path)

    def save(self, vocab2idx_path: str, idx2vocab_path: str) -> None:
        write_json(data=self.get_vocab2idx(), path=vocab2idx_path)
        write_json(data=self.get_idx2vocab(), path=idx2vocab_path)