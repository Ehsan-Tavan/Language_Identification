# ============================ Third Party libs ============================
from typing import List, Optional
import torch

# ============================ My packages ============================
from Language_Identification.utils import InputExample


class Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 data: List[InputExample],
                 tokenizer,
                 max_len: int = 50,
                 mode: str = "train",
                 device: str = "cpu"):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.mode = mode
        self.device = device

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, item_index):
        sample = self.data[item_index]
        text = sample.text
        chars = sample.chars
        token_length = sample.token_length
        tokenized_sample = self.tokenizer.encode_plus(text=str(text),
                                                      max_length=self.max_len,
                                                      return_tensors="pt",
                                                      padding="max_length",
                                                      truncation=True)

        input_ids = tokenized_sample["input_ids"].flatten()
        attention_mask = tokenized_sample["attention_mask"].flatten()

        tokenized_token_length = self.tokenizer.encode_plus(text=token_length,
                                                            is_split_into_words=True,
                                                            max_length=self.max_len,
                                                            return_tensors="pt",
                                                            padding="max_length",
                                                            truncation=True)

        token_length_input_ids = tokenized_token_length["input_ids"].flatten()
        token_length_attention_mask = tokenized_token_length["attention_mask"].flatten()

        chars = chars[:5 * self.max_len]
        chars_attention_mask = [1] * len(chars)
        chars_attention_mask += [0] * ((5 * self.max_len) - len(chars_attention_mask))
        chars += [0] * ((5 * self.max_len) - len(chars))
        chars = torch.tensor(chars)
        chars_attention_mask = torch.tensor(chars_attention_mask)

        if self.mode in ["train", "test"]:
            label = sample.label
            label = torch.tensor(label)

            return {
                "features": {"token": {"input_ids": input_ids.to(self.device),
                                       "attention_mask": attention_mask.to(self.device)},
                             "token_length": {"input_ids": token_length_input_ids.to(self.device),
                                              "attention_mask": token_length_attention_mask.to(
                                                  self.device)},
                             "character": {"input_ids": chars.to(self.device),
                                           "attention_mask": chars_attention_mask.to(self.device)}},
                "targets": label.to(self.device)}
        return {"features": {"token": {"input_ids": input_ids.to(self.device),
                                       "attention_mask": attention_mask.to(self.device)},
                             "token_length": {"input_ids": token_length_input_ids.to(self.device),
                                              "attention_mask": token_length_attention_mask.to(
                                                  self.device)},
                             "character": {"input_ids": chars.to(self.device),
                                           "attention_mask": chars_attention_mask.to(self.device)}}}
