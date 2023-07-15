# ============================ Third Party libs ============================
from typing import List
import torch

# ============================ My packages ============================
from Language_Identification.utils import InputExample


class Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 data: List[InputExample],
                 tokenizer,
                 max_len: int = None,
                 mode: str = "train"):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.mode = mode

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, item_index):
        sample = self.data[item_index]
        text = sample.text
        tokenized_sample = self.tokenizer.encode_plus(text=str(text),
                                                      max_length=self.max_len,
                                                      return_tensors="pt",
                                                      padding="max_length",
                                                      truncation=True)
        input_ids = tokenized_sample["input_ids"].flatten()
        attention_mask = tokenized_sample["attention_mask"].flatten()

        if self.mode in ["train", "test"]:
            label = sample.label
            label = torch.tensor(label)

            return {"features": {"input_ids": input_ids, "attention_mask": attention_mask},
                    "targets": label}
        return {"features": {"input_ids": input_ids, "attention_mask": attention_mask}}
