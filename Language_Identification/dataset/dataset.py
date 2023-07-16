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

        character_input = self.tokenizer.convert_tokens_to_ids(list(str(text)))
        character_input = character_input[:self.max_len-2]
        character_input = self.tokenizer.build_inputs_with_special_tokens(character_input)
        char_attention_mask = [1] * len(character_input)

        character_input += [self.tokenizer.pad_token_id] * (self.max_len - len(character_input))

        char_attention_mask += [0] * (self.max_len - len(char_attention_mask))
        character_input = torch.tensor(character_input)
        char_attention_mask = torch.tensor(char_attention_mask)

        if self.mode in ["train", "test"]:
            label = sample.label
            label = torch.tensor(label)

            return {
                "features": {"token": {"input_ids": input_ids, "attention_mask": attention_mask},
                             "character": {"input_ids": character_input,
                                           "attention_mask": char_attention_mask}},
                "targets": label}
        return {"features": {"token": {"input_ids": input_ids, "attention_mask": attention_mask},
                             "character": {"input_ids": character_input,
                                           "attention_mask": char_attention_mask}}}
