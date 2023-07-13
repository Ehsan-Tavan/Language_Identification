# ============================ Third Party libs ============================
from typing import List
import argparse
import torch
import pytorch_lightning as pl
import transformers

# ============================ My packages ============================
from Language_Identification.utils import InputExample


class DataModule(pl.LightningDataModule):
    def __init__(self,
                 train_data: List[InputExample],
                 test_data: List[InputExample],
                 dev_data: List[InputExample],
                 config: argparse.ArgumentParser.parse_args,
                 dataset_obj,
                 tokenizer: transformers.AutoTokenizer.from_pretrained,
                 ):
        super().__init__()
        self.config = config
        self.train_data = train_data
        self.dev_data = dev_data
        self.test_data = test_data
        self.dataset_obj = dataset_obj
        self.tokenizer = tokenizer
        self.dataset = {}

    def setup(self, stage=None) -> None:
        self.dataset["train_dataset"] = self.dataset_obj(
            data=self.train_data,
            tokenizer=self.tokenizer,
            max_len=self.config.max_len)
        self.dataset["dev_dataset"] = self.dataset_obj(
            data=self.dev_data,
            tokenizer=self.tokenizer,
            max_len=self.config.max_len)
        self.dataset["test_dataset"] = self.dataset_obj(
            data=self.test_data,
            tokenizer=self.tokenizer,
            max_len=self.config.max_len)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.dataset["train_dataset"],
                                           batch_size=self.config.train_batch_size,
                                           num_workers=self.config.num_workers,
                                           shuffle=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.dataset["dev_dataset"],
                                           batch_size=self.config.train_batch_size,
                                           num_workers=self.config.num_workers)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.dataset["test_dataset"],
                                           batch_size=self.config.train_batch_size,
                                           num_workers=self.config.num_workers)
