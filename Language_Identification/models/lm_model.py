# ============================ Third Party libs ============================
from typing import Type, List, Optional
import tqdm
import torch
import pytorch_lightning as pl
from transformers import AutoModel, AutoTokenizer, MT5EncoderModel, MT5Tokenizer
import torchmetrics
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

# ============================ My packages ============================
from Language_Identification.utils import InputExample
from Language_Identification.dataset import Dataset
from .pooling_model import Pooling


class Classifier(pl.LightningModule):
    def __init__(self,
                 config,
                 loss_fct,
                 num_classes: int,
                 num_characters: int = None,
                 pooling_methods: List[str] = None,
                 optimizer: Type[torch.optim.Optimizer] = torch.optim.AdamW):
        super().__init__()
        if pooling_methods is None:
            pooling_methods = []
        self.config = config
        self.model = MT5EncoderModel.from_pretrained(self.config.lm_path)
        self.tokenizer = MT5Tokenizer.from_pretrained(self.config.lm_path)

        self.optimizer = optimizer
        self.loss_fct = loss_fct
        self.learning_rate = self.config.lr

        if len(pooling_methods) != 1:
            raise ValueError("Using only one type of pooling methods. ['mean', 'max', 'cls']")
        self.pooling_methods = pooling_methods
        self.pooling_model = Pooling()

        self.trainer = None
        self.data_module = None
        self.checkpoint_callback = None

        self.accuracy = torchmetrics.Accuracy(num_classes=num_classes)
        self.f_score = torchmetrics.F1(average="macro", num_classes=num_classes)

        if self.config.use_char:
            self.embedding_layer = torch.nn.Embedding(num_characters,
                                                      self.config.char_embedding_dim)

            self.convs = torch.nn.ModuleList([
                torch.nn.Conv2d(in_channels=1,
                                out_channels=self.config.n_filters,
                                kernel_size=(fs, self.config.char_embedding_dim))
                for fs in self.config.filter_sizes
            ])
            self.max_pool = torch.nn.MaxPool1d(self.config.max_len)
            self.char_classifier = torch.nn.Linear(192, num_classes)
            self.using_char_threshold = self.config.using_char_threshold

        if self.config.use_token_length:
            self.token_length_classifier = torch.nn.Linear(self.model.config.hidden_size,
                                                           num_classes)

        self.lm_classifier = torch.nn.Linear(self.model.config.hidden_size, num_classes)

        self.save_hyperparameters()

    def forward(self, batch):
        token_length_output, character_outputs = None, None
        token_output = self.model(**batch["token"], return_dict=True)
        token_output = \
            self.pooling_model(token_output.last_hidden_state, batch["token"]["attention_mask"],
                               pooling_methods=self.pooling_methods)[0]
        token_output = self.lm_classifier(token_output)

        if self.config.use_token_length:
            token_length_output = self.model(**batch["token_length"], return_dict=True)
            token_length_output = \
                self.pooling_model(token_length_output.last_hidden_state,
                                   batch["token_length"]["attention_mask"],
                                   pooling_methods=self.pooling_methods)[0]
            token_length_output = self.token_length_classifier(token_length_output)

        if self.config.use_char:
            char_embedding = self.embedding_layer(batch["character"]["input_ids"]).unsqueeze(1)

            character_outputs = [torch.nn.ReLU()(conv(char_embedding)).squeeze(3) for conv in
                                 self.convs]
            # conved_n = [batch_size, n_filters, sent_len - filter_sizes[n] + 1]

            character_outputs = [torch.nn.functional.max_pool1d(conv, conv.shape[2]).squeeze(2) for
                                 conv in character_outputs]
            # pooled_n = [batch_size, n_filters]
            character_outputs = torch.cat(character_outputs, dim=1)
            character_outputs = self.char_classifier(character_outputs)

        pred = self._prepare_output(token_output, token_length_output, character_outputs)
        return pred

    def _prepare_output(self, token_output, token_length_output=None, character_outputs=None):
        if self.config.use_char and not self.config.use_token_length:
            character_pred = torch.nn.Softmax(dim=1)(character_outputs)
            token_pred = torch.nn.Softmax(dim=1)(token_output)
            pred = token_pred * (
                    1 - self.using_char_threshold) + character_pred * self.using_char_threshold
            return torch.log(pred)
        elif self.config.use_char and self.config.use_token_length:
            character_pred = torch.nn.Softmax(dim=1)(character_outputs)
            token_pred = torch.nn.Softmax(dim=1)(token_output)
            token_length_pred = torch.nn.Softmax(dim=1)(token_length_output)
            pred = token_pred * (
                    1 - (self.using_char_threshold + self.using_token_length_threshold)) + \
                   character_pred * self.using_char_threshold + \
                   token_length_pred * self.config.using_token_length_threshold
            return torch.log(pred)
        elif not self.config.use_char and self.config.use_token_length:
            token_length_pred = torch.nn.Softmax(dim=1)(token_length_output)
            token_pred = torch.nn.Softmax(dim=1)(token_output)
            pred = token_pred * (1 - self.config.using_token_length_threshold) + \
                   token_length_pred * self.config.using_token_length_threshold
            return torch.log(pred)
        else:
            return token_output

    def training_step(self, batch, _):
        targets = batch["targets"].flatten()
        predicted_labels = self.forward(batch["features"])
        loss = self.loss_fct(predicted_labels, targets)
        metric2value = {"train_loss": loss,
                        "train_acc":
                            self.accuracy(torch.softmax(predicted_labels, dim=1), targets),
                        "train_F1":
                            self.f_score(torch.softmax(predicted_labels, dim=1), targets)}

        self.log_dict(metric2value, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss}

    def validation_step(self, batch, _):
        targets = batch["targets"].flatten()
        predicted_labels = self.forward(batch["features"])
        loss = self.loss_fct(predicted_labels, targets)
        metric2value = {"dev_loss": loss,
                        "dev_acc":
                            self.accuracy(torch.softmax(predicted_labels, dim=1), targets),
                        "dev_F1":
                            self.f_score(torch.softmax(predicted_labels, dim=1), targets)
                        }

        self.log_dict(metric2value, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, _):
        targets = batch["targets"].flatten()
        predicted_labels = self.forward(batch["features"])
        loss = self.loss_fct(predicted_labels, targets)
        metric2value = {"test_loss": loss,
                        "test_acc":
                            self.accuracy(torch.softmax(predicted_labels, dim=1), targets),
                        "test_F1":
                            self.f_score(torch.softmax(predicted_labels, dim=1), targets)
                        }

        self.log_dict(metric2value, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), lr=self.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.1,
                                      verbose=True)
        lr_scheduler = {
            "scheduler": scheduler,
            "monitor": "dev_loss",  # Metric to monitor for learning rate reduction
            "interval": "epoch",
            "frequency": 1
        }

        return [optimizer], [lr_scheduler]

    def predict(self,
                test_data: List[InputExample],
                batch_size: int,
                max_len: Optional[int] = None):
        probabilities = []
        labels = []
        self.eval()
        self.model.to(self.config.device)

        test_dataset = Dataset(data=test_data, tokenizer=self.tokenizer, max_len=max_len,
                               mode="test", device=self.config.device)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        with torch.no_grad():
            for batch in tqdm.tqdm(test_loader):
                logits = self.forward(batch["features"])
                probabilities.append(torch.softmax(logits, dim=-1))
                labels.append(torch.argmax(logits, dim=-1))

        probabilities = torch.cat(probabilities, dim=0)
        labels = torch.cat(labels, dim=0)
        return probabilities, labels
