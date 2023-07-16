# ============================ Third Party libs ============================
from typing import Type, List, Optional
import tqdm
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import CSVLogger
from transformers import AutoModel, AutoTokenizer
import torchmetrics
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

# ============================ My packages ============================
from Language_Identification.utils import InputExample
from Language_Identification.dataset import DataModule, Dataset
from .pooling_model import Pooling


class LmModel(pl.LightningModule):
    def __init__(self,
                 model_path: str,
                 config,
                 loss_fct,
                 num_classes,
                 train_data: List[InputExample],
                 dev_data: List[InputExample],
                 test_data: List[InputExample],
                 pooling_methods: List[str] = [],
                 optimizer: Type[torch.optim.Optimizer] = torch.optim.AdamW,
                 ):
        super().__init__()
        self.config = config
        self.model = AutoModel.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        self.optimizer = optimizer
        self.loss_fct = loss_fct
        self.learning_rate = self.config.lr

        self.train_data = train_data
        self.dev_data = dev_data
        self.test_data = test_data

        if len(pooling_methods) != 1:
            raise ValueError("Using only one type of pooling methods. ['mean', 'max', 'cls']")
        self.pooling_methods = pooling_methods
        self.pooling_model = Pooling()

        self.trainer = None
        self.data_module = None
        self.checkpoint_callback = None

        self.accuracy = torchmetrics.Accuracy(num_classes=num_classes)
        self.f_score = torchmetrics.F1(average="macro", num_classes=num_classes)

        self.convs = torch.nn.ModuleList([
            torch.nn.Conv2d(in_channels=1,
                            out_channels=self.config.n_filters,
                            kernel_size=(fs, self.model.config.hidden_size))
            for fs in self.config.filter_sizes
        ])
        self.max_pool = torch.nn.MaxPool1d(self.config.max_len)

        self.linear = torch.nn.Linear(self.model.config.hidden_size + 192, num_classes)

        self.save_hyperparameters()

    def forward(self, batch):
        token_output = self.model(**batch["token"], return_dict=True)
        token_output = \
            self.pooling_model(token_output.last_hidden_state, batch["token"]["attention_mask"],
                               pooling_methods=self.pooling_methods)[0]

        character_outputs = self.model(**batch["character"],
                                       return_dict=True).last_hidden_state.unsqueeze(1)

        character_outputs = [torch.nn.ReLU()(conv(character_outputs)).squeeze(3) for conv in
                             self.convs]
        # conved_n = [batch_size, n_filters, sent_len - filter_sizes[n] + 1]

        character_outputs = [torch.nn.functional.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv
                             in character_outputs]
        # pooled_n = [batch_size, n_filters]
        character_outputs = torch.cat(character_outputs, dim=1)

        output = torch.cat((token_output, character_outputs), dim=1)

        output = self.linear(output)
        return output

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

    def create_trainer(self,
                       check_point_monitor: Optional[str] = None,
                       check_point_mode: Optional[str] = None,
                       early_stopping_monitor: Optional[str] = None,
                       early_stopping_patience: Optional[int] = None):
        callbacks = []
        if check_point_monitor and check_point_mode:
            check_point_filename = "QTag-{epoch:02d}-{" + check_point_monitor + ":.2f}"
            self.checkpoint_callback = ModelCheckpoint(monitor=check_point_monitor,
                                                       filename=check_point_filename,
                                                       save_top_k=self.config.save_top_k,
                                                       mode=check_point_mode)
            callbacks.append(self.checkpoint_callback)
        if early_stopping_monitor and early_stopping_patience:
            early_stopping_callback = EarlyStopping(monitor=early_stopping_monitor,
                                                    patience=early_stopping_patience)
            callbacks.append(early_stopping_callback)

        callbacks.append(pl.callbacks.LearningRateMonitor(logging_interval="epoch"))

        logger = CSVLogger(self.config.saved_model_dir, name=self.config.model_name)

        trainer = pl.Trainer(max_epochs=self.config.n_epochs, gpus=[int(self.config.device[-1])],
                             callbacks=callbacks, min_epochs=10,
                             progress_bar_refresh_rate=60, logger=logger)
        self.data_module = DataModule(config=self.config,
                                      train_data=self.train_data, dev_data=self.dev_data,
                                      test_data=self.test_data, dataset_obj=Dataset,
                                      tokenizer=self.tokenizer)

        return trainer

    def fit(self,
            check_point_monitor: Optional[str] = None,
            check_point_mode: Optional[str] = None,
            early_stopping_monitor: Optional[str] = None,
            early_stopping_patience: Optional[int] = None
            ):
        self.trainer = self.create_trainer(check_point_monitor=check_point_monitor,
                                           check_point_mode=check_point_mode,
                                           early_stopping_monitor=early_stopping_monitor,
                                           early_stopping_patience=early_stopping_patience)

        self.trainer.fit(self, datamodule=self.data_module)

    def test(self):
        self.trainer.test(self, datamodule=self.data_module)

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
                               mode="test")
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        with torch.no_grad():
            for batch in tqdm.tqdm(test_loader):
                batch["features"]["token"]["input_ids"] = batch["features"]["token"][
                    "input_ids"].to(self.config.device)
                batch["features"]["token"]["attention_mask"] = batch["features"]["token"][
                    "attention_mask"].to(self.config.device)
                batch["features"]["character"]["input_ids"] = batch["features"]["character"][
                    "input_ids"].to(self.config.device)
                batch["features"]["character"]["attention_mask"] = batch["features"]["character"][
                    "attention_mask"].to(self.config.device)
                logits = self.forward(batch["features"])
                probabilities.append(torch.softmax(logits, dim=-1))
                labels.append(torch.argmax(logits, dim=-1))

        probabilities = torch.cat(probabilities, dim=0)
        labels = torch.cat(labels, dim=0)
        return probabilities, labels
