# ============================ Third Party libs ============================
from typing import Type, List, Optional
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import CSVLogger
from transformers import AutoModel, AutoTokenizer
import torchmetrics
from torch.optim.lr_scheduler import ReduceLROnPlateau

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

        self.linear = torch.nn.Linear(self.model.config.hidden_size, num_classes)

        self.save_hyperparameters()

    def forward(self, batch):
        output = self.model(**batch, return_dict=True)
        output = self.pooling_model(output.last_hidden_state, batch["attention_mask"],
                                    pooling_methods=self.pooling_methods)
        output = self.linear(output[0])
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
