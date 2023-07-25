# ============================ Third Party libs ============================
from typing import List, Optional
import pytorch_lightning as pl
from transformers import AutoTokenizer, MT5Tokenizer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import CSVLogger

# ============================ My packages ============================
from Language_Identification.utils import InputExample
from Language_Identification.dataset import DataModule, Dataset


class Trainer(pl.LightningModule):
    def __init__(self,
                 model_path: str,
                 config,
                 classifier,
                 train_data: List[InputExample],
                 dev_data: List[InputExample],
                 test_data: List[InputExample],
                 ):
        super().__init__()
        self.config = config
        self.tokenizer = MT5Tokenizer.from_pretrained(model_path)

        self.train_data = train_data
        self.dev_data = dev_data
        self.test_data = test_data

        self.trainer = None
        self.data_module = None
        self.checkpoint_callback = None

        self.classifier = classifier

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

        self.data_module = DataModule(config=self.config,
                                      train_data=self.train_data, dev_data=self.dev_data,
                                      test_data=self.test_data, dataset_obj=Dataset,
                                      tokenizer=self.tokenizer)

        self.trainer = pl.Trainer(max_epochs=self.config.n_epochs,
                                  gpus=[int(self.config.device[-1])],
                                  callbacks=callbacks, min_epochs=10,
                                  progress_bar_refresh_rate=60, logger=logger)

    def fit(self,
            check_point_monitor: Optional[str] = None,
            check_point_mode: Optional[str] = None,
            early_stopping_monitor: Optional[str] = None,
            early_stopping_patience: Optional[int] = None
            ):
        self.create_trainer(check_point_monitor=check_point_monitor,
                            check_point_mode=check_point_mode,
                            early_stopping_monitor=early_stopping_monitor,
                            early_stopping_patience=early_stopping_patience)

        self.trainer.fit(self.classifier, datamodule=self.data_module)

    def test(self):
        self.trainer.test(self.classifier, datamodule=self.data_module)
