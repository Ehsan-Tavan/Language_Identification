# ============================ Third Party libs ============================
import os
from sklearn.metrics import classification_report

# ============================ My packages ============================
from Language_Identification.configurations import BaseConfig
from Language_Identification.data_loader import read_csv, read_json
from Language_Identification.utils import prepare_example
from Language_Identification.models.lm_model import LmModel

if __name__ == "__main__":
    CONFIG_CLASS = BaseConfig()
    ARGS = CONFIG_CLASS.get_config()

    TEST_DATA = read_csv(os.path.join(ARGS.processed_data_dir, ARGS.test_data_file))
    LABEL2INDEX = read_json(os.path.join(ARGS.assets_dir, ARGS.label2index_file))
    TEST_DATA["labels"] = TEST_DATA["labels"].apply(lambda label: LABEL2INDEX[label])
    TEST_SAMPLES = prepare_example(TEST_DATA)

    MODEL_PATH = "../assets/saved_models/xlm-roberta-base/version_2/checkpoints/" \
                 "QTag-epoch=02-dev_loss=0.01.ckpt"
    MODEL = LmModel.load_from_checkpoint(MODEL_PATH).to(ARGS.device)

    probabilities, predicted_labels = MODEL.predict(TEST_SAMPLES,
                                                    batch_size=ARGS.inference_batch_size,
                                                    max_len=ARGS.max_len)

    report = classification_report(list(TEST_DATA["labels"]), predicted_labels.cpu(),
                                   target_names=LABEL2INDEX.keys(), digits=4)
    print(report)
