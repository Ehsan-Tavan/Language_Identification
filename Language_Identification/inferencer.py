# ============================ Third Party libs ============================
import os
# ============================ My packages ============================
from Language_Identification.configurations import BaseConfig
from Language_Identification.inference import Inference
from Language_Identification.data_loader import read_json
from Language_Identification.data_preparation import normalize_text
from Language_Identification.data_preparation import TokenIndexer

if __name__ == "__main__":
    CONFIG_CLASS = BaseConfig()
    ARGS = CONFIG_CLASS.get_config()

    MODEL_PATH = os.path.join(ARGS.saved_model_dir, ARGS.best_model_path)
    INDEX2LABEL = read_json(os.path.join(ARGS.assets_dir, ARGS.index2label_file))

    CHAR_INDEXER = TokenIndexer()
    CHAR_INDEXER.load(vocab2idx_path=os.path.join(ARGS.assets_dir, "char2idx.json"),
                      idx2vocab_path=os.path.join(ARGS.assets_dir, "idx2char.json"))

    INFERENCE = Inference(config=ARGS, model_path=MODEL_PATH, tokenizer_path=ARGS.lm_path,
                          index2label=INDEX2LABEL, char_indexer=CHAR_INDEXER)

    while True:
        text = input("please enter your text:\n")
        text = normalize_text(text)
        PROBABILITIES, PREDICTED_LABELS, LABELS = INFERENCE.get_language([text])
        print(f"Language of your text is {LABELS[0]}.")
