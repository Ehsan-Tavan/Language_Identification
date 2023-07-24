# ============================ Third Party libs ============================
from typing import List
import transformers

# ============================ My packages ============================
from Language_Identification.models.lm_model import Classifier
from Language_Identification.utils import LABEL2LANGUAGE, InputExample, calculate_token_length
from Language_Identification.data_preparation import TokenIndexer


class Inference:
    def __init__(self,
                 config,
                 model_path: str,
                 tokenizer_path: str,
                 index2label: dict,
                 char_indexer: TokenIndexer):
        self.config = config
        self.index2label = index2label
        self.model = None
        self.tokenizer = None
        self.load_model(model_path)
        self.load_tokenizer(tokenizer_path)

        self.char_indexer = char_indexer

    def load_model(self, path: str) -> None:
        self.model = Classifier.load_from_checkpoint(path).to(self.config.device)

    def load_tokenizer(self, path: str) -> None:
        self.tokenizer = transformers.MT5Tokenizer.from_pretrained(path)

    @staticmethod
    def prepare_token_length(texts: List[str]):
        token_length = calculate_token_length(texts)
        return token_length

    def prepare_sentence_char(self, texts: List[str]):
        chars = [[char for char in str(sample)] for sample in texts]
        chars = self.char_indexer.convert_samples_to_token_indexes(chars)
        return chars

    def convert_sample_to_input_example(self, texts: List[str]):
        examples = []
        token_lengths = self.prepare_token_length(texts)
        chars = self.prepare_sentence_char(texts)

        for index, sample in enumerate(texts):
            example = InputExample(text=sample,
                                   chars=chars[index],
                                   token_length=token_lengths[index])
            examples.append(example)
        return examples

    def get_language(self, texts: List[str]):
        labels = []
        examples = self.convert_sample_to_input_example(texts)
        probabilities, predicted_labels = self.model.predict(
            examples,
            batch_size=self.config.inference_batch_size,
            max_len=self.config.max_len,
            mode="inference")
        predicted_labels = predicted_labels.detach().cpu().numpy()
        for index, label in enumerate(predicted_labels):
            labels.append(LABEL2LANGUAGE[self.index2label[str(label)]])
        return probabilities, predicted_labels, labels
