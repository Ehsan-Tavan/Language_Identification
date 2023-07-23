from typing import Union, List


class InputExample:
    def __init__(self, text: str, chars: List[int] = None, token_length: List[str] = None,
                 label: Union[int, float] = None):
        self.text = text
        self.chars = chars
        self.token_length = token_length
        self.label = label

    def __str__(self):
        return "<InputExample> label: {}, text: {}".format(str(self.label), str(self.text))
