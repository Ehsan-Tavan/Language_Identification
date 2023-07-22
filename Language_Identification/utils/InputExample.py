from typing import Union, List


class InputExample:
    def __init__(self, text: str, chars: List[int] = None, label: Union[int, float] = None):
        self.text = text
        self.chars = chars
        self.label = label

    def __str__(self):
        return "<InputExample> label: {}, text: {}".format(str(self.label), str(self.text))
