# ============================ Third Party libs ============================
import re
from string import punctuation
import demoji


# def remove_emoji(input_text: str):
#     emoji_list = [ch for ch in input_text if ch in emoji.UNICODE_EMOJI]
#     clean_text = " ".join(
#         [word for word in input_text.split() if not any(i in word for i in emoji_list)])
#     return clean_text


def remove_emoji(string):
    return demoji.replace(string, repl="!")


def normalize_text(input_text: str) -> str:
    # Words are converted to lowercase format.
    normalized_text = input_text.lower()

    # Digits are removed from text.
    normalized_text = "".join((ch for ch in normalized_text if not ch.isdigit()))

    # Emojis are removed from text.
    normalized_text = remove_emoji(normalized_text)

    # Punctuations are removed.
    normalized_text = re.sub(f'[{punctuation}؟،٪×÷»«]+', '', normalized_text)
    return normalized_text
