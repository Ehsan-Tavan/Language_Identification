# ============================ Third Party libs ============================
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

CLASSIFIER2OBJECT = {"GaussianNB": GaussianNB(), "SVC": SVC()}
VECTORIZER2OBJECT = {"CountVectorizer": CountVectorizer(), "TfidfVectorizer": TfidfVectorizer()}

LABEL2LANGUAGE = {"ar": "arabic", "bg": "bulgarian", "de": "german", "el": "modern greek",
                  "en": "english", "es": "spanish", "fr": "french", "hi": "hindi", "it": "italian",
                  "ja": "japanese", "nl": "dutch", "pl": "polish", "pt": "portuguese",
                  "ru": "russian", "sw": "swahili", "th": "thai", "tr": "turkish", "ur": "urdu",
                  "vi": "vietnamese", "zh": "chinese"}
