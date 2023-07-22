# ============================ Third Party libs ============================
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

CLASSIFIER2OBJECT = {"GaussianNB": GaussianNB(), "SVC": SVC()}
VECTORIZER2OBJECT = {"CountVectorizer": CountVectorizer(), "TfidfVectorizer": TfidfVectorizer()}
