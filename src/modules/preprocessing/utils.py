import re
import string
from typing import List
import unidecode

from nltk.corpus import stopwords, twitter_samples


def preprocessing(text: str) -> str:
    lowercase_text = text.lower()
    text_without_diacritics = unidecode.unidecode(lowercase_text)
    text_without_brackets = re.sub(r"\[.*?\]", "", text_without_diacritics)
    text_without_urls = re.sub(r"https?://\S+|www\.\S+", "", text_without_brackets)
    text_without_tags = re.sub(r"<.*?>+", "", text_without_urls)
    text_without_punctuation = re.sub(r"[{}]".format(re.escape(string.punctuation)), "", text_without_tags)
    text_without_newlines = re.sub(r"\n", "", text_without_punctuation)
    text_without_digits = re.sub(r"\w*\d\w*", "", text_without_newlines)

    return text_without_digits


def remove_stopwords(text: List[str]) -> List[str]:
    stop_words = stopwords.words("english")
    return [w for w in text if w not in stop_words]


def array_to_str(text: List[str]) -> str:
    return " ".join(text)

