import re
import string
from typing import List
from unidecode import unidecode

from nltk.corpus import stopwords, twitter_samples


def preprocessing(text: str) -> str:
    regex_patterns = {
        "Remove content inside brackets": r"\[.*?\]",
        "Remove URLs": r"https?://\S+|www\.\S+",
        "Remove HTML tags": r"<.*?>+",
        "Remove punctuation": r"[{}]".format(re.escape(string.punctuation)),
        "Remove newlines": r"\n",
        "Remove digits": r"\w*\d\w*"
    }
    
    for description, pattern in regex_patterns.items():
        text = re.sub(pattern, "", text)
    
    text = unidecode(text.lower())
    return text


def remove_stopwords(text: List[str]) -> List[str]:
    stop_words = stopwords.words("english")
    return [w for w in text if w not in stop_words]


def array_to_str(text: List[str]) -> str:
    return " ".join(text)

