import logging
import os
from functools import partial
from typing import List, Callable

import hydra
import joblib
import nltk
import numpy as np
import pandas as pd
import re
import string
import unidecode
import warnings
from nltk.corpus import stopwords, twitter_samples
from omegaconf import DictConfig
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

nltk.download("twitter_samples")

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=UserWarning)


def logger(func: Callable) -> Callable:
    def wrapper(*args, **kwargs) -> None:
        log.info(f"Running {func.__name__}...")
        result = func(*args, **kwargs)
        log.info(f"{func.__name__} completed.")
        return result

    return wrapper


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


def pipeline_preprocess(text: List[str], process) -> List[str]:
    for proc in process:
        text = list(map(proc, text))
    return text


@logger
def build_pipeline(cfg: DictConfig, processing_pipeline: List[Callable]) -> Pipeline:
    log.info("Building pipeline...")

    prep_features = Pipeline(
        steps=[
            ("Preprocessing", FunctionTransformer(partial(pipeline_preprocess, process=processing_pipeline))),
            ("tfidf", TfidfVectorizer(
                min_df=cfg.train.TfidfVectorizer.min_df,
                max_df=cfg.train.TfidfVectorizer.max_df,
                ngram_range=tuple(cfg.train.TfidfVectorizer.ngram_range)
            )),
        ]
    )

    model = LogisticRegression(
        C=cfg.train.LogisticRegression.C,
        l1_ratio=cfg.train.LogisticRegression.l1_ratio,
        solver=cfg.train.LogisticRegression.solver,
        random_state=cfg.train.LogisticRegression.random_state,
        penalty=cfg.train.LogisticRegression.penalty,
    )
    clf = Pipeline(steps=[("preprocessor", prep_features), ("model", model)])

    log.info("Pipeline construction completed.")
    return clf


@logger
def train_model(
    model: BaseEstimator,
    X_train: np.ndarray,
    y_train: np.ndarray,
    param_grid: dict,
    n_cross_validation: int,
    scoring: str,
) -> BaseEstimator:
    log.info("Performing grid search for model training...")
    grid_search = GridSearchCV(model, param_grid, cv=n_cross_validation, scoring=scoring)
    grid_search.fit(X_train, y_train)

    log.info(f"Best score: {grid_search.best_score_}")

    return grid_search.best_estimator_


def save_output(output_data, path_to_output_dir, filename, logger=None) -> None:
    output_dir = os.path.join(path_to_output_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)

    if isinstance(output_data, pd.DataFrame):
        output_data.to_csv(output_path, index=False)
    elif isinstance(output_data, BaseEstimator):
        joblib.dump(output_data, output_path)
    else:
        raise ValueError("Unsupported output_data type.")

    if logger:
        logger.info(f"Output saved at: {output_path}")


@hydra.main(version_base=None, config_path="../config", config_name="config")
def process(cfg: DictConfig):
    log.info("Reading raw data...")

    positive_tweets = twitter_samples.strings("positive_tweets.json")
    negative_tweets = twitter_samples.strings("negative_tweets.json")

    train_data = pd.DataFrame(
        {
            "text": positive_tweets + negative_tweets,
            "target": [1] * len(positive_tweets) + [0] * len(negative_tweets),
        }
    )

    X_train, X_test, y_train, y_test = train_test_split(
        train_data.text.values,
        train_data.target.values,
        test_size=cfg.split.test_size,
        random_state=cfg.split.random_state,
    )
    inference_data = train_data.sample(5).text.values

    tokenizer = nltk.tokenize.RegexpTokenizer(r"\w+")
    preprocess = [preprocessing, tokenizer.tokenize, remove_stopwords, array_to_str]

    pipeline = build_pipeline(cfg, preprocess)

    best_estimator = train_model(
        pipeline,
        X_train,
        y_train,
        dict(cfg.train.param_grid),
        cfg.train.n_cross_validation,
        cfg.train.scoring,
    )

    path_to_output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    save_output(best_estimator, path_to_output_dir, cfg.path.to_save_model, logger=log)

    log.info("Evaluating model...")

    y_pred_train = best_estimator.predict(X_train)
    f1_score_train = f1_score(y_pred_train, y_train)
    log.info(f"Train f1-score: {f1_score_train}")

    y_pred_test = best_estimator.predict(X_test)
    f1_score_test = f1_score(y_pred_test, y_test)
    log.info(f"Test f1-score: {f1_score_test}")

    results = best_estimator.predict(inference_data)
    submission = pd.DataFrame({"text": inference_data, "target": results})

    save_output(submission, path_to_output_dir, cfg.path.to_save_predicts, logger=log)

    log.info("Process completed.")


if __name__ == "__main__":
    process()
