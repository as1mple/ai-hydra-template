from functools import partial
from typing import List, Callable

from omegaconf import DictConfig
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer


def pipeline_preprocess(text: List[str], process) -> List[str]:
    for proc in process:
        text = list(map(proc, text))
    return text


def build_pipeline(cfg: DictConfig, processing_pipeline: List[Callable], pipeline_function: Callable) -> Pipeline:
    prep_features = Pipeline(
        steps=[
            ("Preprocessing", FunctionTransformer(partial(pipeline_function, process=processing_pipeline))),
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

    return clf