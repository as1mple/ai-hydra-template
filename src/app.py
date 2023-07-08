import logging
from typing import Callable

import hydra
import nltk
import pandas as pd
import warnings
from nltk.corpus import twitter_samples
from omegaconf import DictConfig
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from modules.preprocessing import preprocessing, remove_stopwords, array_to_str
from modules.pipeline import pipeline_preprocess, build_pipeline
from modules.model import train_model, save_output


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


@hydra.main(version_base=None, config_path="../config", config_name="config")
def process(cfg: DictConfig):
    log.info("Reading raw data...")

    positive_tweets = twitter_samples.strings("positive_tweets.json")
    negative_tweets = twitter_samples.strings("negative_tweets.json")

    train_data = pd.DataFrame({
        "text": positive_tweets + negative_tweets,
        "target": [1] * len(positive_tweets) + [0] * len(negative_tweets),
    })

    X_train, X_test, y_train, y_test = train_test_split(
        train_data.text.values,
        train_data.target.values,
        test_size=cfg.split.test_size,
        random_state=cfg.split.random_state,
    )
    inference_data = train_data.sample(5).text.values

    tokenizer = nltk.tokenize.RegexpTokenizer(r"\w+")
    preprocess = [preprocessing, tokenizer.tokenize, remove_stopwords, array_to_str]

    log.info("Building pipeline...")
    pipeline = build_pipeline(cfg, preprocess, pipeline_preprocess)

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
    build_pipeline = logger(build_pipeline)
    train_model = logger(train_model)

    process()
