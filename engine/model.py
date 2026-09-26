"""The classifier behind P(stock goes up); swap it here."""
from lightgbm import LGBMClassifier


def make_model(params: dict) -> LGBMClassifier:
    return LGBMClassifier(**params)
