"""Pipeline configuration.

The same object is saved inside the model bundle, so prediction always runs
with the blocking and decision settings the model was trained with.
"""
from dataclasses import asdict, dataclass, field


def _default_lgb_params():
    return {
        "objective": "binary",
        "learning_rate": 0.08,
        "num_leaves": 127,
        "min_data_in_leaf": 100,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "max_bin": 255,
        "deterministic": True,
        "force_row_wise": True,
        "verbose": -1,
    }


@dataclass(frozen=True)
class Config:
    # Blocking: top-k Source 2 and top-k Source 3 records per Source 1 record,
    # top-k_rev Source 1 records per Source 2/3 record, and the document
    # frequency above which a blocking key is too common to use.
    k: int = 10
    k_rev: int = 3
    max_df: int = 1000
    # Training: Source 1 records sampled for the matcher (their candidates
    # come from the full pool), cross-validation folds, feature chunk size.
    train_s1_sample: int = 400_000
    folds: int = 5
    feature_chunk: int = 1_000_000
    seed: int = 42
    num_boost_round: int = 3000
    early_stopping_rounds: int = 100
    # Decision: each Source 2/3 record goes to its most likely Source 1 record
    # only (the training ground truth never assigns one record to two).
    exclusive: bool = True
    lgb_params: dict = field(default_factory=_default_lgb_params)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**d)

    def lgb(self):
        """LightGBM parameters with the run seed applied."""
        return {**self.lgb_params, "seed": self.seed}
