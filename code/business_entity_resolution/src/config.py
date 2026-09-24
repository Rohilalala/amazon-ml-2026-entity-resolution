"""Pipeline configuration.

The same object is saved inside the model bundle, so prediction always runs
with the blocking and decision settings the model was trained with.
"""
from dataclasses import asdict, dataclass, field


def _default_lgb_params():
    return {
        "objective": "binary",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_data_in_leaf": 40,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "deterministic": True,
        "force_row_wise": True,
        "verbose": -1,
    }


@dataclass(frozen=True)
class Config:
    # Blocking: nearest Source 2/3 records kept per Source 1 record (per source,
    # per similarity matrix), and nearest Source 1 records per Source 2/3 record.
    k: int = 15
    k_rev: int = 3
    block_by_country: bool = True
    # Training.
    folds: int = 5
    seed: int = 42
    num_boost_round: int = 3000
    early_stopping_rounds: int = 100
    lgb_params: dict = field(default_factory=_default_lgb_params)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**d)

    def lgb(self):
        """LightGBM parameters with the run seed applied."""
        return {**self.lgb_params, "seed": self.seed}
