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


def _default_ce_params():
    """Pair encoder (stage 4): fine-tuning selection and settings, and the inference batch."""
    return {
        # Fine-tuned from this checkpoint (MIT licence); only the shipped weights are read at predict time.
        "base": "intfloat/multilingual-e5-small@614241f622f53c4eeff9890bdc4f31cfecc418b3",
        "max_length": 128,
        "train_s1": 360_000,   # training Source 1 records outside the matcher's sample and holdout
        "val_s1": 20_000,      # of those, kept out of fitting for the validation check
        "top_k": 6,            # candidates per record by retrieval score, plus every true pair
        "batch_size": 128,
        "lr": 3e-5,
        "warmup": 0.05,
        "weight_decay": 0.01,
        "seed": 0,
        "infer_batch": 1024,
        "device": "auto",      # "auto" (Apple GPU, else CUDA, else CPU) or "cpu"
    }


@dataclass(frozen=True)
class Config:
    # Blocking: top-k Source 2 and top-k Source 3 records per Source 1 record,
    # top-k_rev Source 1 records per Source 2/3 record, and the document
    # frequency above which a blocking key is too common to use.
    k: int = 10
    k_rev: int = 3
    max_df: int = 3000
    # Training: Source 1 records sampled for the matcher (their candidates
    # come from the full pool), cross-validation folds, feature chunk size.
    train_s1_sample: int = 500_000
    # Source 1 records kept out of both stages and scored exactly like the test
    # set: an unbiased estimate of the leaderboard (per country).
    holdout_s1: int = 100_000
    folds: int = 5
    feature_chunk: int = 1_000_000
    seed: int = 42
    num_boost_round: int = 3000
    # Stage 2 also sees how each pair's stage-1 probability compares with its
    # competitors on both sides; stage 3 compares each candidate with the other
    # likely members of its Source 1 record's group (groupfeat.py).
    stages: int = 3
    early_stopping_rounds: int = 100
    # Decision: each Source 2/3 record goes to its most likely Source 1 record
    # only (the training ground truth never assigns one record to two).
    exclusive: bool = True
    # Countries without training labels (France): "keep nothing" may be up-weighted
    # by at most this factor to match the seen countries' empty-prediction share.
    unseen_none_cap: float = 4.0
    # Fixed empty-share target for that calibration (None = the seen countries' share in
    # the same run). Pinning it keeps an unseen country's decision independent of
    # predict-time changes to the seen countries (training-pool vocabulary tables).
    unseen_empty_target: float | None = None
    # Label-free corrections after the decision for countries without training labels
    # (unseen_rules.py): legal-form conflict drops, true-variant word and name-tail adds.
    unseen_rules: bool = False
    # Which probabilities decide: "full" (stage 3), "robust" (stage 3 without the
    # shared-deviation group features) or "none" (stage 2), with per-country
    # overrides. Stage 3 lets a candidate's likely siblings vouch for it. That helps
    # where test data looks like training (India), but US test decoy businesses come
    # as copy groups (18x the training rate) that vouch for each other, and France
    # is unseen, so both are decided on stage 2.
    stage3_use: str = "none"
    stage3_by_country: dict = field(default_factory=lambda: {"india": "full"})
    # Predict-time clips on stage-1/2 feature columns, e.g. {"amb_s23_addr_2": 1.0},
    # applied to pairs of feature_cap_countries (empty = every country). The test
    # pool's decoy copy groups inflate this ambiguity count into false
    # copy-consensus; the cap restores its training-time meaning.
    feature_caps: dict = field(default_factory=dict)
    feature_cap_countries: list = field(default_factory=list)
    # Stage 4 (stage4.py): a LightGBM stacker over the decided probability and a fine-tuned
    # pair encoder's logit on the raw text of pairs whose probability reaches stage4_active.
    # Scope: "none", "seen" (countries with training labels) or "all" (unseen countries
    # then go through the calibration and unseen_rules on the stage-4 probability).
    stage4_scope: str = "none"
    stage4_active: float = 0.002
    # Stacker feature set ("A" or "ACNRST", stage4.FEATURE_SETS) and LightGBM seeds: one model per
    # seed, probabilities averaged (the leaderboard run: ACNRST, seeds 42, 7, 99).
    stage4_features: str = "A"
    stage4_seeds: list = field(default_factory=lambda: [42])
    ce_params: dict = field(default_factory=_default_ce_params)
    lgb_params: dict = field(default_factory=_default_lgb_params)
    stage3_lgb_params: dict = field(default_factory=lambda: {"learning_rate": 0.05, "num_leaves": 63})

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**d)

    def lgb(self, stage: int = 1):
        """LightGBM parameters for a stage, with the run seed applied."""
        extra = self.stage3_lgb_params if stage == 3 else {}
        return {**self.lgb_params, **extra, "seed": self.seed}
