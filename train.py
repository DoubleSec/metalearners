from xgboost import XGBClassifier
from sklearn.preprocessing import TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import polars as pl


def setup_data(data_path: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Read data, construct the treatment variable and the target."""

    raw_data = pl.read_csv(data_path)

    data = (
        raw_data.with_columns(
            **{
                "grad-degree": (pl.col("educational-num") > 13).cast(pl.Int32),
                "income": pl.col("income") == ">50K",
            }
        )
        # Randomly re-order
        .sample(fraction=1.0, shuffle=False, seed=7700)
    )
    x_train = data[: int(data.shape[0] * 0.75)]
    x_valid = data[int(data.shape[0] * 0.75) :]
    return x_train, x_valid


def train_model(x_train_: pl.DataFrame, x_valid_: pl.DataFrame, cfg: dict) -> Pipeline:
    """Takes data and config and trains an xgboost model according to a very simple recipe.

    Returns the trained pipeline and the validation set."""

    # Split into training and validation, 75/25
    x_train = x_train_[cfg["categorical"] + cfg["numeric"]]
    x_valid = x_valid_[cfg["categorical"] + cfg["numeric"]]
    y_train = x_train_[cfg["target"]]
    y_valid = x_valid_[cfg["target"]]

    encoder = ColumnTransformer(
        transformers=[
            (
                "target_encoding",
                TargetEncoder(target_type="binary"),
                cfg["categorical"],
            ),
            ("passthrough", "passthrough", cfg["numeric"]),
        ]
    )
    encoder.set_output(transform="polars")

    # Awkward, but to have early stopping with the encoder.
    x_train_tr = encoder.fit_transform(x_train, y_train)
    x_valid_tr = encoder.transform(x_valid)

    s_learner = XGBClassifier(**cfg["learner_params"])
    eval_set = [(x_train_tr, y_train), (x_valid_tr, y_valid)]
    s_learner.fit(x_train_tr, y_train, eval_set=eval_set, verbose=10)

    return Pipeline([("encoder", encoder), ("model", s_learner)])
