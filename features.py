import numpy as np
import pandas as pd

RAW = ["nli_agrees", "nli_disagrees", "sent_negative", "sent_positive", "irony"]
NUMERIC = ["nli_agrees_logit", "nli_disagrees_logit", "sent_negative",
           "sent_positive", "irony", "neg_x_irony"]


def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def add_engineered(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["nli_agrees_logit"] = _logit(df["nli_agrees"])
    df["nli_disagrees_logit"] = _logit(df["nli_disagrees"])
    df["neg_x_irony"] = df["sent_negative"] * df["irony"]
    return df[NUMERIC]