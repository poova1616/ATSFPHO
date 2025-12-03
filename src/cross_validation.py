import pandas as pd

def rolling_cv(df, train_size, horizon, step=None):
    """Generator for rolling-origin (expanding window) cross-validation.

    Args:
        df (pd.DataFrame): dataframe with columns ['ds','y'] sorted by ds.
        train_size (int): initial number of observations to use for training.
        horizon (int): number of periods to forecast in each fold.
        step (int or None): step size to move the window forward (default = horizon).

    Yields:
        train_df, val_df (both pd.DataFrame)
    """
    if step is None:
        step = horizon

    n = len(df)
    i = train_size
    while i + horizon <= n:
        train = df.iloc[:i].reset_index(drop=True)
        val = df.iloc[i:i+horizon].reset_index(drop=True)
        yield train, val
        i += step
