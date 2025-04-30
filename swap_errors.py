import pandas as pd
import numpy as np
from error_utils import update_error_record

def swap_fields(df, n_errors, col_names, error_record=None):
    if len(col_names) % 2 != 0:
        raise ValueError("col_names must be provided as pairs")
    df = df.copy()
    n = len(df)
    p = len(col_names) // 2
    errors_col = n_errors // p if p > 0 else n_errors
    if error_record is None:
        error_record = pd.DataFrame(columns=['id', 'field', 'error', 'before', 'after'])
    for i in range(0, len(col_names), 2):
        col_1, col_2 = col_names[i], col_names[i+1]
        rows = np.random.choice(n, size=errors_col, replace=False)
        c1 = df.loc[rows, col_1].copy()
        c2 = df.loc[rows, col_2].copy()
        df.loc[rows, col_1] = c2
        df.loc[rows, col_2] = c1
        before = [f"{c1.iloc[j]},{c2.iloc[j]}" for j in range(len(c1))]
        after = [f"{c2.iloc[j]},{c1.iloc[j]}" for j in range(len(c1))]
        df, error_record = update_error_record(df, df.loc[rows, 'id'], f"{col_1},{col_2}", 'swap', before, after, error_record)
    return df, error_record