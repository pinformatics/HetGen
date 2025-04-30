import pandas as pd
import numpy as np
import random
import re
from error_utils import update_error_record

def first_letter_abbreviate(df, n_errors, col_names, error_record=None):
    df = df.copy()
    n = len(df)
    p = len(col_names)
    errors_col = n_errors // p if p > 0 else n_errors
    if error_record is None:
        error_record = pd.DataFrame(columns=['id', 'field', 'error', 'before', 'after'])
    if n_errors < n * p and errors_col > 0:
        for col_name in col_names:
            rows = np.random.choice(n, size=errors_col, replace=False)
            before = df.loc[rows, col_name]
            after = before.str[0]
            df.loc[rows, col_name] = after
            df, error_record = update_error_record(df, df.loc[rows, 'id'], col_name, 'first_letter_abbreviate', before, after, error_record)
    else:
        for _ in range(n_errors):
            row = random.randint(0, n-1)
            col_name = random.choice(col_names)
            before = df.loc[row, col_name]
            after = str(before)[0] if before else ''
            df.loc[row, col_name] = after
            df, error_record = update_error_record(df, [df.loc[row, 'id']], col_name, 'first_letter_abbreviate', [before], [after], error_record)
    return df, error_record

def blanks_to_hyphens(df, n_errors, col_names, all=True, error_record=None):
    return ch1_to_ch2(df, n_errors, col_names, ' ', '-', all, error_record)

def hyphens_to_blanks(df, n_errors, col_names, all=True, error_record=None):
    return ch1_to_ch2(df, n_errors, col_names, '-', ' ', all, error_record)

def ch1_to_ch2(df, n_errors, col_names, ch1, ch2, all=True, error_record=None):
    df = df.copy()
    n = len(df)
    p = len(col_names)
    errors_col = n_errors // p if p > 0 else n_errors
    org_colnames = df.columns
    if error_record is None:
        error_record = pd.DataFrame(columns=['id', 'field', 'error', 'before', 'after'])
    for col_name in col_names:
        pattern = r'[a-zA-Z0-9]' + re.escape(ch1) + r'[a-zA-Z0-9]'
        feasible_recs = df[df[col_name].str.contains(pattern, na=False, regex=True)]
        n_to_sample = min(errors_col, len(feasible_recs))
        if n_to_sample == 0:
            print(f"Not enough records with '{ch1}' found in column {col_name}")
            continue
        feasible_recs = feasible_recs.sample(n=n_to_sample, random_state=None)
        feasible_recs['blanks'] = True
        feasible_recs = feasible_recs[['id', 'blanks']].astype({'blanks': bool})
        df = df.merge(feasible_recs, on='id', how='left')
        df['old_names'] = df[col_name].astype(str)
        if all:
            df[col_name] = np.where(df['blanks'].fillna(False),
                                   df[col_name].str.replace(re.escape(ch1), ch2, regex=True),
                                   df[col_name]).astype(str)
        else:
            df[col_name] = np.where(df['blanks'].fillna(False),
                                   df[col_name].str.replace(re.escape(ch1), ch2, n=1, regex=True),
                                   df[col_name]).astype(str)
        error_table = df[df['blanks'].fillna(False)]
        if not error_table.empty:
            df, error_record = update_error_record(df, error_table['id'], col_name, f'{ch1}to{ch2}',
                                                  error_table['old_names'], error_table[col_name], error_record)
        df = df[org_colnames]
    return df, error_record

def make_missing(df, n_errors, col_names, error_record=None):
    df = df.copy()
    n = len(df)
    p = len(col_names)
    errors_col = n_errors // p if p > 0 else n_errors
    if error_record is None:
        error_record = pd.DataFrame(columns=['id', 'field', 'error', 'before', 'after'])
    if n_errors < n * p and errors_col > 0:
        for col_name in col_names:
            ids = df['id'].astype(str)
            prev_errors = error_record[error_record['field'] == col_name]['id'].astype(str).unique() if not error_record.empty else []
            id_pool = np.setdiff1d(ids, prev_errors)
            if len(id_pool) < errors_col:
                print(f"Not enough unique IDs for make_missing on {col_name}. Using available: {len(id_pool)}")
                errors_col = len(id_pool)
            if errors_col == 0:
                continue
            candidate_ids = np.random.choice(id_pool, size=errors_col, replace=False)
            before = df.loc[df['id'].isin(candidate_ids), col_name]
            df.loc[df['id'].isin(candidate_ids), col_name] = np.nan
            df, error_record = update_error_record(df, candidate_ids, col_name, 'missing', before, [''] * len(before), error_record)
    else:
        for _ in range(n_errors):
            row = random.randint(0, n-1)
            col_name = random.choice(col_names)
            before = df.loc[row, col_name]
            df.loc[row, col_name] = np.nan
            df, error_record = update_error_record(df, [df.loc[row, 'id']], col_name, 'missing', [before], [''], error_record)
    return df, error_record