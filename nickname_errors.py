import pandas as pd
import numpy as np
from data_loader import NAMES_LOOKUP, NICK_REAL_LOOKUP
from error_utils import update_error_record

def real_to_nicknames(df, n_errors, col_names, error_record=None):
    if NAMES_LOOKUP is None:
        raise ValueError("NAMES_LOOKUP data not provided")
    if not all(col in NAMES_LOOKUP.columns for col in ['lookup_name', 'lookup_alternate', 'lookup_type']):
        raise ValueError("NAMES_LOOKUP missing required columns: ['lookup_name', 'lookup_alternate', 'lookup_type']")
    df = df.copy()
    n = len(df)
    p = len(col_names)
    errors_col = n_errors // p if p > 0 else n_errors
    org_colnames = df.columns
    if error_record is None:
        error_record = pd.DataFrame(columns=['id', 'field', 'error', 'before', 'after'])
    for col_name in col_names:
        lookup = NAMES_LOOKUP[NAMES_LOOKUP['lookup_type'] == 'to_nick']
        lookup = lookup.merge(df[[col_name, 'id']].rename(columns={col_name: 'lookup_name'}),
                             on='lookup_name')
        if lookup.empty:
            print(f"Warning: No matching nicknames found for column {col_name}")
            continue
        lookup = lookup.groupby('lookup_name').sample(n=1, random_state=None).reset_index(drop=True)
        n_to_sample = min(errors_col, len(lookup))
        if n_to_sample < errors_col:
            print("Not enough matches found for nicknames. Using all available matches.")
        lookup = lookup.sample(n=n_to_sample, random_state=None)[['lookup_alternate', 'id']]
        df = df.merge(lookup, on='id', how='left')
        df['old_names'] = df[col_name]
        df[col_name] = np.where(df['lookup_alternate'].notna(), df['lookup_alternate'], df[col_name])
        error_table = df[df['lookup_alternate'].notna()]
        if not error_table.empty:
            df, error_record = update_error_record(df, error_table['id'], col_name, 'to_nickname',
                                                  error_table['old_names'], error_table[col_name], error_record)
        df = df[org_colnames]
    return df, error_record

def nick_to_realnames(df, n_errors, col_names, error_record=None):
    if NAMES_LOOKUP is None:
        raise ValueError("NAMES_LOOKUP data not provided")
    if not all(col in NAMES_LOOKUP.columns for col in ['lookup_name', 'lookup_alternate', 'lookup_type']):
        raise ValueError("NAMES_LOOKUP missing required columns: ['lookup_name', 'lookup_alternate', 'lookup_type']")
    df = df.copy()
    n = len(df)
    p = len(col_names)
    errors_col = n_errors // p if p > 0 else n_errors
    org_colnames = df.columns
    if error_record is None:
        error_record = pd.DataFrame(columns=['id', 'field', 'error', 'before', 'after'])
    for col_name in col_names:
        lookup = NAMES_LOOKUP[NAMES_LOOKUP['lookup_type'] == 'to_proper']
        lookup = lookup.merge(df[[col_name, 'id']].rename(columns={col_name: 'lookup_name'}),
                             on='lookup_name')
        if lookup.empty:
            print(f"Warning: No matching real names found for column {col_name}")
            continue
        lookup = lookup.groupby('lookup_name').sample(n=1, random_state=None).reset_index(drop=True)
        n_to_sample = min(errors_col, len(lookup))
        if n_to_sample < errors_col:
            print("Not enough matches found for realnames. Using all available matches.")
        lookup = lookup.sample(n=n_to_sample, random_state=None)[['lookup_alternate', 'id']]
        df = df.merge(lookup, on='id', how='left')
        df['old_names'] = df[col_name]
        df[col_name] = np.where(df['lookup_alternate'].notna(), df['lookup_alternate'], df[col_name])
        error_table = df[df['lookup_alternate'].notna()]
        if not error_table.empty:
            df, error_record = update_error_record(df, error_table['id'], col_name, 'to_realname',
                                                  error_table['old_names'], error_table[col_name], error_record)
        df = df[org_colnames]
    return df, error_record

def invert_real_and_nicknames(df, n_errors, col_names, error_record=None):
    if NICK_REAL_LOOKUP is None:
        raise ValueError("NICK_REAL_LOOKUP data not provided")
    if not all(col in NICK_REAL_LOOKUP.columns for col in ['key', 'lookup']):
        raise ValueError("NICK_REAL_LOOKUP missing required columns: ['key', 'lookup']")
    df = df.copy()
    n = len(df)
    p = len(col_names)
    errors_col = n_errors // p if p > 0 else n_errors
    org_colnames = df.columns
    if error_record is None:
        error_record = pd.DataFrame(columns=['id', 'field', 'error', 'before', 'after'])
    for col_name in col_names:
        lookup = NICK_REAL_LOOKUP.merge(df[[col_name, 'id']].rename(columns={col_name: 'key'}),
                                       on='key')
        if lookup.empty:
            print(f"Warning: No matching names found for column {col_name} in NICK_REAL_LOOKUP")
            continue
        lookup = lookup.groupby('key').sample(n=1, random_state=None).reset_index(drop=True)
        n_to_sample = min(errors_col, len(lookup))
        if n_to_sample < errors_col:
            print("Not enough matches found for nick or realnames. Using all available matches.")
        lookup = lookup.sample(n=n_to_sample, random_state=None)[['lookup', 'id']]
        df = df.merge(lookup, on='id', how='left')
        df['old_names'] = df[col_name]
        df[col_name] = np.where(df['lookup'].notna(), df['lookup'], df[col_name])
        error_table = df[df['lookup'].notna()]
        if not error_table.empty:
            df, error_record = update_error_record(df, error_table['id'], col_name, 'invert_nick_realnames',
                                                  error_table['old_names'], error_table[col_name], error_record)
        df = df[org_colnames]
    return df, error_record

def add_name_suffix(df, n_errors, lname, sex, suffix_list=['JR', 'III', 'II', 'SR', 'IV', 'I', 'V'],
                    suffix_weights=[300, 40, 40, 40, 10, 10, 10], error_record=None):
    df = df.copy()
    ids = df['id']
    males = df[sex] == 'm'
    male_ids = ids[males]
    if error_record is None:
        error_record = pd.DataFrame(columns=['id', 'field', 'error', 'before', 'after'])
    if len(male_ids) < n_errors:
        print("Not enough candidates for suffixes found.")
        n_errors = len(male_ids)
    candidate_ids = np.random.choice(male_ids, size=n_errors, replace=False)
    old_names = df.loc[df['id'].isin(candidate_ids), lname]
    suffixes = np.random.choice(suffix_list, size=len(candidate_ids), p=np.array(suffix_weights)/sum(suffix_weights))
    new_names = old_names + ' ' + suffixes
    df.loc[df['id'].isin(candidate_ids), lname] = new_names
    df, error_record = update_error_record(df, candidate_ids, lname, 'name_suffix', old_names, new_names, error_record)
    return df, error_record