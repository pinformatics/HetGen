import pandas as pd
import numpy as np
from df_pairs import DFPairs
from utils import update_error_record

def prep_data(df_original):
    if df_original.empty:
        raise ValueError("Input DataFrame is empty")
    if not all(col in df_original.columns for col in ['fname', 'lname', 'sex', 'age', 'dob', 'street_address']):
        raise ValueError("Input DataFrame missing required columns")
    df_original = df_original.copy()
    df_original['file'] = 'A'
    df_original['id'] = df_original.index.astype(str)  # Ensure id is string
    cols = ['file', 'id'] + [col for col in df_original.columns if col not in ['file', 'id']]
    df_original = df_original[cols]
    for col in df_original.select_dtypes(include=['object']).columns:
        if col != 'file' and df_original[col].dtype == 'object':
            df_original[col] = df_original[col].astype(str).str.lower()
    df_secondary = df_original.copy()
    df_secondary['file'] = 'b'
    return DFPairs(df_original, df_secondary)

def mess_data(df_data, error_lookup, error_functions, add_counting_dups=False, verbose=True, error_record=None):
    if isinstance(df_data, DFPairs):
        n = len(df_data.df_original)
        dup_error = error_lookup[error_lookup['error_function'] == 'add_duplicates']
        if not dup_error.empty:
            e = dup_error['amount'].iloc[0]
            if e < 1:
                e = int(np.ceil(e * n))
            if add_counting_dups:
                add_counting_dups = n + e
            df_data.df_secondary, df_data.error_record = mess_data(
                df_data.df_secondary,
                error_lookup[error_lookup['error_function'] != 'add_duplicates'],
                error_functions,
                add_counting_dups,
                verbose,
                df_data.error_record
            )
            df_data = add_duplicates(df_data, e)
        else:
            df_data.df_secondary, df_data.error_record = mess_data(
                df_data.df_secondary,
                error_lookup,
                error_functions,
                add_counting_dups,
                verbose,
                df_data.error_record
            )
        return df_data
    elif isinstance(df_data, pd.DataFrame):
        df = df_data.copy()
        if add_counting_dups:
            n = add_counting_dups
        else:
            n = len(df)
        if error_record is None:
            error_record = pd.DataFrame(columns=['id', 'field', 'error', 'before', 'after'])
        for _, row in error_lookup.iterrows():
            error_function = row['error_function']
            if verbose:
                print(f"\nApplying {error_function}")
            args = {'df': df, 'error_record': error_record}
            e = row['amount']
            if e < 1:
                e = int(np.ceil(e * n))
            args['n_errors'] = e
            col_names = [c.strip() for c in row['columns'].split(',')] if pd.notna(row['columns']) else []
            if col_names:
                args['col_names'] = col_names
            if pd.notna(row['params']):
                params = dict(p.strip().split('=') for p in row['params'].split(',') if '=' in p)
                args.update({k.strip(): v.strip().strip("'") for k, v in params.items()})
            try:
                func = error_functions.get(error_function)
                if func:
                    df, error_record = func(**args)
                else:
                    print(f"Function {error_function} not found")
            except Exception as e:
                print(f"Error applying {error_function}: {e}")
            if verbose:
                print(error_record.tail())
        return df, error_record
    else:
        raise ValueError("Input must be a DataFrame or DFPairs object")

def add_duplicates(df_pairs, n_errors):
    df_original = df_pairs.df_original.copy()
    df_secondary = df_pairs.df_secondary.copy()
    error_record = df_pairs.error_record.copy()
    ids = error_record['id'].unique() if not error_record.empty else df_original['id']
    ids = np.random.choice(ids, size=n_errors, replace=False)
    duplicates = df_original[df_original['id'].isin(ids)].copy()
    duplicates['file'] = 'b'
    df_secondary_new = pd.concat([df_secondary, duplicates]).reset_index(drop=True)
    df_pairs = DFPairs(df_original, df_secondary_new, error_record)
    df_pairs = df_pairs.add_error_record(ids, 'all_fields', 'duplicate', ['original'] * len(ids), ['original'] * len(ids))
    return df_pairs