import pandas as pd
import numpy as np
import random
from data_loader import LNAMES_ALL, FNAMES_MALE, FNAMES_FEMALE
from utils import update_error_record, repl_character

def make_twins(df, n_errors, fname='fname', sex='sex', error_record=None):
    if error_record is None:
        error_record = pd.DataFrame(columns=['id', 'field', 'error', 'before', 'after'])
    df, error_record = twins_generate(df, n_errors=n_errors, fname=fname, sex=sex, error_record=error_record)
    return df, error_record

def married_name_change(df, n_errors, lname, sex, dob=None, age=None, error_record=None):
    if LNAMES_ALL is None:
        raise ValueError("LNAMES_ALL data not provided")
    df = df.copy()
    df_s = df[df[sex] == 'f']
    if age is not None:
        df_s = df_s[df_s[age] > 20]
    if error_record is None:
        error_record = pd.DataFrame(columns=['id', 'field', 'error', 'before', 'after'])
    if n_errors > len(df_s):
        print("Not enough samples found for simulating married last name change.")
        n_errors = len(df_s)
    candidate_ids = df_s['id'].sample(n=n_errors, random_state=None)
    old_lnames = df.loc[df['id'].isin(candidate_ids), lname]
    old_lnames = old_lnames.astype(str)
    available_lnames = [name for name in LNAMES_ALL if name not in old_lnames.values]
    if len(available_lnames) < n_errors:
        print("Not enough unique last names available.")
        available_lnames = LNAMES_ALL
    new_names = pd.Series(available_lnames).sample(n=len(candidate_ids), random_state=None, replace=True).values
    df.loc[df['id'].isin(candidate_ids), lname] = new_names
    candidate_ids_list = list(candidate_ids)
    if len(candidate_ids_list) != len(old_lnames) or len(candidate_ids_list) != len(new_names):
        raise ValueError(
            f"Length mismatch: candidate_ids ({len(candidate_ids_list)}), "
            f"old_lnames ({len(old_lnames)}), new_names ({len(new_names)})"
        )
    df, error_record = update_error_record(df, candidate_ids_list, lname, 'married_name_change', old_lnames, new_names, error_record)
    return df, error_record

def twins_generate(df, n_errors, fname, id_col=None, sex=None, error_record=None):
    if FNAMES_MALE is None or FNAMES_FEMALE is None:
        raise ValueError("FNAMES_MALE and FNAMES_FEMALE data not provided")
    df = df.copy()
    if error_record is None:
        error_record = pd.DataFrame(columns=['id', 'field', 'error', 'before', 'after'])
    if n_errors > len(df):
        print("Not enough samples found for generating duplicates")
        n_errors = len(df)
    
    # Create fnames_lookup
    fnames_lookup = pd.concat([
        pd.DataFrame({'fname': FNAMES_MALE, 'sex': 'm'}),
        pd.DataFrame({'fname': FNAMES_FEMALE, 'sex': 'f'}).sample(
            n=min(3000, len(FNAMES_FEMALE)), random_state=None
        )
    ]).sample(frac=1, random_state=None)
    fnames_lookup['fname_len'] = fnames_lookup['fname'].str.len()
    
    def search_name(name, used_names, gender):
        # Map 'u' to 'm' or 'f' randomly
        if gender == 'u':
            gender = random.choice(['m', 'f'])
        matches = fnames_lookup[
            (fnames_lookup['fname_len'] == len(name)) &
            (fnames_lookup['fname'].str[0] == name[0]) &
            (fnames_lookup['sex'] == gender) &
            (~fnames_lookup['fname'].isin(used_names))
        ]
        if matches.empty:
            fallback_names = fnames_lookup[
                (fnames_lookup['sex'] == gender) &
                (~fnames_lookup['fname'].isin(used_names))
            ]
            if fallback_names.empty:
                fallback_names = fnames_lookup[fnames_lookup['sex'] == gender]
                if fallback_names.empty:
                    print(f"Warning: No suitable {gender} name found for {name}, keeping original")
                    return pd.Series({'fname': name, 'sex': gender})
            return fallback_names.sample(n=1, random_state=None)[['fname', 'sex']].iloc[0]
        return matches.sample(n=1, random_state=None)[['fname', 'sex']].iloc[0]
    
    candidate_ids = df['id'].sample(n=n_errors, random_state=None)
    twins_df = df[df['id'].isin(candidate_ids)].copy()
    fnames_old = twins_df[fname].copy()
    sexes = twins_df[sex].copy() if sex is not None else pd.Series(['m'] * len(twins_df), index=twins_df.index)
    used_names = []
    twins_df_new_data = []
    for name, g in zip(fnames_old, sexes):
        result = search_name(name, used_names, g)
        twins_df_new_data.append(result)
        used_names.append(result['fname'])
    twins_df_new = pd.DataFrame(twins_df_new_data, index=twins_df.index)
    twins_df[fname] = twins_df_new['fname']
    twins_df['id'] = twins_df['id'].astype(str) + '_twin'
    twins_df['file'] = 'b'
    if sex is not None:
        twins_df[sex] = twins_df_new['sex']
    if id_col is not None:
        twins_df[id_col] = twins_df[id_col].apply(lambda x: repl_character(str(x), '0123456789'))
    df['id'] = df['id'].astype(str)
    df = pd.concat([df, twins_df]).reset_index(drop=True)
    df, error_record = update_error_record(df, candidate_ids, fname, 'twins', fnames_old, twins_df[fname], error_record)
    return df, error_record