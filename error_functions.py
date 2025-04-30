import pandas as pd
import numpy as np
import random
import re
from datetime import datetime
from data_loader import NAMES_LOOKUP, NICK_REAL_LOOKUP, LNAMES_ALL, FNAMES_MALE, FNAMES_FEMALE
from df_pairs import DFPairs
from error_utils import update_error_record

# Edit Distance Errors (from 2_edit_distance.R)
def indel(df, n_errors, col_names, error_record=None):
    df = df.copy()
    n = len(df)
    p = len(col_names)
    errors_col = n_errors // p if p > 0 else n_errors
    ids = df['id']
    if error_record is None:
        error_record = pd.DataFrame(columns=['id', 'field', 'error', 'before', 'after'])
    if n_errors <= n * p and errors_col > 0:
        for col_name in col_names:
            col_vals = df[col_name]
            candidate_ids = ids[col_vals.str.len() > 0].sample(n=min(errors_col, len(ids)), random_state=None)
            before = df.loc[df['id'].isin(candidate_ids), col_name]
            after = before.apply(lambda x: indel_character(str(x)))
            df.loc[df['id'].isin(candidate_ids), col_name] = after
            df, error_record = update_error_record(df, candidate_ids, col_name, 'indel', before, after, error_record)
    else:
        for _ in range(n_errors):
            row = random.randint(0, n-1)
            col_name = random.choice(col_names)
            before = df.loc[row, col_name]
            after = indel_character(str(before))
            df.loc[row, col_name] = after
            df, error_record = update_error_record(df, [df.loc[row, 'id']], col_name, 'indel', [before], [after], error_record)
    return df, error_record

def indel_character(edit_string, error_chars='abcdefghijklmnopqrstuvwxyz'):
    if not edit_string:
        return edit_string
    len_str = len(edit_string)
    if random.random() > 0.5:
        add_letter = random.choice(error_chars)
        cut = random.randint(0, len_str)
        return edit_string[:cut] + add_letter + edit_string[cut:]
    else:
        rem = random.randint(0, len_str-1)
        return edit_string[:rem] + edit_string[rem+1:]

def repl(df, n_errors, col_names, error_record=None):
    df = df.copy()
    n = len(df)
    p = len(col_names)
    errors_col = n_errors // p if p > 0 else n_errors
    ids = df['id']
    if error_record is None:
        error_record = pd.DataFrame(columns=['id', 'field', 'error', 'before', 'after'])
    if n_errors <= n * p and errors_col > 0:
        for col_name in col_names:
            col_vals = df[col_name]
            candidate_ids = ids[col_vals.str.len() > 0].sample(n=min(errors_col, len(ids)), random_state=None)
            before = df.loc[df['id'].isin(candidate_ids), col_name]
            after = before.apply(lambda x: repl_character(str(x)))
            df.loc[df['id'].isin(candidate_ids), col_name] = after
            df, error_record = update_error_record(df, candidate_ids, col_name, 'repl', before, after, error_record)
    else:
        for _ in range(n_errors):
            row = random.randint(0, n-1)
            col_name = random.choice(col_names)
            before = df.loc[row, col_name]
            after = repl_character(str(before))
            df.loc[row, col_name] = after
            df, error_record = update_error_record(df, [df.loc[row, 'id']], col_name, 'repl', [before], [after], error_record)
    return df, error_record

def repl_character(edit_string, error_chars='abcdefghijklmnopqrstuvwxyz'):
    if not edit_string:
        return edit_string
    chars = list(edit_string)
    repl_index = random.randint(0, len(chars)-1)
    subs = random.choice(error_chars)
    while chars[repl_index] == subs:
        subs = random.choice(error_chars)
    chars[repl_index] = subs
    return ''.join(chars)

def tpose_eligible(items):
    def count_distinct(s):
        chars = set(c for c in s.lower() if c.isalnum() or c == ' ')
        return len(chars) > 1
    return items.apply(count_distinct)

def tpose(df, n_errors, col_names, error_record=None):
    df = df.copy()
    n = len(df)
    p = len(col_names)
    errors_col = n_errors // p if p > 0 else n_errors
    ids = df['id']
    if error_record is None:
        error_record = pd.DataFrame(columns=['id', 'field', 'error', 'before', 'after'])
    if n_errors < n * p and errors_col > 0:
        for col_name in col_names:
            col_vals = df[col_name]
            is_eligible = tpose_eligible(col_vals)
            candidate_ids = ids[is_eligible].sample(n=errors_col, random_state=None)
            before = df.loc[df['id'].isin(candidate_ids), col_name]
            after = before.apply(tpose_base)
            df.loc[df['id'].isin(candidate_ids), col_name] = after
            df, error_record = update_error_record(df, candidate_ids, col_name, 'tpose', before, after, error_record)
    else:
        for _ in range(n_errors):
            row = random.randint(0, n-1)
            col_name = random.choice(col_names)
            before = df.loc[row, col_name]
            after = tpose_base(before)
            df.loc[row, col_name] = after
            df, error_record = update_error_record(df, [df.loc[row, 'id']], col_name, 'tpose', [before], [after], error_record)
    return df, error_record

def tpose_base(edit_string):
    if len(edit_string) < 2:
        return edit_string
    chars = list(edit_string)
    range_idx = list(range(1, len(chars)))
    tpose_index = random.choice(range_idx)
    tpose_index_l = tpose_index - 1
    while chars[tpose_index] == chars[tpose_index_l] and len(set(chars)) > 1:
        tpose_index = random.choice(range_idx)
        tpose_index_l = tpose_index - 1
    if len(set(chars)) <= 1:
        print("Warning: All characters are the same. tpose is not valid!")
        return edit_string
    chars[tpose_index], chars[tpose_index_l] = chars[tpose_index_l], chars[tpose_index]
    return ''.join(chars)

# Nickname Errors (from 3_nicknames_errors.R)
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

# Abbreviation Errors (from 4_abbreviations.R)
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

# Swap Errors (from 5_swaps.R)
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

# File-Based Errors (from 6_file_based_errors.R)
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

# DOB and Duplicate Errors (from 7_dob_errors.R & duplicate_add_errors.R)
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

def date_swap(df, n_errors, date, error_record=None):
    df = df.copy()
    if date not in df.columns:
        print(f"Skipping date_swap: '{date}' column not available in apr13.csv")
        return df, error_record
    if error_record is None:
        error_record = pd.DataFrame(columns=['id', 'field', 'error', 'before', 'after'])
    if len(df) == 1:
        days = pd.to_datetime(df[date]).day
        if pd.isna(days) or days > 12:
            print("Not enough candidate dates found")
            return df, error_record
        candidate_ids = df['id']
        old_values = df[date]
        new_values = pd.to_datetime(old_values).map(lambda x: x.replace(day=x.month, month=x.day) if pd.notna(x) else x)
        df[date] = new_values
        df, error_record = update_error_record(df, candidate_ids, date, 'date_month_swap', old_values, new_values, error_record)
    else:
        dates = pd.to_datetime(df[date])
        days = dates.dt.day
        months = dates.dt.month
        potential_candidates = df[(days < 13) & (months != days) & (~dates.isna())]
        if n_errors > len(potential_candidates):
            print("Not enough candidate dates found")
            n_errors = len(potential_candidates)
        if n_errors == 0:
            return df, error_record
        candidate_ids = potential_candidates['id'].sample(n=n_errors, random_state=None)
        old_values = df.loc[df['id'].isin(candidate_ids), date]
        new_values = pd.to_datetime(old_values).map(lambda x: x.replace(day=x.month, month=x.day) if pd.notna(x) else x)
        df.loc[df['id'].isin(candidate_ids), date] = new_values
        df, error_record = update_error_record(df, candidate_ids, date, 'date_month_swap', old_values, new_values, error_record)
    return df, error_record

def date_replace(df, n_errors, date, token='year', error_record=None):
    df = df.copy()
    if date not in df.columns:
        print(f"Skipping date_replace: '{date}' column not available in apr13.csv")
        return df, error_record
    if error_record is None:
        error_record = pd.DataFrame(columns=['id', 'field', 'error', 'before', 'after'])
    
    # Filter valid dates and corresponding IDs
    old_dates = pd.to_datetime(df[date], errors='coerce')
    valid_mask = ~old_dates.isna()
    valid_ids = df['id'][valid_mask]
    valid_dates = old_dates[valid_mask]
    
    if len(valid_ids) < n_errors:
        print(f"Not enough valid dates for date_replace ({token}). Using {len(valid_ids)}")
        n_errors = len(valid_ids)
    if n_errors == 0:
        return df, error_record
    
    # Sample unique candidate IDs
    candidate_ids = np.random.choice(valid_ids, size=n_errors, replace=False)
    
    # Get corresponding dates and indices for candidate IDs
    candidate_mask = df['id'].isin(candidate_ids)
    candidate_dates = old_dates[candidate_mask]
    candidate_indices = df[candidate_mask].index
    
    if token == 'year':
        years = candidate_dates.dt.year.astype(int)
        # Replace last two digits of the year
        replacements = years.astype(str).str[2:].apply(
            lambda x: repl_character(x, '0123456789').zfill(2)
        )
        # Construct new years, ensuring they are in valid range (1900–2025)
        new_years = pd.Series([
            min(max(1900, int(str(y)[:2] + r)), 2025)
            for y, r in zip(years, replacements)
        ], index=candidate_dates.index)
        new_dates = pd.to_datetime({
            'year': new_years,
            'month': candidate_dates.dt.month,
            'day': candidate_dates.dt.day
        }, errors='coerce')
        # Handle any remaining NaT by keeping original date
        new_dates = new_dates.where(~new_dates.isna(), candidate_dates)
    elif token == 'month':
        new_months = [random.choice(valid_months(d, y)) for d, y in zip(
            candidate_dates.dt.day, candidate_dates.dt.year
        )]
        new_dates = pd.to_datetime({
            'year': candidate_dates.dt.year,
            'month': new_months,
            'day': candidate_dates.dt.day
        }, errors='coerce')
    else:
        new_days = [
            min(max(valid_days(m, y)), int(repl_character(str(d).zfill(2), '0123456789')))
            for d, m, y in zip(
                candidate_dates.dt.day,
                candidate_dates.dt.month,
                candidate_dates.dt.year
            )
        ]
        new_dates = pd.to_datetime({
            'year': candidate_dates.dt.year,
            'month': candidate_dates.dt.month,
            'day': new_days
        }, errors='coerce')
    
    # Update DataFrame with new dates
    df.loc[candidate_indices, date] = new_dates
    
    # Update error record
    df, error_record = update_error_record(
        df, candidate_ids, date, f'date_replace_{token}', 
        candidate_dates, new_dates, error_record
    )
    return df, error_record

def date_transpose(df, n_errors, date, token='year', error_record=None):
    df = df.copy()
    if date not in df.columns:
        print(f"Skipping date_transpose: '{date}' column not available in apr13.csv")
        return df, error_record
    if error_record is None:
        error_record = pd.DataFrame(columns=['id', 'field', 'error', 'before', 'after'])
    
    # Filter valid dates and corresponding IDs
    old_dates = pd.to_datetime(df[date], errors='coerce')
    valid_mask = ~old_dates.isna()
    valid_ids = df['id'][valid_mask]
    valid_dates = old_dates[valid_mask]
    
    if len(valid_ids) < n_errors:
        print(f"Not enough valid dates for date_transpose ({token}). Using {len(valid_ids)}")
        n_errors = len(valid_ids)
    if n_errors == 0:
        return df, error_record
    
    if token == 'year':
        if len(df) == 1:
            old_values = old_dates
            new_values = old_dates.map(
                lambda x: x.replace(year=int(str(int(x.year))[:2] + str(int(x.year))[3] + str(int(x.year))[2])) if pd.notna(x) else x
            )
            df[date] = new_values
            candidate_ids = df['id']
            df, error_record = update_error_record(df, candidate_ids, date, 'date_transpose_year', old_values, new_values, error_record)
        else:
            year_str = valid_dates.dt.year.astype(float).fillna(0).astype(int).astype(str)
            candidates = valid_ids[year_str.str[2] != year_str.str[3]]
            candidate_ids = np.random.choice(candidates, size=min(n_errors, len(candidates)), replace=False)
            candidate_mask = df['id'].isin(candidate_ids)
            candidate_dates = old_dates[candidate_mask]
            candidate_indices = df[candidate_mask].index
            new_years = [int(str(int(y))[:2] + str(int(y))[3] + str(int(y))[2]) for y in candidate_dates.dt.year]
            new_dates = pd.to_datetime({
                'year': new_years,
                'month': candidate_dates.dt.month,
                'day': candidate_dates.dt.day
            }, errors='coerce')
            df.loc[candidate_indices, date] = new_dates
            df, error_record = update_error_record(df, candidate_ids, date, 'date_transpose_year', candidate_dates, new_dates, error_record)
    elif token == 'month':
        new_months = valid_dates.dt.month.apply(lambda x: int(tpose_base(str(x).zfill(2))))
        new_dates = pd.to_datetime({
            'year': valid_dates.dt.year,
            'month': new_months,
            'day': valid_dates.dt.day
        }, errors='coerce')
        valid = ~new_dates.isna()
        candidate_ids = valid_ids[valid].sample(n=min(n_errors, valid.sum()), random_state=None)
        candidate_mask = df['id'].isin(candidate_ids)
        candidate_dates = old_dates[candidate_mask]
        candidate_indices = df[candidate_mask].index
        df.loc[candidate_indices, date] = new_dates[candidate_ids.index]
        df, error_record = update_error_record(df, candidate_ids, date, 'date_transpose_month',
                                              candidate_dates, new_dates[candidate_ids.index], error_record)
    else:
        transposable_days_all = [1, 2, 10, 12, 20, 21, 30, 31]
        months_30 = [4, 6, 9, 11]
        months_31 = [1, 3, 5, 7, 8, 10, 12]
        transposable = (
            valid_dates.dt.day.isin(transposable_days_all) |
            (~valid_dates.dt.month.isin([2]) & valid_dates.dt.day.isin([3] + transposable_days_all)) |
            (valid_dates.dt.month.isin(months_31) & valid_dates.dt.day.isin([13] + transposable_days_all))
        )
        potential_ids = valid_ids[transposable]
        n_errors = min(n_errors, len(potential_ids))
        if n_errors == 0:
            print("Not enough transposable dates found. Transposing all available.")
            return df, error_record
        candidate_ids = np.random.choice(potential_ids, size=n_errors, replace=False)
        candidate_mask = df['id'].isin(candidate_ids)
        candidate_dates = old_dates[candidate_mask]
        candidate_indices = df[candidate_mask].index
        new_days = candidate_dates.dt.day.apply(lambda x: int(tpose_base(str(x).zfill(2))))
        new_dates = pd.to_datetime({
            'year': candidate_dates.dt.year,
            'month': candidate_dates.dt.month,
            'day': new_days
        }, errors='coerce')
        df.loc[candidate_indices, date] = new_dates
        df, error_record = update_error_record(df, candidate_ids, date, 'date_transpose_day', candidate_dates, new_dates, error_record)
    return df, error_record

def valid_days(month, year):
    if month in [1, 3, 5, 7, 8, 10, 12]:
        return list(range(1, 32))
    elif month == 2 and pd.Timestamp(year=year, month=1, day=1).is_leap_year:
        return list(range(1, 30))
    elif month == 2:
        return list(range(1, 29))
    else:
        return list(range(1, 31))

def valid_months(day, year):
    if day <= 28:
        return list(range(1, 13))
    elif day <= 29 and pd.Timestamp(year=year, month=1, day=1).is_leap_year:
        return list(range(1, 13))
    elif day <= 30:
        return [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    else:
        return [1, 3, 5, 7, 8, 10, 12]