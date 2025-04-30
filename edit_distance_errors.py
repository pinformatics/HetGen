import pandas as pd
import numpy as np
import random
from utils import update_error_record, repl_character, tpose_base

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