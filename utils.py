import pandas as pd
import numpy as np
import random
from df_pairs import DFPairs

def update_error_record(df, ids, field, error, before, after, error_record=None):
    if isinstance(df, DFPairs):
        return df.add_error_record(ids, field, error, before, after)
    elif isinstance(df, pd.DataFrame):
        if error_record is None:
            error_record = pd.DataFrame(columns=['id', 'field', 'error', 'before', 'after'])
        before = [before] if not isinstance(before, (list, pd.Series, np.ndarray)) else before
        after = [after] if not isinstance(after, (list, pd.Series, np.ndarray)) else after
        ids = list(ids) if isinstance(ids, (pd.Series, np.ndarray)) else ids
        len_ids = len(ids)
        len_before = len(before)
        len_after = len(after)
        if not (len_ids == len_before == len_after):
            raise ValueError(f"Length mismatch: ids ({len_ids}), before ({len_before}), after ({len_after})")
        error_record_new = pd.DataFrame({
            'id': ids,
            'field': field,
            'error': error,
            'before': [str(b) for b in before],
            'after': [str(a) for a in after]
        })
        error_record = pd.concat([error_record, error_record_new], ignore_index=True)
        return df, error_record
    else:
        raise ValueError("Input must be a DataFrame or DFPairs object")

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