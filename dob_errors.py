import pandas as pd
import numpy as np
import random
from datetime import datetime
from utils import update_error_record, tpose_base, repl_character

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
        print(f" Skippping date_replace: '{date}' column not available in apr13.csv")
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