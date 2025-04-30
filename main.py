import pandas as pd
from data_loader import load_data, load_and_prepare_apr13, load_error_table
from error_utils import prep_data, mess_data

# Suppress FutureWarning for downcasting
pd.set_option('future.no_silent_downcasting', True)

def main():
    # Define paths
    data_path = '/Users/joshuaimmanuel/vs_lab3/HetGen/data/'
    apr13_path = '/Users/joshuaimmanuel/vs_lab3/HetGen/exta/apr13.csv'
    error_table_path = '/Users/joshuaimmanuel/vs_lab3/HetGen/benchmarking/error_table.csv'

    # Load data first to set global variables
    load_data(data_path)
    df = load_and_prepare_apr13(apr13_path)
    error_table = load_error_table(error_table_path)

    # Import error functions after load_data
    from edit_distance_errors import indel, repl, tpose
    from nickname_errors import real_to_nicknames, nick_to_realnames, invert_real_and_nicknames, add_name_suffix
    from abbreviation_errors import first_letter_abbreviate, blanks_to_hyphens, hyphens_to_blanks, make_missing
    from swap_errors import swap_fields
    from file_based_errors import make_twins, married_name_change
    from dob_errors import date_swap, date_replace, date_transpose

    # Define error functions dictionary
    error_functions = {
        'indel': indel,
        'repl': repl,
        'tpose': tpose,
        'real_to_nicknames': real_to_nicknames,
        'nick_to_realnames': nick_to_realnames,
        'invert_real_and_nicknames': invert_real_and_nicknames,
        'add_name_suffix': add_name_suffix,
        'first_letter_abbreviate': first_letter_abbreviate,
        'blanks_to_hyphens': blanks_to_hyphens,
        'hyphens_to_blanks': hyphens_to_blanks,
        'make_missing': make_missing,
        'swap_fields': swap_fields,
        'make_twins': make_twins,
        'married_name_change': married_name_change,
        'date_swap': date_swap,
        'date_replace': date_replace,
        'date_transpose': date_transpose
    }

    # Prepare data
    df_pairs = prep_data(df)

    # Apply errors
    print("Columns in df_pairs.df_secondary before mess_data:", df_pairs.df_secondary.columns.tolist())
    df_pairs = mess_data(df_pairs, error_table, error_functions, add_counting_dups=False, verbose=True)

    # Verify DataFrame integrity
    print("\nVerifying DataFrame integrity:")
    print(f"Original DataFrame file column unique values: {df_pairs.df_original['file'].unique()}")
    print(f"Secondary DataFrame file column unique values: {df_pairs.df_secondary['file'].unique()}")
    print(f"Original DataFrame row count: {len(df_pairs.df_original)}")
    print(f"Secondary DataFrame row count: {len(df_pairs.df_secondary)}")
    print(f"Columns in df_pairs.df_secondary after mess_data: {df_pairs.df_secondary.columns.tolist()}")

    # Verify results
    print("\nOriginal DataFrame (first 5 rows, unchanged):")
    print(df_pairs.df_original.head())
    print("\nSecondary DataFrame (first 5 rows, with errors):")
    print(df_pairs.df_secondary.head())
    print("\nError Record (first 10 rows or all if smaller):")
    print(df_pairs.error_record.head(10))

if __name__ == "__main__":
    main()