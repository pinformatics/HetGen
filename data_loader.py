import pandas as pd
import numpy as np
import pyreadr
import re
from datetime import datetime, timedelta

# Global variables for data
NAMES_LOOKUP = None
NICK_REAL_LOOKUP = None
LNAMES_ALL = None
FNAMES_MALE = None
FNAMES_FEMALE = None
DOB = None

def load_data(data_path):
    global NAMES_LOOKUP, NICK_REAL_LOOKUP, LNAMES_ALL, FNAMES_MALE, FNAMES_FEMALE, DOB
    # Load R data files
    NAMES_LOOKUP = pyreadr.read_r(f'{data_path}names_lookup.rda')['names_lookup']
    NICK_REAL_LOOKUP = pyreadr.read_r(f'{data_path}nick_real_lookup.rda')['nick_real_lookup']
    LNAMES_ALL = pyreadr.read_r(f'{data_path}lnames_all.rda')['lnames_all']['lnames_all'].tolist()
    FNAMES_MALE = pyreadr.read_r(f'{data_path}fnames_male.rda')['fnames_male']['fnames_male'].tolist()
    FNAMES_FEMALE = pyreadr.read_r(f'{data_path}fnames_female.rda')['fnames_female']['fnames_female']
    FNAMES_FEMALE = FNAMES_FEMALE.str.replace(r'^c\("|"?\)$', '', regex=True).tolist()
    DOB = pd.read_csv(f'{data_path}dob.csv')
    DOB['DOB'] = pd.to_datetime(DOB['DOB'])

    # Sample DOB for testing
    DOB = DOB.sample(n=min(1000, len(DOB)), random_state=42) if len(DOB) > 1000 else DOB
    FNAMES_FEMALE = FNAMES_FEMALE[:min(3000, len(FNAMES_FEMALE))]

    # Debug prints to verify data loading
    print("Loaded NAMES_LOOKUP:", len(NAMES_LOOKUP) if NAMES_LOOKUP is not None else None)
    print("Loaded NICK_REAL_LOOKUP:", len(NICK_REAL_LOOKUP) if NICK_REAL_LOOKUP is not None else None)
    print("Loaded LNAMES_ALL:", len(LNAMES_ALL) if LNAMES_ALL is not None else None)
    print("Loaded FNAMES_MALE:", len(FNAMES_MALE) if FNAMES_MALE is not None else None)
    print("Loaded FNAMES_FEMALE:", len(FNAMES_FEMALE) if FNAMES_FEMALE is not None else None)
    print("Loaded DOB:", len(DOB) if DOB is not None else None)

def load_and_prepare_apr13(file_path):
    df = pd.read_csv(file_path, sep=',', encoding='latin1', low_memory=False)
    df = df[[
        'first_name', 'last_name', 'gender_code', 'birth_age', 'res_street_address'
    ]].rename(columns={
        'first_name': 'fname',
        'last_name': 'lname',
        'gender_code': 'sex',
        'birth_age': 'age',
        'res_street_address': 'street_address'
    })
    df['dob'] = pd.to_datetime('2025-12-31') - pd.to_timedelta(df['age'] * 365.25, unit='D')
    df['dob'] = df['dob'].dt.date
    df = df.dropna(subset=['fname', 'lname', 'sex', 'age', 'street_address'])
    df['fname'] = df['fname'].str.strip().str.lower()
    df['lname'] = df['lname'].str.strip().str.lower()
    df['street_address'] = df['street_address'].str.strip().str.lower()
    df['sex'] = df['sex'].str.strip().str.lower()
    return df

def load_error_table(file_path):
    def parse_csv_line(line, delimiter=','):
        fields = []
        current_field = []
        in_quotes = False
        for char in line:
            if char == '"':
                in_quotes = not in_quotes
            elif char == delimiter and not in_quotes:
                fields.append(''.join(current_field).strip())
                current_field = []
            else:
                current_field.append(char)
        fields.append(''.join(current_field).strip())
        return fields

    error_table = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        header = parse_csv_line(lines[0].strip())
        for line in lines[1:]:
            fields = parse_csv_line(line.strip())
            if len(fields) == len(header):
                error_table.append(fields)
            else:
                print(f"Warning: Skipping malformed line: {line.strip()}")
    error_table = pd.DataFrame(error_table, columns=header)
    error_table = error_table.replace('', pd.NA)
    error_table = error_table.rename(columns={
        'error': 'error_function',
        'amount': 'amount',
        'col_names': 'columns',
        'arguments': 'params'
    })
    error_table['amount'] = error_table['amount'].astype(float)
    error_table['params'] = error_table['params'].replace(
        ["sex = 'gender_code'", "age = 'birth_age'"],
        ["sex = 'sex'", "age = 'age'"],
        regex=True
    )
    return error_table