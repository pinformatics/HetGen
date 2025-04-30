import pandas as pd

class DFPairs:
    def __init__(self, df_original, df_secondary, error_record=None):
        self.df_original = df_original
        self.df_secondary = df_secondary
        self.error_record = error_record if error_record is not None else pd.DataFrame(
            columns=['id', 'field', 'error', 'before', 'after']
        )

    def add_error_record(self, ids, field, error, before, after):
        error_record = pd.DataFrame({
            'id': ids,
            'field': field,
            'error': error,
            'before': [str(b) for b in before],
            'after': [str(a) for a in after]
        })
        self.error_record = pd.concat([self.error_record, error_record], ignore_index=True)
        return self