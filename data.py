'''
This script contains functions for data reading and preprocessing steps:
'''
import pandas as pd
from sklearn.compose import make_column_selector as selector
import numpy as np
from sklearn.model_selection import train_test_split

def read_and_inspect_data(excel_file, sheet_name):
    '''
    Reads an Excel file and prints its first rows and column data types.
    '''
    df = pd.read_excel(excel_file, sheet_name=sheet_name)
    print(df.head())
    print(df.dtypes)
    return df

