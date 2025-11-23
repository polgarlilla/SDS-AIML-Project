'''
This script contains functions for data reading and preprocessing steps:
'''
import pandas as pd
from sklearn.compose import make_column_selector as selector
import numpy as np
from sklearn.model_selection import train_test_split

def read_and_inspect_data(excel_file, sheet_name):
    '''
    Reads an Excel file and prints its first rows, column data types,
    and missing value counts.
    '''
    df = pd.read_excel(excel_file, sheet_name=sheet_name)
    print("\n=== First 5 rows ===")
    print(df.head())
    print("\n=== Columns' data types ===")
    print(df.dtypes)
    print("\n=== Missing values per column ===")
    print(df.isnull().sum())
    return df


def create_holdout_set(df, test_size=0.10, random_state=3):
    """
    Adds an ID for reproducibility and splits the data into a 90% working dataset and a 10% final holdout set.
    """
    df["StudentID"] = df.index + 1

    df, df_holdout = train_test_split(
        df,
        test_size=test_size,
        stratify=df["Target"],
        random_state=random_state)
    
    df = df.reset_index(drop=True) #we need this because train test split messes up our row indices
    df_holdout = df_holdout.reset_index(drop=True)

    print(f"Length of working dataset: {len(df)}")
    print(f"Length of holdout dataset: {len(df_holdout)}")
    return df,df_holdout

cols_to_keep = [
    "Target",
    "Previous qualification (grade)",
    "Mother's qualification",
    "Father's qualification",
    "Debtor",
    "Tuition fees up to date",
    "Gender",
    "Scholarship holder",
    "International",
    "Inflation rate",
    "GDP"
]

qualification_map = {
    # 0 – Unknown
    34: 0,

    # 1: Low/no education
    35: 1,  # Can't read or write
    36: 1,  # Can read without 4th year

    # 2: Basic education (1st–3rd cycles, secondary, complementary, commerce, etc.)
    1: 2,   # Secondary Education - 12th Year
    9: 2,   # 12th Year - Not Completed
    10: 2,  # 11th Year - Not Completed
    11: 2,  # 7th Year (Old)
    12: 2,  # Other - 11th Year
    13: 2,  # 2nd year complementary high school course (father)
    14: 2,  # 10th Year
    18: 2,  # General commerce course
    19: 2,  # Basic Education 3rd Cycle
    20: 2,  # Complementary High School Course (father)
    22: 2,  # Technical-professional course
    25: 2,  # Complementary High School Course - not concluded (father)
    26: 2,  # 7th year of schooling
    27: 2,  # 2nd cycle of general high school
    29: 2,  # 9th Year - Not Completed
    30: 2,  # 8th year
    31: 2,  # General Course of Administration and Commerce (father)
    33: 2,  # Supplementary Accounting and Administration (father)
    37: 2,  # Basic education 1st cycle
    38: 2,  # Basic education 2nd cycle

    # 3: Higher education (non-master)
    2: 3,   # Higher Education - Bachelor's
    3: 3,   # Higher Education - Degree
    6: 3,   # Frequency of Higher Education
    39: 3,  # Technological specialization course
    40: 3,  # Higher education - degree (1st cycle)
    41: 3,  # Specialized higher studies course
    42: 3,  # Professional higher technical course

    # 4: High-level higher education
    4: 4,   # Higher Education - Master's
    5: 4,   # Higher Education - Doctorate
    43: 4,  # Higher Education - Master (2nd cycle)
    44: 4   # Higher Education - Doctorate (3rd cycle)
}

categoricals = [
    "Debtor",
    "Tuition fees up to date",
    "Gender",
    "Scholarship holder",
    "International",
    "Target"
]

def preprocess_data(df,cols_to_keep,qualification_map,categoricals):
    """
    Selects important columns, encodes the target, recodes parents' education,
    sets categorical variables types, and returns X (features) and y (target).
    """
    df = df.copy()
    df = df[cols_to_keep].copy()

    #Remove "Enrolled" and encode Target (Dropout=0, Graduated=1)
    df = df[df["Target"] != "Enrolled"].copy()
    df["Target"] = df["Target"].astype("category").cat.codes

    #Recode parents educations
    df["Mother's qualification"] = df["Mother's qualification"].map(qualification_map)
    df["Father's qualification"] = df["Father's qualification"].map(qualification_map)

    #Convert categorical columns
    df[categoricals] = df[categoricals].astype("category")

    #Ordered categorical for parents qualifications
    qual_order = [0, 1, 2, 3, 4]
    qualification_type = pd.CategoricalDtype(categories=qual_order, ordered=True)
    df["Mother's qualification"] = df["Mother's qualification"].astype(qualification_type)
    df["Father's qualification"] = df["Father's qualification"].astype(qualification_type)

    y = df["Target"]
    X = df.drop(columns=["Target"])

    print("\n=== X (features) first 5 rows ===")
    print(X.head())
    print("\n=== y (target) first 5 rows ===")
    print(y.head())
    print("\n=== X columns' data types ===")
    print(X.dtypes)
    print("\n=== Target's distribution ===")
    print(y.value_counts())

    return X, y