'''
This script contains functions for modeling-oriented preprocessing:
'''
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

numeric = ["Previous qualification (grade)",
           "Inflation rate",
           "GDP"]
categorical = ["Mother's qualification", 
               "Father's qualification",
               "Debtor",
               "Tuition fees up to date",
               "Gender",
               "Scholarship holder",
               "International"]

def preprocessor_for_modeling():
    """
    Creates a ColumnTransformer that scales numeric features and 
    one-hot encodes categorical features.
    """
    preprocess = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical),
        ]
    )
    return preprocess
