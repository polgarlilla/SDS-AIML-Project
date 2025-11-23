from data import (
    read_and_inspect_data,
    preprocess_data,
    create_holdout_set,
    cols_to_keep,
    qualification_map,
    categoricals,
)

df = read_and_inspect_data("data/Dropout_Data.xlsx", "Original Data")
df,df_holdout=create_holdout_set(df)
X, y = preprocess_data(df, cols_to_keep, qualification_map, categoricals)
