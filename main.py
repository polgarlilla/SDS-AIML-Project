#Data loading and preprocessing

from data import (
    read_and_inspect_data,
    create_holdout_set,
    cols_to_keep,
    qualification_map,
    categoricals,
    preprocess_data,
)

df = read_and_inspect_data("data/Dropout_Data.xlsx", "Original Data")
df,df_holdout=create_holdout_set(df)
X, y = preprocess_data(df, cols_to_keep, qualification_map, categoricals)


#Data preprocessing for modeling: 
#scaling numeric features and one-hot encoding categorical ones

from features import preprocessor_for_modeling

preprocess = preprocessor_for_modeling()
