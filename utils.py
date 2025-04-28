from sklearn.preprocessing import LabelEncoder


def encode_table_data(df_X, df_y):
    n_uniques = df_X.nunique()
    dtypes_X = df_X.dtypes

    for col in df_X.columns:
        if dtypes_X[col] == 'object' or n_uniques[col] < 200:
            l_enc = LabelEncoder()
            df_X[col] = df_X[col].fillna("VV_likely")
            df_X[col] = l_enc.fit_transform(df_X[col].values)
        else:
            df_X[col] = df_X[col].fillna(df_X[col].mode()[0])

    l_enc = LabelEncoder()
    df_y = l_enc.fit_transform(df_y)
    return df_X, df_y


def count_params(model):
    return sum(p.numel() for p in model.parameters())