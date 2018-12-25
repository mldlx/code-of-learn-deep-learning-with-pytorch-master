# coding=utf-8

import pandas as pd
import numpy as np

# -----------------------------------------------------------------------------
DF = pd.read_csv('data/adult_data.csv')
DF['income_label'] = (DF["income_bracket"].apply(lambda x: ">50K" in x)).astype(int)

DF.head()
# -----------------------------------------------------------------------------
# 1. set the experiment.
# wide side: wide and crossed cols; deep side: embedding and continuous cols
# embedding cols: a list of tuples with two elements ('col-name', 'dims of corresponding embeddings')
wide_cols = ['age', 'hours_per_week', 'education', 'relationship', 'workclass', 'occupation', 'native_country', 'gender']
crossed_cols = (['education', 'occupation'], ['native_country', 'occupation'])
embeddings_cols = [('education', 10), ('relationship', 8), ('workclass', 10), ('occupation', 10), ('native_country', 12)]
continuous_cols = ['age', 'hours_per_week']
target = 'income_label'
method = 'logistic'


emb_dim = dict(embeddings_cols)
embeddings_cols = [emb[0] for emb in embeddings_cols]
deep_cols = embeddings_cols + continuous_cols
# -----------------------------------------------------------------------------
# 2. cross-product for binary features
Y = np.array(DF[target])
df_tmp = DF.copy()[list(set(wide_cols + deep_cols))]

crossed_columns = []
for cols in crossed_cols:
    colname = '_'.join(cols)
    df_tmp[colname] = df_tmp[cols].apply(lambda x: '_'.join(x), axis=1)
    crossed_columns.append(colname)

categorical_columns = list(df_tmp.select_dtypes(include=['object']).columns)


# -----------------------------------------------------------------------------
# 3. label-encoding and spliting the dataframe into wide and deep
def label_encode(df, cols=None):
    if cols == None:
        cols = list(df.select_dtypes(include=['object']).columns)
    val_types = dict()
    for c in cols:
        val_types[c] = df[c].unique()

    val_to_idx = dict()
    for k, v in val_types.items():
        val_to_idx[k] = {o: i for i, o in enumerate(val_types[k])}

    for k, v in val_to_idx.items():
        df[k] = df[k].apply(lambda x: v[x])

    return val_to_idx, df

encoding_dict, df_tmp = label_encode(df_tmp)
encoding_dict = {k: encoding_dict[k] for k in encoding_dict if k in deep_cols}
embeddings_inputs = []
for k, v in encoding_dict.items():
    embeddings_inputs.append((k, len(v), emb_dim[k]))
