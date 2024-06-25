import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.preprocessing import MultiLabelBinarizer

from STL.model import perform_Kfold_cv


def get_one_hot_encoding(df_in: DataFrame) -> DataFrame:
    empty_column = ''
    mlb = MultiLabelBinarizer()

    df_in = df_in.replace(np.nan, '')
    if len(df_in.columns) > 1:
        df_in = pd.DataFrame(df_in.apply(lambda row: '/'.join(row.values.astype(str)), axis=1))
    df_in1 = df_in.apply(lambda row: row.values[0].split('/'), axis=1)
    x_one_hot = pd.DataFrame(mlb.fit_transform(df_in1), columns=mlb.classes_, index=df_in1.index)
    if empty_column in x_one_hot.columns:
        x_one_hot = x_one_hot.drop(empty_column, axis=1)

    return x_one_hot


def run(w_size: int, use_dev_area: bool = True, use_cmp_inv: bool = True, use_dev_cmd: bool = True):
    label_names = ['CASSETTE', 'CT', 'NE', 'NF', 'NV', 'SHUTTER']
    seed_value = 1

    df = pd.read_csv("../data/outraw.data")

    col_dev_areas = []
    col_cmd = []
    col_err = []
    col_cmp_inv = []

    for idx in range(w_size):
        col_dev_areas.append(f"DEVICE AREA {str(idx + 1)}")
        col_cmd.append(f"CMD {str(idx + 1)}")
        col_err.append(f"ERRORCODE {str(idx + 1)}")
        col_cmp_inv.append(f"COMPONENT INVOLVED {str(idx + 1)}")

    # filter incidents - w_size
    df1 = df.loc[(df['W_SIZE'] <= w_size)]

    # filter action - remove 'CHECK'
    df1 = df1.loc[(df1['ACTION'].str.contains('CHECK') == False)]

    df_area = df1[col_dev_areas]
    df_cmd = df1[col_cmd]
    df_err = df1[col_err]
    df_cmp = df1[col_cmp_inv]

    x = pd.DataFrame()

    if use_dev_area:
        x_area = get_one_hot_encoding(df_area)
        x = pd.concat([x, x_area], axis=1)

    if use_cmp_inv:
        x_cmp = get_one_hot_encoding(df_cmp)
        x = pd.concat([x, x_cmp], axis=1)

    if use_dev_cmd:
        x_cmd = get_one_hot_encoding(df_cmd)
        x = pd.concat([x, x_cmd], axis=1)

    y = df1[label_names]

    models = {'XGBoost': {
        'multi_output__estimator__n_estimators': [50, 100, 200],
        'multi_output__estimator__max_depth': [3, 5, 10],
        'multi_output__estimator__learning_rate': [0.1, 0.01, 0.001]
    }
    }

    for model_name, hyperparameters in models.items():
        print(hyperparameters)
        cv_results = perform_Kfold_cv(x, y, seed_value, model_name, hyperparameters)

        print(f'{model_name} - Cross-validation results:')
        print(cv_results)
        print('-' * 40)


if __name__ == '__main__':
    #########
    # params
    w = 1
    dv_area = True
    cmp_inv = False
    dv_cmd = False
    #########

    run(w, dv_area, cmp_inv, dv_cmd)
