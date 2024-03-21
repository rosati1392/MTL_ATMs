import numpy as np
import pandas as pd
from numpy import random
from pandas import DataFrame, Series
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MultiLabelBinarizer
import os
import modelMTL
import modelMTLweight
#from modelMTL import perform_Kfold_cv_MTL
from sklearn.metrics import confusion_matrix


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


def get_encoding_counter(df_in: DataFrame, columns: Series) -> DataFrame:
    empty_column = ''

    # create dataframe where columns are made of data available in df_in
    df_in = df_in.replace(np.nan, '')
    if len(df_in.columns) > 1:
        df_in = pd.DataFrame(df_in.apply(lambda row: '/'.join(row.values.astype(str)), axis=1))
    df_in1 = df_in.apply(lambda row: row.values[0].split('/'), axis=1)
    df_in1 = df_in1.apply(lambda row: [i.strip() for i in row])
    df_in2 = df_in1.apply(lambda row: pd.Series(row).value_counts())
    if empty_column in df_in2.columns:
        df_in2 = df_in2.drop(empty_column, axis=1)

    # create dataframe with all columns
    df_nan = pd.DataFrame(None, index=df_in2.index, columns=columns)
    df_nan = df_nan.drop(columns=df_in2.columns)

    df_out = pd.concat([df_in2, df_nan], axis=1)
    df_out = df_out.fillna(0)

    return df_out


def get_encoding_weight(df_in: DataFrame, columns: Series) -> DataFrame:
    empty_column = ''

    # create dataframe with all columns
    df_out = pd.DataFrame(None, index=df_in.index, columns=columns)
    df_out = df_out.fillna(0)

    # check
    weights = np.linspace(1, 0.1, len(df_in.columns))

    for i in range(len(df_in)):
        idx_row = df_in.index[i]
        for j in range(len(df_in.columns)):
            item = df_in.iloc[i, j]
            if not pd.isnull(item):
                item_list = item.split('/')
                for cmp_inv_itm in item_list:
                    cmp_inv_itm = cmp_inv_itm.strip()
                    if cmp_inv_itm in df_out.columns:
                        df_out[cmp_inv_itm][idx_row] += weights[j]

    return df_out


def get_encoding_binary(df_in: DataFrame, bin_encode: DataFrame, df_counters: DataFrame) -> DataFrame:
    if df_counters is None:
        columns_len = len(bin_encode.index) * len(df_in.columns)
    else:
        columns_len = len(bin_encode.index) * len(df_in.columns) + len(df_counters.columns)

    df_out = pd.DataFrame(None, index=df_in.index, columns=range(columns_len))

    for i in range(len(df_in)):
        idx_row = df_in.index[i]
        inc_cmp_list = []  # list of series with binary encoding component involved
        for j in range(len(df_in.columns)):
            item = df_in.iloc[i, j]
            if not pd.isnull(item):
                item_list = item.split('/')
                for cmp_inv_itm in item_list:
                    cmp_inv_itm = cmp_inv_itm.strip()
                    if cmp_inv_itm in bin_encode.columns:
                        inc_cmp_list.append(bin_encode[cmp_inv_itm])
                        if df_counters is not None:
                            inc_cmp_list.append(pd.Series(df_counters.iloc[i, j]))

        if len(inc_cmp_list) > 0:
            cmp_series = pd.concat(inc_cmp_list, ignore_index=True)
            cmp_series = cmp_series.reindex(range(columns_len), fill_value=np.nan)
            df_out.loc[idx_row, :] = cmp_series

    df_out.columns = df_out.columns.astype(str)
    df_out = df_out.fillna(0)

    return df_out


def get_one_hot_encoding_column(df_in: DataFrame, cat_list) -> DataFrame:
    x = pd.DataFrame()
    for column in df_in:
        column_data = df_in[column]
        column_data_categories = column_data.astype(pd.CategoricalDtype(categories=cat_list))
        one_hot_data = pd.get_dummies(column_data_categories).astype(int)
        x = pd.concat([x, one_hot_data], axis=1)
    return x


def generic_encoding(x, names):
    ret = x
    if x in names:
        ret = names.index(x)
    return ret


def area_encoding(x):
    area_names = ['CASSETTE', 'CT', 'NE', 'NF', 'NV', 'SHUTTER', 'UNK']
    return generic_encoding(x, area_names)


def action_status_encoding(x):
    action_status_names = ["OK", "KO"]
    return generic_encoding(x, action_status_names)


def action_encoding(x):
    action_names = ["DISPENSE", "DEPOSIT", "EXCHANGE", "RESET", "CLEAR_COUNTS", "COUNT", "RETRACT", "REJECT",
                    "ROLLBACK", "CASHINSTART", "PRESENT", "OPERATOR_IN", "OPERATOR_OUT", "REBOOT"]
    return generic_encoding(x, action_names)


def get_area_encoding(df_in: DataFrame) -> DataFrame:
    return df_in.applymap(area_encoding)


def get_action_encoding(df_in: DataFrame) -> DataFrame:
    return df_in.applymap(action_encoding)


def get_action_status_encoding(df_in: DataFrame) -> DataFrame:
    return df_in.applymap(action_status_encoding)


def my_unpackbits(x, num_bits):
    if np.issubdtype(x.dtype, np.floating):
        raise ValueError("numpy data type needs to be int-like")
    xshape = list(x.shape)
    x = x.reshape([-1, 1])
    mask = 2 ** np.arange(num_bits, dtype=x.dtype).reshape([1, num_bits])
    return (x & mask).astype(bool).astype(int).reshape(xshape + [num_bits])

def get_encoding_counters_cmp_inv_binary(df_cmp_inv: DataFrame, df_pre_cmd_counters: DataFrame) -> DataFrame:
    cmp_inv_current_count = np.ones(len(df_cmp_inv.index))
    df_counter_curr = pd.DataFrame(cmp_inv_current_count, index=df_cmp_inv.index)
    df_out = pd.concat([df_counter_curr, df_pre_cmd_counters], axis=1)
    df_out = df_out.fillna(0)
    return df_out

def format_filename_res(cfg_dict: dict, infores: str) -> str:
    ret = f"{filename_res}_model-{cfg_dict['model']}_lr-{cfg_dict['lr']}_l1ratio-{cfg_dict['l1ratio']}_lambdatc-{cfg_dict['lambdatc']}_{infores}.csv"
    return ret

def generate_confusion_matrix(preds, truths, num_tasks):
    preds=preds.values
    truths=truths.values

    totcm=np.zeros((num_tasks,num_tasks))
    temp=np.zeros((num_tasks,num_tasks))
    for i in range(len(preds)):
        loctruth=np.where(truths[i,:]==1)[0]
        locpred=np.where(preds[i,:]==1)[0] 
    
        for r in range(len(loctruth)):
            for s in range(len(locpred)):
                   temp[loctruth[r],locpred[s]]=1
                   totcm=totcm+temp
                   temp=np.zeros((num_tasks,num_tasks))

    return totcm

def balanced_accuracy(cm):
    """
    Compute the balanced accuracy from a confusion matrix for a multiclass classification problem.
    
    :param cm: Confusion matrix (numpy array) where cm[i, j] is the number of observations known to be in group i and predicted to be in group j.
    :return: Balanced accuracy score
    """
        # Calculate recall for each class
    recall_per_class = cm.diagonal() / cm.sum(axis=1)
    
    # Handle potential division by zero
    recall_per_class = recall_per_class[~np.isnan(recall_per_class)]
    
    # Calculate balanced accuracy
    balanced_acc = recall_per_class.mean()
        
    return balanced_acc

def run(data_filename: str, w_size: int, pre_action_size: int, use_dev_area: bool = True, use_cmp_inv: bool = True,
        use_dev_cmd: bool = True, use_rej_data: bool = True, use_pre_action: bool = True,
        use_pre_action_onehot: bool = True,
        use_pre_action_status: bool = True,
        use_pre_dev_area=True, use_pre_dev_area_onehot=True, use_pre_cmp_inv=True, use_note_details=True,
        use_cmp_inv_freq=True, use_cmp_inv_weight=True, cmp_inv_list_filename: str = "", enable_plots=True,
        use_cmp_inv_binary=True, use_cmp_inv_binary_count=True, use_w_0=False, use_dev_area_confidence_low=True,
        seed_value: int = 1):
    label_names = ['CASSETTE', 'CT', 'NE', 'NF', 'NV', 'SHUTTER']
    # seed_value = 1

    df = pd.read_csv(data_filename)

    # col_rej_data = ["REJ RATE DISPENSE TOT", "REJ RATE L2L3", "REJ RATE DISPENSE 20", "REJ RATE DISPENSE 50"]
    col_rej_data = ["REJ RATE DISPENSE TOT", "REJ RATE L2L3"]
    col_rej_l4b = "REJ RATE L4B"

    if col_rej_l4b in df:
        col_rej_data.append(col_rej_l4b)

    col_dev_areas = []
    col_cmd = []
    col_err = []
    col_cmp_inv = []
    col_pre_action = []
    col_pre_action_status = []
    col_pre_dev_area = []
    col_pre_cmp_inv = []
    col_pre_cmd_counter = []

    col_note_details = ["STACKER", "RETRACT", "DEPOSIT ALL", "DEPOSIT L2", "DEPOSIT L3", "RECYCLING"]

    for idx in range(w_size):
        col_dev_areas.append(f"DEVICE AREA {str(idx + 1)}")
        col_cmd.append(f"CMD {str(idx + 1)}")
        col_err.append(f"ERRORCODE {str(idx + 1)}")
        col_cmp_inv.append(f"COMPONENT INVOLVED {str(idx + 1)}")

    if pre_action_size > 0:
        for pre_id in range(pre_action_size):
            col_pre_action.append(f"PRE ACTION {str(pre_id + 1)}")
            col_pre_action_status.append(f"PRE ACTION STATUS {str(pre_id + 1)}")
            col_pre_dev_area.append(f"PRE DEVICE AREA {str(pre_id + 1)}")
            col_pre_cmp_inv.append(f"PRE COMPONENT INVOLVED {str(pre_id + 1)}")
            col_pre_cmd_counter.append(f"PRE CMD COUNTER {str(pre_id + 1)}")

    # filter incidents - w_size
    df1 = df.loc[(df['W_SIZE'] <= w_size) & (df['W_SIZE'] > 0)]

    # incidents - w_size=0
    df0 = df.loc[(df['W_SIZE'] == 0)]

    # filter action - remove 'CHECK'
    df1 = df1.loc[(df1['ACTION'].str.contains('CHECK') == False)]
    df0 = df0.loc[(df0['ACTION'].str.contains('CHECK') == False)]

    ########################
    # export a subset of 150 samples (small dataset)
    # df1 = df1.loc[(df['W_SIZE'] == w_size)]
    # df_sample = df1.sample(n=150)
    # df_sample.to_csv('../data/outraw_20230921.small.data', index=False)
    ########################

    df_area = df1[col_dev_areas]
    df0_area = df0[col_dev_areas]
    df_cmd = df1[col_cmd]
    df0_cmd = df0[col_cmd]
    df_err = df1[col_err]
    df_cmp = df1[col_cmp_inv]
    df0_cmp = df0[col_cmp_inv]
    df_rej_data = df1[col_rej_data]
    df0_rej_data = df0[col_rej_data]

    df_pre_dev_area = df1[col_pre_dev_area]
    df0_pre_dev_area = df0[col_pre_dev_area]
    df_pre_cmp_inv = df1[col_pre_cmp_inv]
    df0_pre_cmp_inv = df0[col_pre_cmp_inv]

    x = pd.DataFrame()
    x0 = pd.DataFrame()             # w_size = 0
    x_baseline = pd.DataFrame()     # x for baseline, contains only dev_area
    x0_baseline = pd.DataFrame()  # x for baseline, contains only dev_area

    if use_dev_area:
        x_baseline = get_one_hot_encoding(df_area) # compute x_baseline before removal of device area for confidence low
        x0_baseline = get_one_hot_encoding(df0_area)

        if not use_dev_area_confidence_low:
            # remove the device area if confidence is low
            df2 = df1.loc[(pd.isnull(df1['CONFIDENCE']) == False)]
            df3 = df2.loc[(df2['CONFIDENCE'].str.contains('Low'))]
            ids = df3.index
            df_area.loc[ids, :] = np.nan

            df2 = df0.loc[(pd.isnull(df0['CONFIDENCE']) == False)]
            df3 = df2.loc[(df2['CONFIDENCE'].str.contains('Low'))]
            ids = df3.index
            df0_area.loc[ids, :] = np.nan

        x_area = get_one_hot_encoding(df_area)
        x = pd.concat([x, x_area], axis=1)
        x0_area = get_one_hot_encoding(df0_area)
        x0 = pd.concat([x0, x0_area], axis=1)

    if use_cmp_inv:
        x_cmp = get_one_hot_encoding(df_cmp)
        x = pd.concat([x, x_cmp], axis=1)
        x0_cmp = get_one_hot_encoding(df0_cmp)
        x0 = pd.concat([x0, x0_cmp], axis=1)

    if use_dev_cmd:
        x_cmd = get_one_hot_encoding(df_cmd)
        x = pd.concat([x, x_cmd], axis=1)

    if use_rej_data:
        x = pd.concat([x, df_rej_data], axis=1)
        df0_rej_data.fillna(-1, inplace=True)
        x0 = pd.concat([x0, df0_rej_data], axis=1)

    if use_pre_action:
        df_pre_action = df1[col_pre_action]
        x_pre_action = get_action_encoding(df_pre_action)
        x = pd.concat([x, x_pre_action], axis=1)
        df0_pre_action = df0[col_pre_action]
        x0_pre_action = get_action_encoding(df0_pre_action)
        x0 = pd.concat([x0, x0_pre_action], axis=1)

    if use_pre_action_onehot:
        df_pre_action = df1[col_pre_action]
        action_names = ["DISPENSE", "DEPOSIT", "EXCHANGE", "RESET", "CLEAR_COUNTS", "COUNT", "RETRACT", "REJECT",
                        "ROLLBACK", "CASHINSTART", "PRESENT", "OPERATOR_IN", "OPERATOR_OUT", "REBOOT"]
        x_pre_action_onehot = get_one_hot_encoding_column(df_pre_action, action_names)
        x = pd.concat([x, x_pre_action_onehot], axis=1)
        df0_pre_action = df0[col_pre_action]
        x0_pre_action_onehot = get_one_hot_encoding_column(df0_pre_action, action_names)
        x0 = pd.concat([x0, x0_pre_action_onehot], axis=1)

    if use_pre_action_status:
        df_pre_action_status = df1[col_pre_action_status]
        x_pre_action_status = get_action_status_encoding(df_pre_action_status)
        x = pd.concat([x, x_pre_action_status], axis=1)
        df0_pre_action_status = df0[col_pre_action_status]
        x0_pre_action_status = get_action_status_encoding(df0_pre_action_status)
        x0 = pd.concat([x0, x0_pre_action_status], axis=1)

    if use_pre_dev_area:
        x_pre_dev_area = get_area_encoding(df_pre_dev_area)
        x = pd.concat([x, x_pre_dev_area], axis=1)
        x0_pre_dev_area = get_area_encoding(df0_pre_dev_area)
        x0 = pd.concat([x0, x0_pre_dev_area], axis=1)

    if use_pre_dev_area_onehot:
        area_names = ['CASSETTE', 'CT', 'NE', 'NF', 'NV', 'SHUTTER', 'UNK']
        x_pre_dev_area_onehot = get_one_hot_encoding_column(df_pre_dev_area, area_names)
        x = pd.concat([x, x_pre_dev_area_onehot], axis=1)
        x0_pre_dev_area_onehot = get_one_hot_encoding_column(df0_pre_dev_area, area_names)
        x0 = pd.concat([x0, x0_pre_dev_area_onehot], axis=1)

    if use_pre_cmp_inv:
        x_pre_cmp_inv = get_one_hot_encoding(df_pre_cmp_inv)
        x = pd.concat([x, x_pre_cmp_inv], axis=1)
        x0_pre_cmp_inv = get_one_hot_encoding(df0_pre_cmp_inv)
        x0 = pd.concat([x0, x0_pre_cmp_inv], axis=1)

    if use_note_details:
        df_note_details = df1[col_note_details]
        note_details_names = ['NONE', 'IN', 'OUT']
        x_note_details = get_one_hot_encoding_column(df_note_details, note_details_names)
        x = pd.concat([x, x_note_details], axis=1)
        df0_note_details = df0[col_note_details]
        x0_note_details = get_one_hot_encoding_column(df0_note_details, note_details_names)
        x0 = pd.concat([x0, x0_note_details], axis=1)

    if use_cmp_inv_freq or use_cmp_inv_weight:
        df_cmps = pd.concat([df_cmp, df_pre_cmp_inv], axis=1)
        df0_cmps = pd.concat([df0_cmp, df0_pre_cmp_inv], axis=1)
        cmp_list = pd.read_csv(cmp_inv_list_filename)
        if use_cmp_inv_freq:
            x_cmp = get_encoding_counter(df_cmps, cmp_list.iloc[:, 0])
            x0_cmp = get_encoding_counter(df0_cmps, cmp_list.iloc[:, 0])
        else:
            x_cmp = get_encoding_weight(df_cmps, cmp_list.iloc[:, 0])
            x0_cmp = get_encoding_weight(df0_cmps, cmp_list.iloc[:, 0])
        x = pd.concat([x, x_cmp], axis=1)
        x0 = pd.concat([x0, x0_cmp], axis=1)

    if use_cmp_inv_binary or use_cmp_inv_binary_count:
        cmp_list = pd.read_csv(cmp_inv_list_filename)

        # need to check number of bits required
        bit_len = cmp_list.index.max().bit_length()
        cmp_idx_array = cmp_list.index.to_numpy()
        # sum 1 to start encoding from 1, leave 0 for nan
        cmp_idx_array += 1
        index_array = my_unpackbits(cmp_idx_array, bit_len)
        # index_array = np.unpackbits(cmp_list.index.to_numpy(dtype=np.uint8)).reshape(-1, 8)
        bin_enc = pd.DataFrame(index_array.transpose(), columns=cmp_list.iloc[:, 0])
        df_cmps = pd.concat([df_cmp, df_pre_cmp_inv], axis=1)
        df0_cmps = pd.concat([df0_cmp, df0_pre_cmp_inv], axis=1)

        df_counters = None
        df0_counters = None

        if use_cmp_inv_binary_count:
            df_pre_cmd_counter = df1[col_pre_cmd_counter]
            df0_pre_cmd_counter = df0[col_pre_cmd_counter]
            df_counters = get_encoding_counters_cmp_inv_binary(df_cmps, df_pre_cmd_counter)
            df0_counters = get_encoding_counters_cmp_inv_binary(df0_cmps, df0_pre_cmd_counter)

        x_cmp = get_encoding_binary(df_cmps, bin_enc, df_counters)
        x0_cmp = get_encoding_binary(df0_cmps, bin_enc, df0_counters)

        x = pd.concat([x, x_cmp], axis=1)
        x0 = pd.concat([x0, x0_cmp], axis=1)

    y = df1[label_names]
    y0 = df0[label_names]

    # concatenate data with w=0
    if use_w_0:
        x = pd.concat([x, x0])
        y = pd.concat([y, y0])
        x_baseline = pd.concat([x_baseline, x0_baseline])

    df_res_tot = pd.DataFrame()

    for hyps_dict in hyps_dict_list:

        # CV straified
        outer_cv = KFold(n_splits=10, shuffle=True, random_state=seed_value)
        fold=0
        ba0list=[]
        ba1list=[]
        ba2list=[]
        ba3list=[]
        ba4list=[]
        ba5list=[]
        base0list=[]
        base1list=[]
        base2list=[]
        base3list=[]
        base4list=[]
        base5list=[]
        label_cm_list=[]
        y_pred_list=[]
        y_test_list=[]

        for train_idx, test_idx in outer_cv.split(x, y):
            fold = fold + 1
            print('Fold:')
            print(fold)
            #x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)
            x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            x_baseline_fold = x_baseline.iloc[test_idx] if len(x_baseline) > 0 else None
            # export df_y_test
            col_df_y_test = ['ID', 'CSP', 'DATE']
            for label in label_names:
                col_df_y_test.append(label)
            df_y_test = df.iloc[y_test.index.values]
            df_y_test['ID'] = range(df_y_test.shape[0])
            df_y_test = df_y_test[col_df_y_test]

            # filename_ytest = filename + str(fold)+ ".ytest.csv"
            # df_y_test.to_csv(filename_ytest, index=False)

            if hyps_dict['model'] == 'MTL-LR':
                label_cm, ba0, ba1, ba2, ba3, ba4, ba5, base0, base1, base2, base3, base4, base5, y_pred = modelMTL.perform_Kfold_cv_MTL(x_train, x_test, y_train, y_test, seed_value, use_dev_area, enable_plots, hyps_dict)
            elif hyps_dict['model'] == 'MTL-LR-weight' or hyps_dict['model'] == 'MTL-SH-weight':
                label_cm, ba0, ba1, ba2, ba3, ba4, ba5, base0, base1, base2, base3, base4, base5, y_pred = modelMTLweight.perform_Kfold_cv_MTL(x_train, x_test, y_train, y_test, seed_value, use_dev_area, enable_plots, hyps_dict)
            else:
                raise ValueError('Invalid model name.')

            ba0list.append(ba0)
            ba1list.append(ba1)
            ba2list.append(ba2)
            ba3list.append(ba3)
            ba4list.append(ba4)
            ba5list.append(ba5)

            base0list.append(base0)
            base1list.append(base1)
            base2list.append(base2)
            base3list.append(base3)
            base4list.append(base4)
            base5list.append(base5)

            y_pred_list.append(y_pred)
            y_test_list.append(y_test)
            label_cm_list.append(label_cm)
            # ypreddf=pd.DataFrame(np.squeeze(y_pred_list))  
            # ytestdf=pd.DataFrame(np.squeeze(y_test_list)) 
            # cm_tot_multiclass=generate_confusion_matrix(ypreddf, ytestdf, 6)
            # ba_tot_multiclass=balanced_accuracy(cm_tot_multiclass)
            # ba_tot_multiclass.to_csv('prova.csv')

        ba0s = pd.Series(ba0list)
        ba1s = pd.Series(ba1list)
        ba2s = pd.Series(ba2list)
        ba3s = pd.Series(ba3list)
        ba4s = pd.Series(ba4list)
        ba5s = pd.Series(ba5list)
        df_ba = pd.concat([ba0s, ba1s, ba2s, ba3s, ba4s, ba5s], axis=1)
        # create idx column
        list_idx = list(df_ba.index)
        list_idx.append('average')
        df_idx = pd.DataFrame({'idx': list_idx})

        # calculate and concatenate average row
        df_ba_mean = df_ba.mean()
        df_ba = pd.concat([df_ba, df_ba_mean.to_frame().transpose()], ignore_index=True)
        # concatenate idx column
        df_ba = pd.concat([df_idx, df_ba], axis=1)
        df_ba.set_index('idx')
        ba_res = format_filename_res(hyps_dict, 'ba')
        df_ba.to_csv(ba_res, index=False)

        # combine configuration parameters and average results
        df_average = df_ba[df_ba['idx'] == 'average']
        df_cur_res = pd.DataFrame([hyps_dict])

        df_av = df_average.loc[:, df_average.columns != 'idx']
        df_cur_res = df_cur_res.merge(df_av, how='cross')
        df_res_tot = pd.concat([df_res_tot, df_cur_res], ignore_index=True, axis=0)

        base0s = pd.Series(base0list)
        base1s = pd.Series(base1list)
        base2s = pd.Series(base2list)
        base3s = pd.Series(base3list)
        base4s = pd.Series(base4list)
        base5s = pd.Series(base5list)
        df_baseline = pd.concat([base0s, base1s, base2s, base3s, base4s, base5s], axis=1)
        # calculate and concatenate average row
        df_baseline_mean = df_baseline.mean()
        df_baseline = pd.concat([df_baseline, df_baseline_mean.to_frame().transpose()], ignore_index=True)
        # concatenate idx column
        df_baseline = pd.concat([df_idx, df_baseline], axis=1)
        df_baseline.set_index('idx')
        baseline_res = format_filename_res(hyps_dict, 'baseline')
        df_baseline.to_csv(baseline_res, index=False)
        
        ypreddf=pd.DataFrame(y_pred_list)  
        ytestdf=pd.DataFrame(y_test_list)  
        cm_tot_multiclass=generate_confusion_matrix(ypreddf, ytestdf, 6)
        ba_tot_multiclass=balanced_accuracy(cm_tot_multiclass)
        ba_tot_multiclass.to_csv('CMmulticlass.csv')

    df_res_tot.to_csv(filename_res_tot, index=False)

if __name__ == '__main__':
    #########
    '''fixing seed'''
    seed_value = 1
    # 1. Set `PYTHONHASHSEED` environment variable at a fixed value
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    # 2. Set `python` built-in pseudo-random generator at a fixed value
    random.seed(seed_value)
    random_state_val = seed_value
    # 3. Set `numpy` pseudo-random generator at a fixed value
    np.random.seed(seed_value)

    # params
    filename = "./data/outraw_20240109.data"
    filename_res = "./results/MTL_6tasks/outraw_20240109"
    filename_res_tot = "./results/MTL_6tasks/outraw_20240109_tot.csv"
    w = 1
    dv_area = True
    cmp_inv = False
    dv_cmd = False
    reject_data = True
    n_pre_actions = 10
    pre_action = False
    pre_action_onehot = False
    pre_action_status = False
    pre_dev_area = False
    pre_dev_area_onehot = False
    pre_cmp_inv = False
    note_details = True
    cmp_inv_freq = False
    cmp_inv_weight = False
    cmp_inv_listfile = "./data/cmp_inv_list.csv"
    enable_plot = False
    cmp_inv_binary = True
    cmp_inv_binary_count = False
    w_0 = False
    dev_area_confidence_low = False     # if False, remove device area from features, only for incidents with confidence Low
    hyps_dict_list = [
                      {'model': 'MTL-LR','lr': 0.00507,'l1ratio': 0.0094,'lambdatc': 0.93573}
                      #{'model': 'MTL-LR-weight', 'lr': 0.0097, 'l1ratio': 0.00543, 'lambdatc': 0.34612},
                      #{'model': 'MTL-SH-weight', 'lr': 0.00745, 'l1ratio': 0.000042146, 'lambdatc': 0.84734}
                      # {'model': 'MTL-LR','lr': 0.01,'l1ratio': 0.0001,'lambdatc':0.1},
                      # {'model': 'MTL-LR','lr': 0.01,'l1ratio': 0.001,'lambdatc':0.1},
                      # {'model': 'MTL-LR','lr': 0.01,'l1ratio': 0.01,'lambdatc':0.1},
                      # {'model': 'MTL-LR','lr': 0.01,'l1ratio': 0.1,'lambdatc':0.1},
                      # {'model': 'MTL-LR','lr': 0.01, 'l1ratio': 0.0001, 'lambdatc': 1},
                      # {'model': 'MTL-LR','lr': 0.01, 'l1ratio': 0.001, 'lambdatc': 1},
                      # {'model': 'MTL-LR','lr': 0.01, 'l1ratio': 0.01, 'lambdatc': 1},
                      # {'model': 'MTL-LR','lr': 0.01, 'l1ratio': 0.1, 'lambdatc': 1},
                      # {'model': 'MTL-LR','lr': 0.01, 'l1ratio': 0.0001, 'lambdatc': 10},
                      # {'model': 'MTL-LR','lr': 0.01, 'l1ratio': 0.001, 'lambdatc': 10},
                      # {'model': 'MTL-LR','lr': 0.01, 'l1ratio': 0.01, 'lambdatc': 10},
                      # {'model': 'MTL-LR','lr': 0.01, 'l1ratio': 0.1, 'lambdatc': 10},
                      # {'model': 'MTL-LR-weight','lr': 0.01,'l1ratio': 0.0001,'lambdatc':0.1},
                      # {'model': 'MTL-LR-weight','lr': 0.01, 'l1ratio': 0.001, 'lambdatc': 0.1},
                      # {'model': 'MTL-LR-weight','lr': 0.01, 'l1ratio': 0.01, 'lambdatc': 0.1},
                      # {'model': 'MTL-LR-weight','lr': 0.01, 'l1ratio': 0.1, 'lambdatc': 0.1},
                      # {'model': 'MTL-LR-weight', 'lr': 0.01, 'l1ratio': 0.0001, 'lambdatc': 1},
                      # {'model': 'MTL-LR-weight', 'lr': 0.01, 'l1ratio': 0.001, 'lambdatc': 1},
                      # {'model': 'MTL-LR-weight', 'lr': 0.01, 'l1ratio': 0.01, 'lambdatc': 1},
                      # {'model': 'MTL-LR-weight', 'lr': 0.01, 'l1ratio': 0.1, 'lambdatc': 1},
                      # {'model': 'MTL-LR-weight', 'lr': 0.01, 'l1ratio': 0.0001, 'lambdatc': 10},
                      # {'model': 'MTL-LR-weight', 'lr': 0.01, 'l1ratio': 0.001, 'lambdatc': 10},
                      # {'model': 'MTL-LR-weight', 'lr': 0.01, 'l1ratio': 0.01, 'lambdatc': 10},
                      # {'model': 'MTL-LR-weight', 'lr': 0.01, 'l1ratio': 0.1, 'lambdatc': 10},
                      # {'model': 'MTL-SH-weight', 'lr': 0.01, 'l1ratio': 0.0001, 'lambdatc': 0.1},
                      # {'model': 'MTL-SH-weight', 'lr': 0.01, 'l1ratio': 0.001, 'lambdatc': 0.1},
                      # {'model': 'MTL-SH-weight', 'lr': 0.01, 'l1ratio': 0.01, 'lambdatc': 0.1},
                      # {'model': 'MTL-SH-weight', 'lr': 0.01, 'l1ratio': 0.1, 'lambdatc': 0.1},
                      # {'model': 'MTL-SH-weight', 'lr': 0.01, 'l1ratio': 0.0001, 'lambdatc': 1},
                      # {'model': 'MTL-SH-weight', 'lr': 0.01, 'l1ratio': 0.001, 'lambdatc': 1},
                      # {'model': 'MTL-SH-weight', 'lr': 0.01, 'l1ratio': 0.01, 'lambdatc': 1},
                      # {'model': 'MTL-SH-weight', 'lr': 0.01, 'l1ratio': 0.1, 'lambdatc': 1},
                      # {'model': 'MTL-SH-weight', 'lr': 0.01, 'l1ratio': 0.0001, 'lambdatc': 10},
                      # {'model': 'MTL-SH-weight', 'lr': 0.01, 'l1ratio': 0.001, 'lambdatc': 10},
                      # {'model': 'MTL-SH-weight', 'lr': 0.01, 'l1ratio': 0.01, 'lambdatc': 10},
                      # {'model': 'MTL-SH-weight', 'lr': 0.01, 'l1ratio': 0.1, 'lambdatc': 10}
    ]

    #########

    run(filename, w, n_pre_actions, dv_area, cmp_inv, dv_cmd, reject_data, pre_action, pre_action_onehot,
        pre_action_status, pre_dev_area, pre_dev_area_onehot, pre_cmp_inv, note_details, cmp_inv_freq, cmp_inv_weight,
        cmp_inv_listfile, enable_plot, cmp_inv_binary, cmp_inv_binary_count, w_0, dev_area_confidence_low, seed_value)
