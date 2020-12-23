import numpy as np
import pandas as pd
from itertools import product
import datetime
import pickle


def prepare_sales_monthly(sales):
    print('prepare_sales_monthly')
    sales_train_df_no_dupl_shops = adjust_duplicated_shops(sales)

    sales_train_montly_df = sales_train_df_no_dupl_shops.groupby(
        ['date_block_num', 'shop_id', 'item_id'])['item_price', 'item_cnt_day', 'date'].agg(
        {'item_price': 'mean',
         'date': 'min',
         'item_cnt_day': 'sum'})
    sales_train_montly_df = sales_train_montly_df.reset_index()

    colnames = ['date_block_num', 'shop_id', 'item_id', 'item_price_avg', 'date_min', 'item_cnt_month']
    sales_train_montly_df.columns = colnames

    print('sales_train_montly_df has shape ', sales_train_montly_df.shape)
    return sales_train_montly_df


def adjust_duplicated_shops(df):
    'Function that combines duplicated shop names'
    # from https://www.kaggle.com/taranenkodaria/predict-future-sales-the-russian-forecast
    # Test Set unique shop_id --> we should only use these ids
    # array([ 2,  3,  4,  5,  6,  7, 10, 12, 14, 15, 16, 18, 19, 21, 22, 24, 25,
    #   26, 28, 31, 34, 35, 36, 37, 38, 39, 41, 42, 44, 45, 46, 47, 48, 49,
    #   50, 52, 53, 55, 56, 57, 58, 59], dtype=int64)

    df.loc[df['shop_id'] == 0, 'shop_id'] = 57
    df.loc[df['shop_id'] == 1, 'shop_id'] = 58
    df.loc[df['shop_id'] == 11, 'shop_id'] = 10
    df.loc[df['shop_id'] == 40, 'shop_id'] = 39
    df.loc[df['shop_id'] == 23, 'shop_id'] = 24

    return df


def remove_outliers(df):
    return df[(df["item_price"] < np.percentile(df["item_price"], q=99))
              & (df["item_price"] > 0)
              & (df["item_cnt_day"] >= 0)
              & (df["item_cnt_day"] < np.percentile(df["item_cnt_day"], q=99))]


def prepare_sales(use_cache):
    if use_cache:
        try:
            infile = open("sales_df.pickle.dat", "rb")
            sales = pickle.load(infile)
            infile.close()
            return sales
        except (OSError, IOError) as e:
            pass

    print('prepare_sales')
    # load and preprocess sales
    sales = pd.read_csv("competitive-data-science-predict-future-sales/sales_train.csv")
    sales = remove_outliers(sales)
    sales = prepare_sales_monthly(sales)
    sales = add_zero_sales(sales, None)

    print('prepare_sales has shape ', sales.shape)
    if use_cache:
        pickle.dump(sales, open("sales_df.pickle.dat", "wb"))
    return sales


def create_matrix(df):
    x = datetime.date(2013, 1, 1)
    matrix = []
    cols = ["date_block_num", "shop_id", "item_id", "Year", "Month"]
    for i in range(df.date_block_num.max() + 1):
        try:
            sales = df[df.date_block_num == i]
            matrix.append(np.array(list(product([i], sales.shop_id.unique(), sales.item_id.unique(), [x.year], [x.month])), dtype=np.int16))
            x = x.replace(month=x.month+1)
        except ValueError:
            if x.month == 12:
                x = x.replace(year=x.year+1, month=1)
            else:
                # next month is too short to have "same date"
                # pick your own heuristic, or re-raise the exception:
                raise

    matrix = pd.DataFrame(np.vstack(matrix), columns = cols )
    matrix["date_block_num"] = matrix["date_block_num"].astype(np.int8)
    matrix["shop_id"] = matrix["shop_id"].astype(np.int8)
    matrix["item_id"] = matrix["item_id"].astype(np.int16)
    matrix.sort_values(cols, inplace=True)

    return matrix


def add_zero_sales(df, matrix):
    if matrix is None:
        matrix = create_matrix(df)

    # df.drop(columns=["Year", "Month"], inplace=True)
    df = pd.merge(matrix, df, how='left',
                  left_on=['date_block_num', 'shop_id', 'item_id'],
                  right_on=['date_block_num', 'shop_id', 'item_id'])

    return df