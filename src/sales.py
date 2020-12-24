import numpy as np
import pandas as pd
import functions as f
from itertools import product
import datetime


def prepare_sales_monthly(sales):
    print('prepare_sales_monthly')
    sales_train_df_no_dupl_shops = f.adjust_duplicated_shops(sales)

    sales_train_montly_df = sales_train_df_no_dupl_shops.groupby(
        ['date_block_num', 'shop_id', 'item_id'])['item_price', 'item_cnt_day', 'date'].agg(
        {'item_price': 'mean',
         'date': 'min',
         'item_cnt_day': 'sum'})
    sales_train_montly_df = sales_train_montly_df.reset_index()

    colnames = ['date_block_num', 'shop_id', 'item_id', 'item_price_avg','date_min','item_cnt_month']
    sales_train_montly_df.columns = colnames

    print('sales_train_montly_df has shape ', sales_train_montly_df.shape)
    return sales_train_montly_df


def prepare_sales():
    print('prepare_sales')
    # load and preprocess sales
    sales = pd.read_csv("competitive-data-science-predict-future-sales/sales_train.csv")
    sales = f.remove_outliers(sales)

    first_items_sales = sales.groupby('item_id')['date_block_num'].min()

    sales = prepare_sales_monthly(sales)
    # matrix
    matrix = create_matrix(sales)
    zero_sales = add_zero_sales(sales, matrix)

    zero_sales['when_first_sold'] = zero_sales['item_id'].map(first_items_sales)
    zero_sales['when_first_sold'] = zero_sales['date_block_num'] - zero_sales['when_first_sold']

    print('prepare_sales has shape ', zero_sales.shape)
    return zero_sales


def create_matrix(df):
    x = datetime.date(2013, 1, 1)
    matrix = []
    cols = ["date_block_num", "shop_id", "item_id", "Year", "Month"]
    for i in range(34):
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