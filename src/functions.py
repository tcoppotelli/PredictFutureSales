import pandas as pd
import numpy as np
import shops as sh
import sales as sa
import item_category as ic


def create_df(use_cache=False):
    """
    This is a helper function that creates the train df.
    """

    sales = sa.prepare_sales(use_cache)

    shops = pd.read_csv("competitive-data-science-predict-future-sales/shops.csv")
    shops = sh.fix_shops(shops)  # fix the shops as we have seen before

    items_category = pd.read_csv("competitive-data-science-predict-future-sales/item_categories.csv")
    items_category = ic.fix_item_category(items_category)

    items = pd.read_csv("competitive-data-science-predict-future-sales/items.csv")
    items.drop(columns=['item_name'], inplace=True)

    merged_df = sales.merge(shops, on = 'shop_id', how = 'left')
    print('df size with zero sales ', merged_df.shape)

    items_to_merge = items.merge(items_category, on = 'item_category_id')
    merged_df = merged_df.merge(items_to_merge, on = 'item_id', how = 'left')

    merged_df = add_previous_months_sales(merged_df)
    print('df size with previous months ', merged_df.shape)

    return remove_nan(merged_df)


def calculate_missing_prices_for_train_set(df):
    average_price = df.sort_values(['date_block_num']).dropna(subset = ['item_price_avg'])
    average_price = average_price.drop_duplicates(subset = ['item_id'], keep = 'last')[['item_id','item_price_avg']]


    train_df_is_null = df.loc[df['item_price_avg'].isnull()]
    train_df_is_null.drop(columns = ['item_price_avg'], inplace = True)
    train_df_is_null = train_df_is_null.merge(average_price, on = 'item_id', how = 'left')

    train_df_is_not_null = df.loc[~df['item_price_avg'].isnull()]

    train_df_final = pd.concat([train_df_is_null, train_df_is_not_null], axis = 0)

    return train_df_final


def add_previous_months_sales(origin_df, list_lags=None):
    if list_lags is None:
        list_lags = [1, 2, 3]

    final_df = origin_df.copy()

    for month_shift in list_lags:
        shifted = origin_df.copy()
        shifted = shifted[['date_block_num', 'shop_id', 'item_id', 'item_cnt_month']]
        shifted.date_block_num = shifted.date_block_num + month_shift
        shifted.rename(columns={"item_cnt_month": "item_cnt_month_"+str(month_shift)}, inplace=True)
        final_df = pd.merge(final_df, shifted,  how='left', left_on=['date_block_num', 'shop_id', 'item_id'],
                            right_on=['date_block_num', 'shop_id', 'item_id'])
    del shifted

    return final_df.drop_duplicates()


def remove_nan(df):
    df['item_cnt_month'] = df['item_cnt_month'].fillna(0)
    df['item_cnt_month_1'] = df['item_cnt_month_1'].fillna(0)
    df['item_cnt_month_2'] = df['item_cnt_month_2'].fillna(0)
    df['item_cnt_month_3'] = df['item_cnt_month_3'].fillna(0)
    # df['holidays'] = df['holidays'].fillna(0)
    df['item_cnt_month'] = df['item_cnt_month'].fillna(0)

    return df


def downcast_dtypes(df):
    """
        Changes column types in the dataframe:

                `float64` type to `float32`
                `int64`   type to `int32`
    """

    # Select columns to downcast
    float_cols = [c for c in df if df[c].dtype == "float64"]
    int_cols = [c for c in df if df[c].dtype == "int64"]

    # Downcast
    df[float_cols] = df[float_cols].astype(np.float32)
    df[int_cols] = df[int_cols].astype(np.int32)

    return df

def construct_lag(df, colname, number_of_months=12):
    '''
    function that constructs lag
    :arg 
    df (pandas df) - train df after all transformation
    colname (list of str) - list of columns to build lag onto (example: ['shop_id'] or ['shop_id','item_id'])
    number_of_months (int) - number of months the lag will be calculated
    :return: df with additional lag columns. Important (!) the months where lag is impossible to calculate are
            thrown away (for example if number_of_months = 12 then all 2013 data is deleted)
    '''

    temp_df = df.copy()

    df = df.loc[df['date_block_num'] >= number_of_months]

    # calculate montly statistics over lag columns
    stat = pd.DataFrame(temp_df.groupby(['date_block_num', *colname])['item_cnt_month'].mean()).reset_index()
    stat['item_cnt_month'] = stat['item_cnt_month'].round(2)
    stat = downcast_dtypes(stat)

    for month_shift in range(1, number_of_months + 1):
        # rename a resulting column in the copy of stats for further merge
        stat_copy = stat.copy()
        new_colname = 'lag_m ean_{}_{}'.format('_'.join(colname), month_shift)
        stat_copy.rename(columns={'item_cnt_month': new_colname}, inplace=True)

        # merge a lagged column with original dataset
        df['temp_col'] = df['date_block_num'] - month_shift
        df = df.merge(stat_copy, left_on=['temp_col', *colname],
                      right_on=['date_block_num', *colname], how='left')

        # perform some final cleaning steps
        df.drop(columns=['date_block_num_y', 'temp_col'], inplace=True)
        df.rename(columns={'date_block_num_x': 'date_block_num'}, inplace=True)
        df[new_colname].fillna(0, inplace=True)

        print(colname, month_shift, df.shape)

    return df


def add_lag(df, number_of_months = 12):
    """ adds lag columns to original dataframe
    :arg df (pandas df) train df after all modifications
    :return df (pandas df) original df with lag columns
    """
    
    lag_columns_list = [
        ['shop_id', 'item_id'],
        ['shop_id', 'item_category_id'],
        ['shop_id'],
        ['item_id'],
        ['item_category_id'],
        ['item_category_main']
    ]

    df_with_lag = df.copy()
    for colname in lag_columns_list:
        df_with_lag = construct_lag(df_with_lag, colname, number_of_months=number_of_months)

    return df_with_lag