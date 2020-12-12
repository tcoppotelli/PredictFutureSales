from sklearn.preprocessing import LabelEncoder
from string import punctuation
import pandas as pd
import numpy as np
from itertools import product
import datetime


def fix_shops(shops):
    """
    This function modifies the shops df inplace.
    It correct's 3 shops that we have found to be 'duplicates'
    and also creates a few more features: extracts the city and encodes it using LabelEncoder
    """

    d = {0: 57, 1: 58, 10: 11, 23: 24}

    # this 'tricks' allows you to map a series to a dictionary, but all values that are not in the dictionary won't
    # be affected it's handy since if we blindly map the values, the missing values will be replaced with nan
    shops["shop_id"] = shops["shop_id"].apply(lambda x: d[x] if x in d.keys() else x)

    # replace all the punctuation in the shop_name columns
    shops["shop_name_cleaned"] = shops["shop_name"].apply(lambda s: "".join([x for x in s if x not in punctuation]))

    # extract the city name
    shops["city"] = shops["shop_name_cleaned"].apply(lambda s: s.split()[0])

    # encode it using a simple LabelEncoder
    shops["city_id"] = LabelEncoder().fit_transform(shops['city'])

    shops.drop(columns=['shop_name'], inplace=True)


# a simple function that creates a global df with all joins and also shops corrections
def create_df():
    """
    This is a helper function that creates the train df.
    """
    # import all df
    shops = pd.read_csv("competitive-data-science-predict-future-sales/shops.csv")
    fix_shops(shops)  # fix the shops as we have seen before

    items_category = pd.read_csv("competitive-data-science-predict-future-sales/item_categories.csv")
    items = pd.read_csv("competitive-data-science-predict-future-sales/items.csv")
    sales = pd.read_csv("competitive-data-science-predict-future-sales/sales_train.csv")

    # fix shop_id in sales so that we can later merge the df
    d = {0: 57, 1: 58, 10: 11, 23: 24}
    sales["shop_id"] = sales["shop_id"].apply(lambda x: d[x] if x in d.keys() else x)

    # create df by merging the previous dataframes
    df = pd.merge(items, items_category, left_on="item_category_id", right_on="item_category_id")
    df = pd.merge(sales, df, left_on="item_id", right_on="item_id")
    df = pd.merge(df, shops, left_on="shop_id", right_on="shop_id")

    # convert to datetime and sort the values
    #     df["date"] = pd.to_datetime(df["date"], format = "%d.%m.%Y")
    df.sort_values(by=["shop_id", "date"], ascending=True, inplace=True)

    return df


def remove_outliers(df):
    return df[(df["item_price"] < np.percentile(df["item_price"], q=99))
              & (df["item_price"] > 0)
              & (df["item_cnt_day"] >= 0)
              & (df["item_cnt_day"] < np.percentile(df["item_cnt_day"], q=99))]


def add_date_and_count(original_df):
    original_df["date"] = pd.to_datetime(original_df["date"], format="%d.%m.%Y")
    original_df["Year"] = original_df["date"].dt.year
    original_df["Month"] = original_df["date"].dt.month

    simple_df = original_df[['date_block_num', 'shop_id', 'item_id', 'item_cnt_day']]
    grouped_df = simple_df.groupby(['date_block_num', 'shop_id', 'item_id']).sum()
    # remove item_cnt_day
    original_df = original_df[
        ['date_block_num', 'shop_id', 'item_id', 'item_price', 'item_name', 'item_category_id', 'item_category_name',
         'shop_name_cleaned', 'city', 'city_id', 'Year', 'Month']].drop_duplicates()
    final_df = pd.merge(original_df, grouped_df, left_on=['date_block_num', 'shop_id', 'item_id'],
                        right_on=['date_block_num', 'shop_id', 'item_id'])
    final_df.rename(columns={"item_cnt_day": "item_cnt_month"}, inplace=True)
    return final_df


def change_price_to_average(original_df):
    grouped_df = original_df[['date_block_num', 'shop_id', 'item_id','item_price']].groupby(['date_block_num', 'shop_id', 'item_id']).mean()
    final_df = pd.merge(original_df, grouped_df, left_on=['date_block_num', 'shop_id', 'item_id'],
                    right_on=['date_block_num', 'shop_id', 'item_id'])
    final_df.drop(columns=['item_price_x'], inplace=True)
    final_df.rename(columns={"item_price_y": "item_price"}, inplace=True)
    return final_df.drop_duplicates()


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

    df.drop(columns=["Year", "Month"], inplace=True)
    df = pd.merge(matrix, df, how='left',
                  left_on=['date_block_num', 'shop_id', 'item_id'],
                  right_on=['date_block_num', 'shop_id', 'item_id'])

    return df


def remove_nan(df):
    df['item_cnt_month'] = df['item_cnt_month'].fillna(0)
    df['item_cnt_month_1'] = df['item_cnt_month_1'].fillna(0)
    df['item_cnt_month_2'] = df['item_cnt_month_2'].fillna(0)
    df['item_cnt_month_3'] = df['item_cnt_month_3'].fillna(0)
    df['holidays'] = df['holidays'].fillna(0)
    df['item_cnt_month'] = df['item_cnt_month'].fillna(0)


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

