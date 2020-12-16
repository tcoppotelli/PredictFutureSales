import pandas as pd
import functions as f
import shops as sh
import item_category as ic

def fill_nan_price(final_df, df):
    mean_price = df[df.Year == 2015][['item_id', 'item_price']].groupby('item_id').mean()
    # null values final_df[final_df.item_price.isna()]
    final_df = pd.merge(final_df, mean_price, how='left',
                        left_on=['item_id'],
                        right_on=['item_id'])
    final_df['item_price'] = final_df['item_price_x'].fillna(final_df['item_price_y'])
    final_df.drop(columns=['item_price_x', 'item_price_y'], inplace=True)



def add_lag(final_df, df):
    block33 = df[df['date_block_num'] == 33].copy()
    block33 = block33[['shop_id', 'item_id', 'item_cnt_month']]
    block33.rename(columns={"item_cnt_month": "item_cnt_month_1"}, inplace=True)
    final_df = pd.merge(final_df, block33,  how='left',
                        left_on = ['shop_id', 'item_id'],
                        right_on = ['shop_id', 'item_id'])
    final_df['item_cnt_month_1'] = final_df['item_cnt_month_1'].fillna(0)
    final_df = final_df.drop_duplicates()
    final_df = final_df.reset_index(drop=True)

    block32 = df[df['date_block_num'] == 32].copy()
    block32 = block32[['shop_id', 'item_id', 'item_cnt_month']]
    block32.rename(columns={"item_cnt_month": "item_cnt_month_2"}, inplace=True)
    final_df = pd.merge(final_df, block32,  how='left',
                        left_on = ['shop_id', 'item_id'],
                        right_on = ['shop_id', 'item_id'])
    final_df['item_cnt_month_2'] = final_df['item_cnt_month_2'].fillna(0)
    final_df = final_df.drop_duplicates()
    final_df = final_df.reset_index(drop=True)

    block31 = df[df['date_block_num'] == 31].copy()
    block31 = block31[['shop_id', 'item_id', 'item_cnt_month']]
    block31.rename(columns={"item_cnt_month": "item_cnt_month_3"}, inplace=True)
    final_df = pd.merge(final_df, block31,  how='left',
                        left_on = ['shop_id', 'item_id'],
                        right_on = ['shop_id', 'item_id'])
    final_df['item_cnt_month_3'] = final_df['item_cnt_month_3'].fillna(0)
    final_df = final_df.drop_duplicates()
    final_df = final_df.reset_index(drop=True)

    return final_df


def create_test_df(train_df):

    test = pd.read_csv("competitive-data-science-predict-future-sales/test.csv")
    test['date_block_num'] = 34
    test['Year'] = 2015
    test['Month'] = 11

    test = add_lag(test, train_df)

    # load shops and preprocess it
    shops = pd.read_csv("competitive-data-science-predict-future-sales/shops.csv")
    shops = sh.fix_shops(shops)  # fix the shops as we have seen before

    # load item_category and preprocess it
    items_category = pd.read_csv("competitive-data-science-predict-future-sales/item_categories.csv")
    items_category = ic.fix_item_category(items_category)

    # load items
    items = pd.read_csv("competitive-data-science-predict-future-sales/items.csv")
    items.drop(columns = ['item_name'], inplace = True)

    # merge data
    items_to_merge = items.merge(items_category, on = 'item_category_id')
    test_merged = test.merge(shops, on = 'shop_id', how = 'left')
    test_merged = test_merged.merge(items_to_merge, on = 'item_id', how = 'left')

    return test_merged
#
#
# def create_base_submission():
#     shops = pd.read_csv("competitive-data-science-predict-future-sales/shops.csv")
#     f.fix_shops(shops)
#     shops.drop(columns=['shop_name_cleaned', 'city'], inplace=True)
#     shops = shops.drop_duplicates()
#
#     items = pd.read_csv("competitive-data-science-predict-future-sales/items.csv")
#     items.drop(columns=['item_name'], inplace=True)
#
#     sales = pd.read_csv("competitive-data-science-predict-future-sales/sales_train.csv")
#     # d = {0: 57, 1: 58, 10: 11, 23: 24, 39: 40}
#     # sales["shop_id"] = sales["shop_id"].apply(lambda x: d[x] if x in d.keys() else x)
#     sales = f.adjust_duplicated_shops(sales)
#
#     sales = f.remove_outliers(sales)
#     sales.drop(columns=['item_cnt_day', 'date'], inplace=True)
#     sales = sales.drop_duplicates()
#
#     test = pd.read_csv("competitive-data-science-predict-future-sales/test.csv")
#     print('test size ', test.shape)
#
#     unique_sales = sales.groupby(['shop_id', 'item_id']).tail(1).reset_index()
#
#     solution = pd.merge(test, unique_sales,  how='left', left_on=['shop_id', 'item_id'], right_on=['shop_id','item_id'])
#     print('1 solution size ', solution.shape)
#
#     solution = pd.merge(solution, shops,  how='left', left_on=['shop_id'], right_on=['shop_id'])
#     print('2 solution size ', solution.shape)
#
#     solution = pd.merge(solution, items,  how='left', left_on=['item_id'], right_on=['item_id'])
#     print('3 solution size ', solution.shape)
#
#     solution['date_block_num'] = 34
#     solution['Year'] = 2015
#     solution['Month'] = 11
#     solution = solution.set_index('ID')
#     solution.drop(columns=['index'], inplace=True)
#
#     return solution
#

def apply_0_to_not_sold_categories(df):

    items = pd.read_csv("competitive-data-science-predict-future-sales/items.csv")
    sales_raw = pd.read_csv("competitive-data-science-predict-future-sales/sales_train.csv")
    sales_raw = f.adjust_duplicated_shops(sales_raw)
    merged_df = sales_raw.merge(items[['item_id','item_category_id']], on = 'item_id')

    all_item_categories = list(merged_df['item_category_id'].unique())

    not_available_categories_per_shop = {}

    for shop_id in merged_df['shop_id'].unique():
        shop_item_categories = list(merged_df.loc[merged_df['shop_id'] == shop_id,'item_category_id'].unique())
        not_available_categories_per_shop[shop_id] = list(set(all_item_categories) - set(shop_item_categories))

    counter = 0
    for key, value in not_available_categories_per_shop.items():
        counter += len(df.loc[(df['shop_id'] == key) & (df['item_category_id'].isin(value)), 'item_cnt_month'])
        df.loc[(df['shop_id'] == key) & (df['item_category_id'].isin(value)), 'item_cnt_month'] = 0

    print(counter)
    return df

def add_price_col_to_test(test):

    # Algorithm:
    # 1. take the last price for the shop/item_id
    # 2. take the average price for the item_id
    # 3. take the median price for the category


    colnames = ['date_block_num', 'shop_id', 'item_id', 'Year', 'Month', 'shop_type_1',
                'shop_type_2', 'shop_city_type', 'shop_city', 'item_category_id',
                'item_category_main', 'is_category_digital', 'is_category_ps_related',
                'item_cnt_month_1', 'item_cnt_month_2', 'item_cnt_month_3',
                'item_price_avg']

    items = pd.read_csv("competitive-data-science-predict-future-sales/items.csv")
    sales_raw = pd.read_csv("competitive-data-science-predict-future-sales/sales_train.csv")
    sales_raw = f.adjust_duplicated_shops(sales_raw)

    last_price = sales_raw.sort_values(['shop_id','date_block_num'], ascending = [True, True])
    last_price = last_price.drop_duplicates(subset = ['shop_id','item_id'], keep = 'last')
    last_price = last_price[['shop_id','item_id','item_price']]
    last_price.columns = ['shop_id','item_id','item_price_avg']
    test_last_price = test.merge(last_price, on = ['shop_id','item_id'], how = 'left').dropna(subset = ['item_price_avg'])
    test_last_price_rest = test.loc[~test['ID'].isin(test_last_price['ID'])]

    mean_item_price = pd.DataFrame(sales_raw.groupby('item_id')['item_price'].median()).reset_index()
    mean_item_price.columns = ['item_id','item_price_avg']
    test_mean_item = test_last_price_rest.merge(mean_item_price, on = 'item_id', how = 'left').dropna(subset = ['item_price_avg'])
    id_list = list(test_mean_item['ID']) + list(test_last_price['ID'])
    test_mean_item_rest = test.loc[~test['ID'].isin(id_list)]

    sales_items = sales_raw.merge(items[['item_id','item_category_id']], on = 'item_id')

    median_item_cat_price = pd.DataFrame(sales_items.groupby('item_category_id')['item_price'].median()).reset_index()
    median_item_cat_price.columns = ['item_category_id','item_price_avg']

    test_category_price = test_mean_item_rest.merge(median_item_cat_price, on = 'item_category_id', how = 'left')

    test_with_price = pd.concat([test_last_price,test_mean_item,test_category_price], axis = 0)

    test_with_price.sort_values('ID', inplace = True)

    test_with_price.set_index('ID', inplace = True)

    test_with_price = test_with_price[colnames]

    return test_with_price
