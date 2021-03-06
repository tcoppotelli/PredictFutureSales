import pandas as pd
import sales as s
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


def add_when_first_sold_to_test(test):
    """ads a 'when_first_sold column to test df'"""
    sales = pd.read_csv("competitive-data-science-predict-future-sales/sales_train.csv")
    sales = s.remove_outliers(sales)
    first_items_sales = sales.groupby('item_id')['date_block_num'].min()
    test['when_first_sold'] = test['item_id'].map(first_items_sales)
    test['when_first_sold'] = test['date_block_num'] - test['when_first_sold']
    test['when_first_sold'].fillna(0, inplace=True)

    return test


def create_test_df(train_df):

    test = pd.read_csv("competitive-data-science-predict-future-sales/test.csv")
    test['date_block_num'] = 34
    test['Year'] = 2015
    test['Month'] = 11

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


def apply_0_to_not_sold_categories(df):

    items = pd.read_csv("competitive-data-science-predict-future-sales/items.csv")
    sales_raw = pd.read_csv("competitive-data-science-predict-future-sales/sales_train.csv")
    sales_raw = s.adjust_duplicated_shops(sales_raw)
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

def correct_submission_for_not_sold_items(test_to_correct):
    """
    function that checks whether particular item_id is still sold in shops.
    arg: path_to_submission_to_improve (str) - path to submission file
         path_to_sales_train (str) - path to sales_train.csv file

    return: corrected_submission (pandas df) - an adjusted submission
    """

    shop_sales = pd.read_csv("competitive-data-science-predict-future-sales/sales_train.csv")

    items_not_sold_in_last_3_months = {}

    test = pd.read_csv("competitive-data-science-predict-future-sales/test.csv")

    all_test_items = test.item_id.unique()

    for shop_id in test['shop_id'].unique():
        shop = shop_sales.loc[(shop_sales['shop_id'] == shop_id) & (shop_sales['item_id'].isin(all_test_items))]
        shop_items = list(shop['item_id'].unique())
        shop_items_3_months = list(shop.loc[shop['date_block_num'] > 30, 'item_id'].unique())

        items_not_sold_in_last_3_months[shop_id] = list(set(shop_items) - set(shop_items_3_months))

    test_work = test[['ID', 'shop_id', 'item_id']]

    test_work_done = pd.DataFrame()

    for key, value in items_not_sold_in_last_3_months.items():
        test_df = test_work.loc[test_work['shop_id'] == key]
        test_df['coeff'] = 1
        test_df.loc[test_df['item_id'].isin(value),'coeff'] = 0
        test_work_done = pd.concat([test_work_done, test_df], axis = 0)

    test_to_correct_1 = test_to_correct.merge(test_work_done, on = 'ID', how = 'left')

    # test_to_correct_1.fillna(1, inplace=True)

    test_to_correct_1['item_cnt_month'] = test_to_correct_1['item_cnt_month'] * test_to_correct_1['coeff']

    corrected_submission = test_to_correct_1[['ID','item_cnt_month']]

    return corrected_submission

def add_price_col_to_test(test, regime = 'test'):

    # Algorithm:
    # 1. take the last price for the shop/item_id
    # 2. take the average price for the item_id
    # 3. take the median price for the category

    items = pd.read_csv("competitive-data-science-predict-future-sales/items.csv")
    sales_raw = pd.read_csv("competitive-data-science-predict-future-sales/sales_train.csv")
    sales_raw = s.adjust_duplicated_shops(sales_raw)
    if regime == 'val':
        sales_raw = sales_raw.loc[sales_raw['date_block_num'] < 33]

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

    return test_with_price
