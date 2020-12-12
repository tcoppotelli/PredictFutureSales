import pandas as pd
import functions as f
import holidays as h
import os
import pickle
import time
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)


def create_base_submission():
    shops = pd.read_csv("competitive-data-science-predict-future-sales/shops.csv")
    f.fix_shops(shops) # fix the shops as we have seen before
    items = pd.read_csv("competitive-data-science-predict-future-sales/items.csv")
    sales = pd.read_csv("competitive-data-science-predict-future-sales/sales_train.csv")
    test = pd.read_csv("competitive-data-science-predict-future-sales/test.csv")
    print('test size ', test.shape)

    items.drop(columns=['item_name'], inplace=True)
    shops.drop(columns=['shop_name_cleaned', 'city'], inplace=True)
    shops = shops.drop_duplicates()

    sales = f.remove_outliers(sales)
    sales.drop(columns=['item_cnt_day', 'date'], inplace=True)
    sales = sales.drop_duplicates()
    unique_sales = sales.groupby(['shop_id', 'item_id']).tail(1).reset_index()

    solution = pd.merge(test, unique_sales,  how='left', left_on=['shop_id', 'item_id'], right_on=['shop_id','item_id'])
    print('1 solution size ', solution.shape)

    solution = pd.merge(solution, shops,  how='left', left_on=['shop_id'], right_on=['shop_id'])
    print('2 solution size ', solution.shape)

    solution = pd.merge(solution, items,  how='left', left_on=['item_id'], right_on=['item_id'])
    print('3 solution size ', solution.shape)

    solution['date_block_num'] = 34
    solution['Year'] = 2015
    solution['Month'] = 11
    solution = solution.set_index('ID')
    solution.drop(columns=['index'], inplace=True)

    return solution


final_df = create_base_submission()
print('df base size ', final_df.shape)

final_df = h.add_holidays(final_df)

final_df['Year'] = final_df['Year'].astype(int)
final_df['Month'] = final_df['Month'].astype(int)
final_df['holidays'] = final_df['holidays'].fillna(0)
final_df['holidays'] = final_df['holidays'].astype(int)
final_df['Year'] = final_df['Year'].astype(int)
final_df['Month'] = final_df['Month'].astype(int)
final_df['holidays'] = final_df['holidays'].fillna(0)
final_df['holidays'] = final_df['holidays'].astype(int)

print('df size with holiday ', final_df.shape)


infile = open('20201212-153401df.pickle.dat', "rb")
df = pickle.load(infile)
infile.close()


def add_lag(final_df):
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


final_df = add_lag(final_df)
print('df size with lag ', final_df.shape)


infile = open('20201212-154951model.pickle.dat', "rb")
model = pickle.load(infile)
infile.close()
final_df = f.downcast_dtypes(final_df)

y_pred = model.predict(final_df[['date_block_num', 'shop_id', 'item_id', 'item_price', 'item_category_id', 'city_id',
                                 'Year', 'Month', 'holidays', 'item_cnt_month_1', 'item_cnt_month_2', 'item_cnt_month_3']])

final_df['item_cnt_month'] = y_pred.clip(0, 20)
final_df.drop(columns=['date_block_num', 'shop_id', 'item_id', 'item_price', 'item_category_id', 'city_id', 'Year',
                       'Month', 'holidays',
                       'item_cnt_month_1', 'item_cnt_month_2', 'item_cnt_month_3'], inplace=True)

final_df = final_df.reset_index()
final_df.rename(columns={"index": "ID"}, inplace=True)

print('Expect (214200, 2)')
print('Actual ', final_df.shape)

timestr = time.strftime("%Y%m%d-%H%M%S")
final_df.to_csv(timestr+'solution_xgboost.csv', index=False)
print("Your submission was successfully saved!")