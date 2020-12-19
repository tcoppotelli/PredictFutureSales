import pandas as pd
import functions as f
import holidays as h
import functions_test as t
import pickle
import time
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# Remember to change model and dataframe before running
original_df_location = '20201213-112015df.pickle.dat'
original_model = '20201214-093238RandomForestRegressor.pickle.dat'


final_df = t.create_base_submission()
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


infile = open(original_df_location, "rb")
df = pickle.load(infile)
infile.close()

final_df = t.add_lag(final_df, df)
print('df size with lag ', final_df.shape)


infile = open(original_model, "rb")
model = pickle.load(infile)
infile.close()
final_df = f.downcast_dtypes(final_df)

final_df['item_price'] = final_df['item_price'].fillna(0)
# f.add_city_nan(final_df)

y_pred = model.predict(
    final_df[['date_block_num', 'shop_id', 'item_id', 'item_price', 'item_category_id', 'city_id',
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
final_df.to_csv(timestr+'solution_rf.csv', index=False)
print("Your submission was successfully saved!")