import pandas as pd
import functions as f
import holidays as h
import functions_test as t
import pickle
import time
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# Remember to change model and dataframe before running
original_df_location = '20201216-163745df.pickle.dat'
original_model = '20201216-171150model.pickle.dat'

infile = open(original_df_location, "rb")
df = pickle.load(infile)
infile.close()

test = t.create_test_df(df)
print('df base size ', test.shape)

test_with_price = t.add_price_col_to_test(test)
test_with_price = f.downcast_dtypes(test_with_price)


infile = open(original_model, "rb")
model = pickle.load(infile)
infile.close()


y_pred = model.predict(test_with_price)
test_with_price['item_cnt_month'] = y_pred.clip(0, 20)

submission = test_with_price.copy()

submission = t.apply_0_to_not_sold_categories(submission)
submission = submission[['item_cnt_month']].reset_index()


print('Expect (214200, 2)')
print('Actual ', submission.shape)

timestr = time.strftime("%Y%m%d-%H%M%S")
submission.to_csv(timestr+'solution_xgboost.csv', index=False)
print("Your submission was successfully saved!")