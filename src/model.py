import functions as f
import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb
import time
import pickle
from xgboost import plot_importance


pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

df = f.create_df()
print('original df size is ', df.shape)
print('original df columns ', df.columns)

df = f.calculate_missing_prices_for_train_set(df)
print('df size after averaging price ', df.shape)
df = f.downcast_dtypes(df)
df = f.add_lag(all_data_df = df, df_to_add_lag= df, number_of_months=3)
df = f.add_days_stat(df)


print(df.columns)
# df = h.add_holidays(df)
# print('df size with holidays ', df.shape)

timestr = time.strftime("%Y%m%d-%H%M%S")
pickle.dump(df, open(timestr+"df.pickle.dat", "wb"))


features = ['date_block_num', 'shop_id', 'item_id', 'Year', 'Month', 'shop_type_1',
            'shop_type_2', 'shop_city_type', 'shop_city', 'item_category_id',
            'item_category_main', 'is_category_digital', 'is_category_ps_related', 'item_price_avg','when_first_sold',
            'number_of_mondays','number_of_saturdays','number_of_sundays','number_of_days_in_month']
lag_cols = [x for x in df.columns if 'lag' in x]
features = features + lag_cols

print('Number of features Train_set: ', len(features))
print('Features: ', features)

target = ['item_cnt_month']

train = df[(df["date_block_num"] < 33)]
test = df[(df["date_block_num"] >= 33)]
X_train = train[features]
X_test = test[features]
Y_train = train[target]
Y_test = test[target]

X_train['Year'] = X_train['Year'].astype(int)
X_train['Month'] = X_train['Month'].astype(int)
X_test['Year'] = X_test['Year'].astype(int)
X_test['Month'] = X_test['Month'].astype(int)


ts = time.time()

model = xgb.XGBRegressor(
    max_depth=10,
    n_estimators=200,
    min_child_weight=0.5,
    colsample_bytree=0.8,
    subsample=0.8,
    eta=0.1,
    seed=42)

model.fit(
    X_train,
    Y_train,
    eval_metric="rmse",
    eval_set=[(X_train, Y_train), (X_test, Y_test)],
    verbose=True,
    early_stopping_rounds=20)

print("took ", time.time() - ts)


timestr = time.strftime("%Y%m%d-%H%M%S")
pickle.dump(model, open(timestr+"model.pickle.dat", "wb"))

# 20201214-215722model.pickle.dat
# [49]	validation_0-rmse:0.83141	validation_1-rmse:0.79650

# 20201214-231219model.pickle.dat
# [57]	validation_0-rmse:0.89769	validation_1-rmse:0.89495

# 20201217-175331model.pickle.dat
# [80]	validation_0-rmse:0.81682	validation_1-rmse:0.79703

def plot_features(booster, figsize):
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    return plot_importance(booster=booster, ax=ax)


plot_features(model, (10, 14))
plt.show()
