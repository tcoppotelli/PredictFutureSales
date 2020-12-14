import functions as f
import holidays as h
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb
import time
import pickle
from xgboost import plot_importance


pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

df = f.create_df()
print('original df size is ', df.shape)

showEDA = False

if showEDA:
    # remove outliers
    plt.figure(figsize=(10, 4))
    sns.boxplot(x=df["item_cnt_day"])
    plt.show()

    # let's look at outliers for item price
    plt.figure(figsize=(10, 4))
    plt.xlim(df["item_price"].min(), df["item_price"].max()*1.1)
    sns.boxplot(x=df["item_price"])
    plt.show()

    plt.figure(figsize=(10, 4))
    sns.jointplot(x="item_price", y="item_cnt_day", data=df)
    plt.show()

df = f.remove_outliers(df)
print('df size without outliers is ', df.shape)

if showEDA:
    plt.figure(figsize=(10, 4))
    sns.jointplot(x="item_price", y="item_cnt_day", data=df)
    plt.show()

df = f.add_date_and_count(df)
print('df size with date grouped by month is ', df.shape)

df = f.change_price_to_average(df)
print('df size after averaging price ', df.shape)

df = h.add_holidays(df)
print('df size with holidays ', df.shape)

# TODO: Move this matrix somewhere else
# matrix = f.create_matrix(df)
# df = f.add_zero_sales(df, matrix)
# print('df size with zero sales ', df.shape)

df = f.add_previous_months_sales(df)
print('df size with previous months ', df.shape)


f.remove_nan(df)
print('df size with zero sales ', df.shape)

df = f.downcast_dtypes(df)


df = f.add_category_and_city_nan(df)

timestr = time.strftime("%Y%m%d-%H%M%S")
pickle.dump(df, open(timestr+"df.pickle.dat", "wb"))


features = ['date_block_num', 'shop_id', 'item_id', 'item_price', 'item_category_id', 'city_id',
            'Year', 'Month', 'holidays', 'item_cnt_month_1', 'item_cnt_month_2','item_cnt_month_3']

target = ['item_cnt_month']

train = df[(df["date_block_num"] < 33) & (df["date_block_num"] > 12)]
test = df[(df["date_block_num"] >= 33)]
X_train = train[features]
X_test = test[features]
Y_train = train[target]
Y_test = test[target]

X_train['Year'] = X_train['Year'].astype(int)
X_train['Month'] = X_train['Month'].astype(int)
X_train['holidays'] = X_train['holidays'].fillna(0)
X_train['holidays'] = X_train['holidays'].astype(int)
X_test['Year'] = X_test['Year'].astype(int)
X_test['Month'] = X_test['Month'].astype(int)
X_test['holidays'] = X_test['holidays'].fillna(0)
X_test['holidays'] = X_test['holidays'].astype(int)


ts = time.time()

# Randomized search time: 2563.608281135559
# Best score: -14.475570268737748
# Best params:
# base_score: 0.75
# booster: 'gbtree'
# learning_rate: 0.2
# max_depth: 10
# min_child_weight: 2
# n_estimators: 2100

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

time.time() - ts


timestr = time.strftime("%Y%m%d-%H%M%S")
pickle.dump(model, open(timestr+"model.pickle.dat", "wb"))

# 20201212-154951model.pickle.dat
# [62]	validation_0-rmse:0.91334	validation_1-rmse:0.77302

# 20201213-094055model.pickle.dat
# [75]	validation_0-rmse:0.90515	validation_1-rmse:0.80479

# 20201213-104633model.pickle.dat
# [77]	validation_0-rmse:0.92198	validation_1-rmse:0.80973

# 20201213-113657model.pickle.dat
# [70]	validation_0-rmse:1.04922	validation_1-rmse:0.92126

# 20201214-215722model.pickle.dat
# [49]	validation_0-rmse:0.83141	validation_1-rmse:0.79650

# 20201214-231219model.pickle.dat
# [57]	validation_0-rmse:0.89769	validation_1-rmse:0.89495


def plot_features(booster, figsize):
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    return plot_importance(booster=booster, ax=ax)


plot_features(model, (10, 14))
plt.show()
