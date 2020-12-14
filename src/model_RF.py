import pandas as pd
import functions as f
import holidays as h
import os
import pickle
import time
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# Remember to change model and dataframe before running
original_df_location = '20201213-112015df.pickle.dat'
infile = open(original_df_location, "rb")
df = pickle.load(infile)
infile.close()

df['item_price'] = df['item_price'].fillna(0)

features = ['date_block_num', 'shop_id', 'item_id', 'item_price', 'item_category_id', 'city_id', 'Year', 'Month',
            'holidays', 'item_cnt_month_1', 'item_cnt_month_2','item_cnt_month_3']

target = ['item_cnt_month']

train = df[(df["date_block_num"] < 33)]
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


# train random forest
from sklearn.ensemble import RandomForestRegressor
import numpy as np
X = X_train.append(X_test)
Y = np.append(Y_train, Y_test)
rf = RandomForestRegressor(bootstrap=0.7, criterion='mse', max_depth=10,
                           max_features=6, max_leaf_nodes=None, min_impurity_decrease=0.0,
                           min_impurity_split=None, min_samples_leaf=1,
                           min_samples_split=2, min_weight_fraction_leaf=0.0,
                           n_estimators=10, n_jobs=4, oob_score=False, random_state=None,
                           verbose=1, warm_start=False)
rf.fit(X,Y)

timestr = time.strftime("%Y%m%d-%H%M%S")
pickle.dump(rf, open(timestr+"RandomForestRegressor.pickle.dat", "wb"))