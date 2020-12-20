import pandas as pd
import numpy as np
from itertools import product
import pickle

from sklearn.model_selection import KFold

sales = pd.read_csv('competitive-data-science-predict-future-sales/sales_train.csv')

index_cols = ['shop_id', 'item_id', 'date_block_num']

# For every month we create a grid from all shops/items combinations from that month
grid = []
for block_num in sales['date_block_num'].unique():
    cur_shops = sales[sales['date_block_num']==block_num]['shop_id'].unique()
    cur_items = sales[sales['date_block_num']==block_num]['item_id'].unique()
    grid.append(np.array(list(product(*[cur_shops, cur_items, [block_num]])),dtype='int32'))

#turn the grid into pandas dataframe
grid = pd.DataFrame(np.vstack(grid), columns = index_cols,dtype=np.int32)

#get aggregated values for (shop_id, item_id, month)
gb = sales.groupby(index_cols,as_index=False).agg({'item_cnt_day':'sum'}).rename(columns={'item_cnt_day': 'target'})

#fix column names
gb.columns = [col[0] if col[-1]=='' else col[-1] for col in gb.columns.values]
#join aggregated data to the grid
all_data = pd.merge(grid,gb,how='left',on=index_cols).fillna(0)
#sort the data
all_data.sort_values(['date_block_num','shop_id','item_id'],inplace=True)



# BASIC METHOD
'''
     Differently to `.target.mean()` function `transform` 
   will return a dataframe with an index like in `all_data`.
   Basically this single line of code is equivalent to the first two lines from of Method 1.
'''
# Calculate a mapping: {item_id: target_mean}
item_id_target_mean = all_data.groupby('item_id').target.mean()

# In our non-regularized case we just *map* the computed means to the `item_id`'s
all_data['item_target_enc'] = all_data['item_id'].map(item_id_target_mean)
# all_data['item_target_enc'] = all_data.groupby('item_id')['target'].transform('mean')

# Fill NaNs
all_data['item_target_enc'].fillna(0.3343, inplace=True)

# Print correlation
encoded_feature = all_data['item_target_enc'].values
print(np.corrcoef(all_data['target'].values, encoded_feature)[0][1])

pickle.dump(all_data, open("all_data.pickle.dat", "wb"))

# 1. KFold scheme
# First, implement KFold scheme with five folds. Use KFold(5) from sklearn.model_selection.
# Split your data in 5 folds with sklearn.model_selection.KFold with shuffle=False argument.
y_tr = all_data['target'].values
kf = KFold(n_splits=5, shuffle=False)
for train_index, test_index in kf.split(all_data):
    # print("TRAIN:", train_index.shape, "TEST:", test_index.shape)
    x_tr, x_val = all_data.iloc[train_index], all_data.iloc[test_index]
    item_enc = x_tr.groupby('item_id').target.mean()
    all_data['item_target_enc'].iloc[test_index] = x_val['item_id'].map(item_enc)

# Fill NaNs
all_data['item_target_enc'].fillna(0.3343, inplace=True)

encoded_feature = all_data['item_target_enc'].values
print(np.corrcoef(all_data['target'].values, encoded_feature)[0][1])

infile = open("all_data.pickle.dat", "rb")
all_data = pickle.load(infile)
infile.close()


# 2. Live one out
# To implement a faster version, note, that to calculate mean target value using all the objects but one given object, you can:
#
# 1 Calculate sum of the target values using all the objects.
item_id_target_sum = all_data.groupby('item_id').target.sum()
item_id_target_count = all_data.groupby('item_id').target.count()
# 2 Then subtract the target of the given object and divide the resulting value by n_objects - 1.
# all_data['item_target_enc'] = all_data.apply(lambda x: ((item_id_target_sum.iloc[int(x['item_id'])] - x['target'])/(item_id_target_count.iloc[int(x['item_id'])] - 1)), axis=1)
all_data['item_target_enc'] = (all_data['item_id'].map(item_id_target_sum) - all_data['target']) / (all_data['item_id'].map(item_id_target_count) - 1)

# 3. Smoothing
# Next, implement smoothing scheme with  α=100 . Use the formula from the first slide in the video and  0.3343 as globalmean.
# Note that nrows is the number of objects that belong to a certain category (not the number of rows in the dataset).
item_id_target_mean = all_data.groupby('item_id').target.mean()
item_id_target_mean.fillna(0.3343, inplace=True)


item_id_target_sum = all_data.groupby('item_id').target.sum()
item_id_target_mean_weight = item_id_target_mean * item_id_target_sum

alpha = 100
reg = 0.3343 * alpha

all_data['item_target_enc'] = (all_data['item_id'].map(item_id_target_mean_weight) + reg) / (all_data['item_id'].map(item_id_target_sum) + alpha)


# 4. Expanding mean scheme
cumsum = all_data.groupby('item_id').target.cumsum() - all_data['target']
cumcnt = all_data.groupby('item_id').cumcount()
all_data['item_target_enc'] = cumsum/cumcnt
# Fill NaNs
all_data['item_target_enc'].fillna(0.3343, inplace=True)
