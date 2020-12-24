import functions as f
import pandas as pd
import time
import pickle


pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)


def create_feature_names_list(df):
    """
    Function that builds a list of all features required to train/test model
    :arg df (pandas df) - train set df
    :return features (list) - list of feature names
    """
    features = ['date_block_num', 'shop_id', 'item_id', 'Year', 'Month', 'shop_type_1',
                'shop_type_2', 'shop_city_type', 'shop_city', 'item_category_id',
                'item_category_main', 'is_category_digital', 'is_category_ps_related', 'item_price_avg',
                'when_first_sold',
                'number_of_mondays', 'number_of_saturdays', 'number_of_sundays', 'number_of_days_in_month']
    lag_cols = [x for x in df.columns if 'lag' in x]
    features = features + lag_cols

    return features


def create_train_set(addition_to_filename = ''):
    """
    Function that creates a train set and saves it in pickle file for further use
    :arg addition_to_filename (str) - any additional information appended to filename
    :return pickle file with train set without mean encoding
    """
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
    pickle.dump(df, open(f"{timestr}_{addition_to_filename}_train.pickle.dat", "wb"))

    # save feature names for further use
    features_list = create_feature_names_list(df)
    pickle.dump(features_list, open(f"{timestr}_{addition_to_filename}_features.pickle.dat", "wb"))


if __name__ == '__main__':
    create_train_set('first_ver')