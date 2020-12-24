import models_library as m_l
import functions as f
import pickle
import time
import matplotlib.pyplot as plt


def create_train_val_split(df, features):
    """
    Function that splits the df into train / val splits
    :param df: (pandas df) all training data loaded from pickle file
    :param features: (list) oll feature names loaded from pickle file
    :return: train/val splits
    """
    print('Number of features Train_set: ', len(features))
    print('Features: ', features)

    target = ['item_cnt_month']

    train = df[(df["date_block_num"] < 33)]
    test = df[(df["date_block_num"] >= 33)]
    X_train = train[features]
    X_val = test[features]
    Y_train = train[target]
    Y_val = test[target]

    X_train['Year'] = X_train['Year'].astype(int)
    X_train['Month'] = X_train['Month'].astype(int)
    X_val['Year'] = X_val['Year'].astype(int)
    X_val['Month'] = X_val['Month'].astype(int)

    return X_train, Y_train, X_val, Y_val


def train_model(model, datasets, is_save = False, is_plot_features = True):
    """
    Function that trains a model with both train and validation splits
    :arg model (xgboost model) model object from models_library
         datasets (tuple of pandas dfs (X_xx) or pandas Series (Y_xx)) - train and validation splits and targets
         is_save (bool) - whether to save the model or not
         is_plot_features (bool) - whether to plot the feature importance
    :return model (xgboost model) trained model
    """

    X_train, Y_train, X_val, Y_val = datasets

    ts = time.time()

    model.fit(
        X_train,
        Y_train,
        eval_metric="rmse",
        eval_set=[(X_train, Y_train), (X_val, Y_val)],
        verbose=True,
        early_stopping_rounds=20)

    print("took ", time.time() - ts)

    if is_save:
        timestr = time.strftime("%Y%m%d-%H%M%S")
        pickle.dump(model, open(timestr+"model.pickle.dat", "wb"))
    if is_plot_features:
        f.plot_features(model, (10, 14))
        plt.show()




if __name__ == '__main__':
    version = ''
    model = m_l.xgb_reg_1

    df, features = f.load_train_set_and_features_list(version)
    print('Train Set:')
    print(df.shape)
    print(df.head())
    print('Features')
    print(features)

    X_train, Y_train, X_val, Y_val = create_train_val_split(df, features)
    train_model(model, (X_train, Y_train,X_val, Y_val))