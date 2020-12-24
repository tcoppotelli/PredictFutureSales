import pandas as pd
import functions as f
import models_library as m_l
import shap
import functions_test as t
import pickle
import time
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)


def create_test_df(train, features):
    """
    Function that creates test set
    :param train: (pandas df) train set from pickle file
    :param features: (list) - list of feature names
    :return: test_with_price (pandas df) - test set
    """
    test = t.create_test_df(train)
    print('df base size ', test.shape)

    test = t.add_when_first_sold_to_test(test)

    test = f.add_lag(all_data_df = train, df_to_add_lag= test, number_of_months=3)

    test = f.add_days_stat(test)

    test_with_price = t.add_price_col_to_test(test)

    test_with_price = test_with_price[features]

    test_with_price = f.downcast_dtypes(test_with_price)

    return test_with_price


def train_model(model, train, features, is_save = False, is_plot_features = True):
    """
    Function that trains a model with whole train set without validation split
    :arg model (xgboost model) model object from models_library
         train (pandas df) - train set from pickle file
         features (list) - list of feature names
         is_save (bool) - whether to save the model or not
         is_plot_features (bool) - whether to plot the feature importance
    :return model (xgboost model) trained model
    """
    X_train = train[features]
    Y_train = train['item_cnt_month']

    ts = time.time()

    model.fit(
        X_train,
        Y_train,
        eval_metric="rmse",
        eval_set=[(X_train, Y_train)],
        verbose=True,
        early_stopping_rounds=20)

    print("took ", time.time() - ts)

    if is_save:
        timestr = time.strftime("%Y%m%d-%H%M%S")
        pickle.dump(model, open(timestr + "model.pickle.dat", "wb"))
    if is_plot_features:
        f.plot_features(model, (10, 14))
        plt.show()

    return model



def explain(test, model):
    """PLEASE ADD DOCSTRING"""
    explainer = shap.TreeExplainer(model)
    # shap_values = explainer.shap_values(test_with_price)
    infile = open('test.shap.pickle.dat', "rb")
    shap_values = pickle.load(infile)
    infile.close()
    # features = ['date_block_num', 'shop_id', 'item_id', 'Year', 'Month', 'shop_type_1',
    #             'shop_type_2', 'shop_city_type', 'shop_city', 'item_category_id',
    #             'item_category_main', 'is_category_digital', 'is_category_ps_related',
    #             'item_cnt_month_1', 'item_cnt_month_2', 'item_cnt_month_3',
    #             'item_price_avg']
    features = list(test.columns)

    for feature in features:
        shap.dependence_plot(feature, shap_values, test, interaction_index=None)
    # shap.initjs()
    # idx = 2
    # shap.force_plot(explainer.expected_value, shap_values[idx,:], test_with_price.iloc[idx,:])


def create_submission_file(test, model):
    """
    Function that creates and saves a submission-ready file
    :param test: (pandas df) test set
    :param model: (xgboost model) trained model
    :return: saved submission file
    """
    y_pred = model.predict(test)
    test['item_cnt_month'] = y_pred.clip(0, 20)

    submission = test.copy()

    submission = t.apply_0_to_not_sold_categories(submission)
    submission = submission[['item_cnt_month']].reset_index()
    submission.rename(columns={"index": "ID"}, inplace=True)
    print('apply_0_to_not_sold_categories ', submission.shape)

    # This did not change much
    submission = t.correct_submission_for_not_sold_items(submission)

    print('Expect (214200, 2)')
    print('Actual ', submission.shape)
    timestr = time.strftime("%Y%m%d-%H%M%S")
    submission.to_csv(timestr+'solution_xgboost.csv', index=False)
    print("Your submission was successfully saved!")


if __name__ == '__main__':
    version = ''
    is_explain = False
    is_submit = True
    model = m_l.xgb_reg_1

    train, features = f.load_train_set_and_features_list(version)
    test = create_test_df(train, features)
    print('Test')
    print(test.shape)
    print(test.head())

    model = train_model(model, train, features, is_save = False, is_plot_features = True)
    if is_explain:
        explain(test, model)
    if is_submit:
        create_submission_file(test, model)