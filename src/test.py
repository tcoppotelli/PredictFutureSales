import pandas as pd
import functions as f
import shap
import functions_test as t
import pickle
import time
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# Remember to change model and dataframe before running
original_df_location = '20201220-211939df.pickle.dat'
original_model = '20201220-212303model.pickle.dat'
explain = False
submit = True

infile = open(original_df_location, "rb")
df = pickle.load(infile)
infile.close()

test = t.create_test_df(df)
print('df base size ', test.shape)

test = t.add_when_first_sold_to_test(test)

test = f.add_lag(all_data_df = df, df_to_add_lag= test, number_of_months=3)

test = f.add_days_stat(test)

test_with_price = t.add_price_col_to_test(test)

# moved that from "add_price_col_to_test" function to be able to add new "lag" columns
colnames = ['date_block_num', 'shop_id', 'item_id', 'Year', 'Month', 'shop_type_1',
                'shop_type_2', 'shop_city_type', 'shop_city', 'item_category_id',
                'item_category_main', 'is_category_digital', 'is_category_ps_related','item_price_avg','when_first_sold',
            'number_of_mondays','number_of_saturdays','number_of_sundays','number_of_days_in_month']
lag_colnames = [x for x in test.columns if 'lag' in x]
test_with_price = test_with_price[colnames + lag_colnames]

test_with_price = f.downcast_dtypes(test_with_price)

infile = open(original_model, "rb")
model = pickle.load(infile)
infile.close()

print('Validations')
print('Test columns')
print(test_with_price.columns)
print('Number of NANs in columns')
print(test_with_price.isnull().sum())

if explain:
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
    features = list(test_with_price.columns)

    for feature in features:
        shap.dependence_plot(feature, shap_values, test_with_price, interaction_index=None)
    # shap.initjs()
    # idx = 2
    # shap.force_plot(explainer.expected_value, shap_values[idx,:], test_with_price.iloc[idx,:])

if submit:
    y_pred = model.predict(test_with_price)
    test_with_price['item_cnt_month'] = y_pred.clip(0, 20)

    submission = test_with_price.copy()

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

