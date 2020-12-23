from sklearn import preprocessing
import numpy as np


def correct_item_category_name(df):
    'adjust the format of the "item_category_name" column'
    df.loc[df['item_category_name'] == 'Билеты (Цифра)','item_category_name'] = 'Билеты - Цифра'
    df.loc[df['item_category_name'] == 'Доставка товара','item_category_name'] = 'Доставка товара - service'
    df.loc[df['item_category_name'] == 'Карты оплаты (Кино, Музыка, Игры)',
           'item_category_name'] = 'Карты оплаты - Кино, Музыка, Игры'
    df.loc[df['item_category_name'] == 'Служебные','item_category_name'] = 'Служебные - none'
    df.loc[df['item_category_name'] == 'Чистые носители (шпиль)','item_category_name'] = 'Чистые носители - шпиль'
    df.loc[df['item_category_name'] == 'Чистые носители (штучные)','item_category_name'] = 'Чистые носители - штучные'
    df.loc[df['item_category_name'] == 'Элементы питания','item_category_name'] = 'Элементы питания - none'

    return df


def extract_main_category(df):
    df['item_category_main'] = df['item_category_name'].str.split(' - ').str[0]

    le = preprocessing.OrdinalEncoder(dtype=np.int32)
    df['item_category_main'] = le.fit_transform(df[['item_category_main']])

    return df


def extract_whether_digital(df):
    df['is_category_digital'] = 0
    df.loc[df['item_category_name'].str.contains('Цифра'),'is_category_digital'] = 1

    return df


def extract_ps_related(df):
    df['is_category_ps_related'] = 0
    df.loc[df['item_category_name'].str.contains('PS', case=False), 'is_category_ps_related'] = 1

    return df


def fix_item_category(df):
    df = correct_item_category_name(df)
    df = extract_main_category(df)
    df = extract_whether_digital(df)
    df = extract_ps_related(df)
    df.drop(columns = ['item_category_name'], inplace = True)

    return df