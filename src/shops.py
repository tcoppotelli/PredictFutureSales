from sklearn import preprocessing

def extract_shop_type(df):
    'Extracts type of the shop and creates the shop_type_1 and shop_type_2 columns'

    df.loc[df['shop_name'].str.contains('ТЦ'),'shop_type_1'] = 'type_1'
    df.loc[df['shop_name'].str.contains('ТК'),'shop_type_1'] = 'type_2'
    df.loc[df['shop_name'].str.contains('ТРЦ'),'shop_type_1'] = 'type_3'
    df.loc[df['shop_name'].str.contains('ТРК'),'shop_type_1'] = 'type_4'

    df.loc[(df['shop_name'].str.contains('ТЦ')) |
           (df['shop_name'].str.contains('ТК')),'shop_type_2'] = 'type_1'
    df.loc[(df['shop_name'].str.contains('ТРЦ')) |
           (df['shop_name'].str.contains('ТРК')),'shop_type_2'] = 'type_2'

    df.shop_type_1 = df.shop_type_1.fillna('NONE')
    df.shop_type_2 = df.shop_type_2.fillna('NONE')

    le_1 = preprocessing.OrdinalEncoder()
    df['shop_type_1'] = le_1.fit_transform(df[['shop_type_1']])
    le_2 = preprocessing.OrdinalEncoder()
    df['shop_type_2'] = le_2.fit_transform(df[['shop_type_2']])

    return df


def extract_shop_city(df):
    'Extracts shop city name and city type and creates two new columns'

    # City type: 1 if city is Moscow or Sankt Petersburg (they are quite different from the rest of Russia)
    df['shop_city_type'] = 0

    df['shop_city'] = df['shop_name'].str.split(' ').str[0]
    df.drop(columns=['shop_name'], inplace=True)

    df.loc[df['shop_city'].isin(['Москва', 'СПб']), 'shop_city_type'] = 1

    le = preprocessing.OrdinalEncoder()
    df['shop_city'] = le.fit_transform(df[['shop_city']])

    return df


def fix_shops(shops_df):
    """
    This function modifies the shops df inplace.
    It correct's 3 shops that we have found to be 'duplicates'
    and also creates a few more features: extracts the city and encodes it using OrdinalEncoder
    """

    shops_df = shops_df.loc[~shops_df['shop_id'].isin([0, 1, 11, 40, 23])]
    shops_df = extract_shop_type(shops_df)
    shops_df = extract_shop_city(shops_df)

    return shops_df
