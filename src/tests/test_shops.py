from unittest import TestCase
import pandas as pd
import numpy as np
import shops as s

class Test(TestCase):

    def test_duplicated_shops_should_be_removed(self):
        shops_df = pd.DataFrame(np.array([["!Якутск Орджоникидзе, 56 фран", 0],
                                          ['!Якутск ТЦ "Центральный" фран', 1],
                                          ['Адыгея ТЦ "Мега"', 2]]),
                           columns=['shop_name', 'shop_id'])
        shops_df['shop_id'] = shops_df['shop_id'].astype(np.int64)

        fixed = s.fix_shops(shops_df)

        print(fixed)
        assert fixed.shape == (1, 5)
        assert fixed.index.values == 2
        assert fixed.iloc[0].shop_id == 2

    def test_shop_types(self):
        shops_df = pd.DataFrame(np.array([["!Якутск ТЦ фран", 2],
                                          ['!Якутск ТК фран', 3],
                                          ['Адыгея ТРЦ "Мега"', 4],
                                          ['Адыгея ТРК "Мега"', 5],
                                          ['Адыгея "Мега"', 6]
                                          ]),
                           columns=['shop_name', 'shop_id'])
        shops_df['shop_id'] = shops_df['shop_id'].astype(np.int64)

        fixed = s.fix_shops(shops_df)

        print(fixed)
        assert fixed.shape == (5, 5)
        assert fixed.iloc[0].shop_type_1 == 1
        assert fixed.iloc[1].shop_type_1 == 2
        assert fixed.iloc[2].shop_type_1 == 3
        assert fixed.iloc[3].shop_type_1 == 4
        assert fixed.iloc[4].shop_type_1 == 0

        assert fixed.iloc[0].shop_type_2 == 1
        assert fixed.iloc[1].shop_type_2 == 1
        assert fixed.iloc[2].shop_type_2 == 2
        assert fixed.iloc[3].shop_type_2 == 2
        assert fixed.iloc[4].shop_type_2 == 0

    def test_shop_city(self):
        shops_df = pd.DataFrame(np.array([["Москва ТЦ фран", 2],
                                          ['СПб ТК фран', 3],
                                          ['Адыгея ТРЦ "Мега"', 4]
                                          ]),
                           columns=['shop_name', 'shop_id'])
        shops_df['shop_id'] = shops_df['shop_id'].astype(np.int64)

        fixed = s.fix_shops(shops_df)

        print(fixed)
        assert fixed.shape == (3, 5)
        assert fixed.iloc[0].shop_city_type == 1
        assert fixed.iloc[1].shop_city_type == 1
        assert fixed.iloc[2].shop_city_type == 0

        assert fixed.iloc[0].shop_city == 1
        assert fixed.iloc[1].shop_city == 2
        assert fixed.iloc[2].shop_city == 0
