from unittest import TestCase
import pandas as pd
import numpy as np
import item_category as it
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)


class Test(TestCase):
    def test_correct_item_category_name(self):
        df = pd.DataFrame(np.array([["Чистые носители (шпиль)", 0]
                                          ]),
                                columns=['item_category_name', 'item_category_id'])

        fixed = it.correct_item_category_name(df)

        print(fixed)

        assert fixed.shape == (1, 2)
        assert fixed.iloc[0].item_category_name == 'Чистые носители - шпиль'


    def test_extract_main_category(self):

        df = pd.DataFrame(np.array([["PC - Гарнитуры/Наушники", 0],
                                          ['Аксессуары - PS2', 1]
                                          ]),
                                columns=['item_category_name', 'item_category_id'])

        fixed = it.extract_main_category(df)

        print(fixed)

        assert fixed.shape == (2, 3)
        assert fixed.iloc[0].item_category_main == 0
        assert fixed.iloc[1].item_category_main == 1



    def test_extract_digital(self):

        df = pd.DataFrame(np.array([["PC - Гарнитуры/Наушники", 0],
                                    ['Аксессуары - Цифра', 1]
                                    ]),
                          columns=['item_category_name', 'item_category_id'])

        fixed = it.extract_whether_digital(df)

        print(fixed)

        assert fixed.shape == (2, 3)
        assert fixed.iloc[0].is_category_digital == 0
        assert fixed.iloc[1].is_category_digital == 1


    def test_extract_ps_related(self):

        df = pd.DataFrame(np.array([["PC - Гарнитуры/Наушники", 0],
                                    ['Аксессуары - ps3', 1]
                                    ]),
                          columns=['item_category_name', 'item_category_id'])

        fixed = it.extract_ps_related(df)

        print(fixed)

        assert fixed.shape == (2, 3)
        assert fixed.iloc[0].is_category_ps_related == 0
        assert fixed.iloc[1].is_category_ps_related == 1


    def test_fix_item_category(self):

        df = pd.DataFrame(np.array([["PC - Гарнитуры/Наушники", 0]
                                    ]),
                          columns=['item_category_name', 'item_category_id'])

        fixed = it.fix_item_category(df)

        print(fixed)

        assert fixed.shape == (1, 4)
        assert all(fixed.columns == ['item_category_id','item_category_main','is_category_digital','is_category_ps_related'])