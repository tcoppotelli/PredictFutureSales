from unittest import TestCase
import pandas as pd
import numpy as np
import sales as sa
from unittest.mock import patch
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

class Test(TestCase):
    def test_prepare_sales_monthly_should_rename_duplicated_shops(self):
        df = pd.DataFrame(np.array([["02.01.2013",0,0,22154,999.0,1.0]
                                    ]),
                          columns=['date','date_block_num','shop_id','item_id','item_price','item_cnt_day'])
        df['date_block_num'] = df['date_block_num'].astype(np.int64)
        df['shop_id'] = df['shop_id'].astype(np.int64)
        df['item_id'] = df['item_id'].astype(np.int64)
        df['item_price'] = df['item_price'].astype(np.float64)
        df['item_cnt_day'] = df['item_cnt_day'].astype(np.float64)

        fixed = sa.adjust_duplicated_shops(df)

        print(fixed)
        assert fixed.shape == (1, 6)
        assert fixed.iloc[0].shop_id == 57

    def test_outliers_should_be_removed(self):
        df = pd.DataFrame(np.array([["02.01.2013",0,0,22154,-999.0,1.0],
                                    ["02.01.2013",0,0,22154,999.0,-1.0],
                                    ["03.01.2013",0,1,22154,20.0,1.0],
                                   ["03.01.2013",0,1,22154,2000.0,20.0]
                                    ]),
                          columns=['date','date_block_num','shop_id','item_id','item_price','item_cnt_day'])
        df['date_block_num'] = df['date_block_num'].astype(np.int64)
        df['shop_id'] = df['shop_id'].astype(np.int64)
        df['item_id'] = df['item_id'].astype(np.int64)
        df['item_price'] = df['item_price'].astype(np.float64)
        df['item_cnt_day'] = df['item_cnt_day'].astype(np.float64)

        fixed = sa.remove_outliers(df)

        print(fixed)
        assert fixed.shape == (1, 6)
        assert fixed.index.values == 2


    def test_prepare_sales_monthly_should_compute_average(self):
        df = pd.DataFrame(np.array([["01.01.2013",0,0,22154,999.0,1.0],
                                    ["02.01.2013",0,0,22154,999.0,1.0],
                                    ["03.01.2013",0,1,22154,997.0,1.0],
                                    ["04.01.2013",1,0,22154,2.0,1.0],
                                    ["02.01.2013",1,0,22154,998.0,1.0]
                                    ]),
                          columns=['date','date_block_num','shop_id','item_id','item_price','item_cnt_day'])
        df['date_block_num'] = df['date_block_num'].astype(np.int64)
        df['shop_id'] = df['shop_id'].astype(np.int64)
        df['item_id'] = df['item_id'].astype(np.int64)
        df['item_price'] = df['item_price'].astype(np.float64)
        df['item_cnt_day'] = df['item_cnt_day'].astype(np.float64)

        fixed = sa.prepare_sales_monthly(df)

        print(fixed)
        assert fixed.shape == (3, 6)
        assert fixed.iloc[0].item_price_avg == 999.0
        assert fixed.iloc[0].date_min == "01.01.2013"
        assert fixed.iloc[0].item_cnt_month == 2
        assert fixed.iloc[0].date_block_num == 0

        assert fixed.iloc[1].item_price_avg == 997.0
        assert fixed.iloc[1].date_min == "03.01.2013"
        assert fixed.iloc[1].item_cnt_month == 1
        assert fixed.iloc[1].date_block_num == 0

        assert fixed.iloc[2].item_price_avg == 500.0
        assert fixed.iloc[2].date_min == "02.01.2013"
        assert fixed.iloc[2].item_cnt_month == 2
        assert fixed.iloc[2].date_block_num == 1


    def test_matrix_should_add_missing_items_per_shop(self):
        df = pd.DataFrame(np.array([["01.01.2013",0,0,1,999.0,1.0],
                                    ["02.01.2013",0,0,2,999.0,1.0],
                                    ["03.01.2013",0,1,3,997.0,1.0],
                                    ["04.01.2013",1,0,4,2.0,1.0],
                                    ["02.01.2013",1,0,5,998.0,1.0]
                                    ]),
                          columns=['date','date_block_num','shop_id','item_id','item_price','item_cnt_day'])
        df['date_block_num'] = df['date_block_num'].astype(np.int64)
        df['shop_id'] = df['shop_id'].astype(np.int64)
        df['item_id'] = df['item_id'].astype(np.int64)
        df['item_price'] = df['item_price'].astype(np.float64)
        df['item_cnt_day'] = df['item_cnt_day'].astype(np.float64)

        fixed = sa.create_matrix(df)

        print(fixed)
        assert fixed.shape == (8, 5)
        assert fixed[fixed.date_block_num == 0].shape == (6, 5)


    @patch('sales.pd.read_csv')
    def test_prepare_sales(self, mock_read_csv):
        df = pd.DataFrame(np.array([["01.01.2013",0,0,1,999.0,1.0],
                                    ["02.01.2013",0,0,2,999.0,1.0],
                                    ["03.01.2013",0,1,3,997.0,1.0],
                                    ["04.01.2013",1,0,4,2.0,1.0],
                                    ["02.01.2013",1,0,5,998.0,1.0],
                                    ["03.01.2013",1,1,22154,2000.0,20.0]#this is to remove outliers
                                    ]),
                          columns=['date','date_block_num','shop_id','item_id','item_price','item_cnt_day'])
        df['date_block_num'] = df['date_block_num'].astype(np.int64)
        df['shop_id'] = df['shop_id'].astype(np.int64)
        df['item_id'] = df['item_id'].astype(np.int64)
        df['item_price'] = df['item_price'].astype(np.float64)
        df['item_cnt_day'] = df['item_cnt_day'].astype(np.float64)
        mock_read_csv.return_value = df

        fixed = sa.prepare_sales(False)

        print(fixed)
        assert fixed.shape == (8, 8)
        assert fixed[fixed.date_block_num == 0].shape == (6, 5)