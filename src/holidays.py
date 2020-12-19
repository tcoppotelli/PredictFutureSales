import datetime
import pandas as pd

russian_holidays_start = [
    datetime.date(2013, 1, 1),
    datetime.date(2013, 2, 23),
    datetime.date(2013, 3, 8),
    datetime.date(2013, 5, 1),
    datetime.date(2013, 5, 9),
    datetime.date(2013, 6, 12),
    datetime.date(2013, 11, 4),

    datetime.date(2014, 1, 1),
    datetime.date(2014, 2, 23),
    datetime.date(2014, 3, 8),
    datetime.date(2014, 5, 1),
    datetime.date(2014, 5, 9),
    datetime.date(2014, 6, 12),
    datetime.date(2014, 11, 4),

    datetime.date(2015, 1, 1),
    datetime.date(2015, 2, 23),
    datetime.date(2015, 3, 8),
    datetime.date(2015, 5, 1),
    datetime.date(2015, 5, 9),
    datetime.date(2015, 6, 12),
    datetime.date(2015, 11, 4)
]

russian_holidays_end = [
    datetime.date(2013, 1, 8),
    datetime.date(2013, 2, 23),
    datetime.date(2013, 3, 8),
    datetime.date(2013, 5, 1),
    datetime.date(2013, 5, 9),
    datetime.date(2013, 6, 12),
    datetime.date(2013, 11, 4),

    datetime.date(2014, 1, 8),
    datetime.date(2014, 2, 23),
    datetime.date(2014, 3, 8),
    datetime.date(2014, 5, 1),
    datetime.date(2014, 5, 9),
    datetime.date(2014, 6, 12),
    datetime.date(2014, 11, 4),

    datetime.date(2015, 1, 8),
    datetime.date(2015, 2, 23),
    datetime.date(2015, 3, 8),
    datetime.date(2015, 5, 1),
    datetime.date(2015, 5, 9),
    datetime.date(2015, 6, 12),
    datetime.date(2015, 11, 4)
]


def russian_holidays():
    holidays = pd.DataFrame(columns=['Year', 'Month', 'holidays'])

    for start_date, end_date in zip(russian_holidays_start, russian_holidays_end):
        holidays = holidays.append(
            {'Year': start_date.year, 'Month': start_date.month, 'holidays': (end_date - start_date).days + 1},
            ignore_index=True)

    return holidays


def add_holidays(final_df):
    final_df = pd.merge(final_df, russian_holidays(),  how='left', left_on=['Year', 'Month'], right_on=['Year', 'Month'])
    return final_df.drop_duplicates()
