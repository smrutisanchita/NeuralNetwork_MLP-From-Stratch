import pandas as pd
import numpy as np

try:
    import cPickle as pickle
except ImportError:
    import pickle


class Data_Manipulation:

    def __init__(self):
        self.ordinal_columns = ['Assortment', 'PromoInterval', 'StoreType', 'Day', 'Month', 'Year']

    def normalize(self, df, numeric_columns):
        """
        Normalize the input Data
        :param df: Dataset
        :param numeric_columns: columns to normalize
        :return: Transformed Dataset
        """
        min_max_list = []
        for col in numeric_columns:
            if df[col].dtype != object:
                max_value = np.max(df[col])
                min_value = np.min(df[col])
                df[col] = (df[col] - min_value) / (max_value - min_value)
                min_max_list.append((max_value, min_value))

        return df, min_max_list

    def store_dataPreprocessing(self, df_store):
        """
        PreProcessing of Store Data
        :param df_store: Store Dataset
        :return: transformed dataset
        """

        df_store['CompetitionDistance'].fillna(df_store['CompetitionDistance'].median(), inplace=True)
        df_store['CompetitionOpenSinceYear'].replace('1900', np.nan, inplace=True)
        df_store['CompetitionOpenSinceYear'].replace('1961', np.nan, inplace=True)
        df_store['Promo2SinceYear'] = df_store.Promo2SinceYear.fillna(1800).astype(np.int32)

        df_store.fillna({'Promo2SinceWeek': 1, 'PromoInterval': 0,
                         'CompetitionOpenSinceMonth': 1, 'CompetitionOpenSinceYear': 1800}, inplace=True)

        return df_store

    def feature_engineering(self, Dataset, avg_customer_per_store, mode='train'):
        """
        Feature engineering for the input data based on mode
        :param Dataset: Datset
        :param avg_customer_per_store: value for avg_customer_per_store
        :param mode: Train or Test
        :return: transformed Dataset
        """

        # based on the mode avg_customer_per_store is added to the dataset
        if mode == 'train':
            Dataset = Dataset.merge(avg_customer_per_store, how='left', on="Store", suffixes=("_X", ""))
            Dataset.drop("Customers_X", axis=1, inplace=True)
        else:
            Dataset = Dataset.merge(avg_customer_per_store, how='left', on="Store")

        # The State holiday type doesnt impact much on sales hence conveting it to 0/1 column
        # Dataset.loc[Dataset["StateHoliday"] != '0', "Customers"] = Dataset.loc[Dataset["StateHoliday"] != '0', "Customers"].apply(lambda x: x == 0)
        Dataset["StateHoliday"] = Dataset["StateHoliday"].apply(lambda x: 0 if x == '0' or x == '0' else 1)

        # Filling all Null values of Customers with 0
        Dataset["Customers"] = Dataset["Customers"].apply(lambda x: 0 if x == False else x)

        # forming a date column for Competition open since
        Dataset["CompetitionOpenSince"] = pd.to_datetime(
            dict(year=Dataset.CompetitionOpenSinceYear, month=Dataset.CompetitionOpenSinceMonth, day=15))

        # converting the date to Datetime format
        Dataset["Date"] = pd.to_datetime(Dataset.Date, format='%Y-%m-%d', errors='ignore')

        # adding a column to calculate days since Competition is open
        Dataset["CompetitionOpenDays"] = (Dataset.Date - Dataset.CompetitionOpenSince).dt.days
        Dataset.loc[Dataset["CompetitionOpenDays"] < 0, 'CompetitionOpenDays'] = 0
        Dataset.loc[Dataset["CompetitionOpenDays"] == 1800, 'CompetitionOpenDays'] = 0

        # creating Date column for Promo2 started date
        Dataset["Promo2Since"] = pd.to_datetime(Dataset.Promo2SinceYear.astype(str), format='%Y') + pd.to_timedelta(
            Dataset.Promo2SinceWeek.mul(7).astype(str) + ' days')

        # adding a column to calculate days since promo2 has started
        Dataset["Promo2Days"] = (Dataset.Date - Dataset["Promo2Since"]).dt.days

        Dataset.loc[Dataset.Promo2Days < 0, "Promo2Days"] = 0
        Dataset.loc[Dataset.Promo2SinceYear < 1810, "Promo2Days"] = 0

        # filling all null with 0
        Dataset["PromoInterval"] = Dataset["PromoInterval"].fillna('0')
        Dataset["Open"] = Dataset["Open"].fillna(1)

        # adding Date columns
        Dataset["Day"] = Dataset.Date.dt.day
        Dataset["Month"] = Dataset.Date.dt.month
        Dataset["Year"] = Dataset.Date.dt.year

        Dataset["StateHoliday"] = Dataset["StateHoliday"].apply(lambda x: 0 if x == '0' else 1)

        #one hot encoding of the odinal columns
        Dataset = pd.get_dummies(Dataset, columns=self.ordinal_columns)

        #dropping columns which are not need for the model
        Dataset.drop(
            ["CompetitionOpenSinceYear", "CompetitionOpenSinceMonth", "CompetitionOpenSince", "Promo2SinceYear",
             "Promo2SinceWeek", "Promo2Since"], axis=1, inplace=True)

        return Dataset