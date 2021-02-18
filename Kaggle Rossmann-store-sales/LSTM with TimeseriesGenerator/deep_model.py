import warnings

warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.metrics import MeanSquaredError, RootMeanSquaredError
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator


class deep_model:

    def __init__(self, past_days, epochs):
        self.past_days = past_days
        self.epochs = epochs

    def store_dataPreprocessing(self, df_store):

        df_store['CompetitionDistance'].fillna(df_store['CompetitionDistance'].median(), inplace=True)
        df_store['CompetitionOpenSinceYear'].replace('1900', np.nan, inplace=True)
        df_store['CompetitionOpenSinceYear'].replace('1961', np.nan, inplace=True)

        df_store['CompetitionOpenSinceMonth'].fillna(00, inplace=True)
        df_store['CompetitionOpenSinceYear'].fillna(2999, inplace=True)

        df_store.fillna({'Promo2SinceWeek': 0, 'Promo2SinceYear': 0, 'PromoInterval': 0}, inplace=True)

        return df_store

    def clean_test_data(self, df_test):

        df_test.sort_values('Date', ascending=True, inplace=True)
        df_test['Sales'] = -1
        df_test['Customers'] = 0

        return df_test

    def train_dataPreprocessing(self, df_train):

        df_train = df_train[df_train['Sales'] != 0]
        df_train = df_train[
            ['Store', 'DayOfWeek', 'Date', 'Open', 'Promo', 'StateHoliday', 'SchoolHoliday', 'Sales', 'Customers']]
        df_train.sort_values('Date', ascending=True, inplace=True)

        return df_train

    def feature_engineering(self, Dataset):

        # Creating Date Parameters
        Dataset['Month'] = Dataset['Date'].dt.month
        Dataset['Year'] = Dataset['Date'].dt.year
        Dataset['Day'] = Dataset['Date'].dt.day
        Dataset['WeekOfYear'] = Dataset['Date'].dt.weekofyear

        # Creating new column CompetitionOpenMonths & PromoOpenMonths
        Dataset['CompetitionOpenMonths'] = 12 * (Dataset['Year'] - Dataset['CompetitionOpenSinceYear']) + Dataset[
            'Month'] - Dataset['CompetitionOpenSinceMonth']
        Dataset['PromoOpenMonths'] = 12 * (Dataset['Year'] - Dataset['Promo2SinceYear']) + (
                    (Dataset['WeekOfYear'] - Dataset['Promo2SinceWeek']) / 4)
        Dataset['CompetitionOpenMonths'][Dataset['CompetitionOpenMonths'] < 0] = 0
        Dataset['PromoOpenMonths'][Dataset['PromoOpenMonths'] < 0] = 0
        Dataset['CompetitionDistance'][Dataset['CompetitionOpenMonths'] < 0] = 0

        # Creating a flag to set to ! if its promo month
        Dataset['MonthName'] = pd.to_datetime(Dataset['Month'], format='%m').dt.month_name().str.slice(stop=3)

        Dataset['PromoMonth'] = 0
        for months in Dataset['PromoInterval'].unique():
            if months != '':
                for month in str(months).split(','):
                    Dataset['PromoMonth'].loc[
                        (Dataset['PromoInterval'] == months) & (Dataset['MonthName'] == month)] = 1

        Dataset["StateHoliday"] = Dataset["StateHoliday"].apply(lambda x: 0 if x == '0' or x == '0' else 1)
        # Dataset["StateHoliday"][Dataset["StateHoliday"] == 0] = '0'
        Dataset = pd.get_dummies(Dataset, columns=["StoreType", "Assortment"], drop_first=False)

        avg_sales_per_customer = pd.DataFrame(Dataset[Dataset['Customers'] > 0].groupby(['Store', 'DayOfWeek']).apply(
            lambda x: x['Sales'].sum() / x['Customers'].sum()))
        avg_sales_per_customer.columns = ['AvgSalesByCustomer']

        avg_number_customer = pd.DataFrame(
            Dataset[Dataset['Customers'] > 0].groupby(['Store', 'DayOfWeek'])['Customers'].mean())
        avg_number_customer.columns = ['AvgCustomerPerStore']

        Dataset = Dataset.merge(avg_sales_per_customer, on=['Store', 'DayOfWeek'], how='left')
        Dataset = Dataset.merge(avg_number_customer, on=['Store', 'DayOfWeek'], how='left')

        Dataset['AvgCustomerPerStore'].fillna(0, inplace=True)
        Dataset['AvgSalesByCustomer'].fillna(0, inplace=True)

        numeric_columns2 = ['Store', 'DayOfWeek', 'Open', 'Promo', 'SchoolHoliday', 'Sales',
                            'CompetitionDistance', 'Promo2', 'Month', 'Year', 'Day',
                            'WeekOfYear', 'CompetitionOpenMonths', 'PromoOpenMonths',
                            'AvgSalesByCustomer', 'AvgCustomerPerStore', 'PromoMonth',
                            'StateHoliday',
                            'StoreType_a', 'StoreType_b', 'StoreType_c',
                            'StoreType_d', 'Assortment_a', 'Assortment_b', 'Assortment_c']
        # 'StateHoliday_a', 'StateHoliday_b', 'StateHoliday_c', 'StateHoliday_0',
        Dataset = Dataset[numeric_columns2]

        return Dataset

    def scale_data(self, dataset):

        train_dataset = dataset[dataset['Sales'] >= 0]
        test_dataset = dataset[dataset['Sales'] == -1]

        test_dataset = pd.concat([train_dataset.iloc[-self.past_days:, :], test_dataset])

        scalerx = MinMaxScaler()
        scalery = MinMaxScaler()

        x_train = scalerx.fit_transform(train_dataset.drop(['Sales'], axis=1))
        y_train = scalery.fit_transform(train_dataset['Sales'].values.reshape(-1, 1))
        x_test = scalerx.transform(test_dataset.drop(['Sales'], axis=1))

        split = int(len(train_dataset) * 0.8)
        x_train_new = x_train[:split]
        y_train_new = y_train[:split]

        x_val = x_train[split:]
        y_val = y_train[split:]

        return x_train_new, y_train_new, x_val, y_val, x_test, scalery

    def genarate_timeseries(self, x_train, y_train, x_val, y_val, x_test):

        train_timeseries = TimeseriesGenerator(x_train, y_train, length=self.past_days, batch_size=512)
        val_timeseries = TimeseriesGenerator(x_val, y_val, length=self.past_days, batch_size=512)
        test_timeseries = TimeseriesGenerator(x_test, np.zeros((x_test.shape[0], 1)), length=self.past_days,
                                              batch_size=512)

        return train_timeseries, val_timeseries, test_timeseries

    def train_model(self, train_timeseries, val_timeseries, features):

        adam = Adam(lr=0.001, decay=1e-3)

        model = Sequential()

        model.add(LSTM(32, activation='relu', input_shape=(self.past_days, features), return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(16, activation='relu', return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(8, activation='relu', return_sequences=False))
        model.add(Dense(1))
        print(model.summary())
        model.compile(loss="mean_squared_error", optimizer=adam, metrics=[MeanSquaredError(), RootMeanSquaredError()])

        reduce_lr_plateau = ReduceLROnPlateau(monitor='val_loss', factor=0.4, patience=3, verbose=1)
        early_stopping = EarlyStopping(monitor='val_loss', patience=4, verbose=1, mode='min', min_delta=0.0001)
        history = model.fit(train_timeseries, validation_data=val_timeseries, epochs=self.epochs,
                            callbacks=[early_stopping, reduce_lr_plateau])

        return model, history

