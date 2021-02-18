# This is a sample Python script.
from Dense_Model import Dense_Model

import pandas as pd


def read_data_files():

    df_train = pd.read_csv('train.csv', parse_dates=['Date'],low_memory=False)
    df_test = pd.read_csv('test.csv', parse_dates=['Date'],low_memory=False)
    df_store = pd.read_csv('store.csv')

    return df_train, df_test, df_store

if __name__ == '__main__':

    # Reading data files
    df_train, df_test, df_store = read_data_files()

    # creating object of Deep Model class
    DM = Dense_Model(past_days=30, epochs=1)

    # Preprocessing the Store data
    df_store = DM.store_dataPreprocessing(df_store)

    # Preprocessing the Train data
    df_train = DM.train_dataPreprocessing(df_train)

    # adding extra columns from the Test data as per train set
    df_test = DM.clean_test_data(df_test)

    # Creating dataframe to store the forecast based on Ids
    df_test_forecast = pd.DataFrame()
    df_test_forecast['Id'] = df_test['Id']

    # dropping Id column from test data set
    df_test.drop('Id', axis=1, inplace=True)

    # we merge the test and train in order the perform feature engineering together
    train_test_merged = pd.concat([df_train, df_test])

    # Merging the store and train/test data
    final_dataset = pd.merge(train_test_merged, df_store, on='Store', how='inner')

    # feature engineering
    final_dataset = DM.feature_engineering(final_dataset)

    x_train, y_train, x_val, y_val, x_test, scalery = DM.scale_data(final_dataset)


    model, history = DM.train_model(x_train, y_train,(x_val, y_val))

    model.save('Assignment2_Model')

    forecast = model.predict(x_test)

    # adjusting the sales - denormalize and fillna if any
    forecast_denorm = scalery.inverse_transform(forecast)
    df_test_forecast['Sales'] = forecast_denorm.reshape(-1, )
    df_test_forecast['Sales'].fillna(0, inplace=True)
    df_test_forecast.sort_values('Id', ascending=True, inplace=True)
    df_test_forecast.to_csv('submit.csv', index=False)