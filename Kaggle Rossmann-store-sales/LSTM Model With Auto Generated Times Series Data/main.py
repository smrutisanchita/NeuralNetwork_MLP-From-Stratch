import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from Data_Manipulation import Data_Manipulation
from Data_Timeseries import Data_Timeseries
from Model_LSTM import Model
from sklearn.model_selection import train_test_split

def Data_preprocessing():

    DM = Data_Manipulation()

    train = pd.read_csv('./Data_Files/train.csv')
    store = pd.read_csv('./Data_Files/store.csv')
    test = pd.read_csv('./Data_Files/test.csv')


    #Preprocess the test data
    store = DM.store_dataPreprocessing(store)

    #Merging Train/Tets data with Store Data
    train_store_merged = train.merge(store, how='left', on='Store')
    test_store_merge = test.merge(store, how='left', on='Store')

    avg_customer_per_store = train_store_merged.groupby("Store")["Customers"].mean()

    #Add or remove columns from train and test dataset
    df_train = DM.feature_engineering(train_store_merged, avg_customer_per_store, "train")
    df_test = DM.feature_engineering(test_store_merge, avg_customer_per_store, "test")

    #List of numeric columns for Train/test
    numeric_columns_train = ['Sales', 'CompetitionDistance', 'CompetitionOpenDays', 'Promo2Days', "Customers"]
    numeric_columns_test = ['CompetitionDistance', 'CompetitionOpenDays', 'Promo2Days', "Customers"]

    #Normalizing the Train & Test Data
    df_train, train_min_max_list = DM.normalize(df_train, numeric_columns_train)
    df_test, test_min_max_list = DM.normalize(df_test, numeric_columns_test)
    train_min_max_list_pd = pd.DataFrame(train_min_max_list)
    train_min_max_list_pd.to_csv("./Data_Files/train_min_max_list.csv", index=False)
    return df_train,df_test,train_min_max_list,train_min_max_list_pd

def  Data_timeseries(df_train, df_test):

    DT = Data_Timeseries()
    #To find the columns in train which are not there in test
    DT.set_columns_diff_train(df_train, df_test)

    #Hyper Parameter for our model
    past_timesteps = 4  # Number of Days in Past to consider
    sampling_rate = 1   # number of days apart to consider 1 = continuous days

    #convert Train to timeseries data
    train_timeseries = DT.timeseries(df_train, df_test.drop("Id", axis=1), past_timesteps, sampling_rate, mode="train")
    #Convert test to timeseries Data
    test_timeseries = DT.timeseries(df_train.drop("sales", axis=1), df_test, past_timesteps, sampling_rate, mode="test")
    #dropping any columns which are not common
    train_timeseries, test_timeseries = DT.drop_columns(train_timeseries, test_timeseries, past_timesteps)

    train_timeseries.to_csv(f"./Data_Files/train_timeseries.csv", index=False)
    test_timeseries.to_csv(f"/Data_Files/test_timeseries.csv", index=False)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':


    # Uncomment below 2 lines of code to pre process and generate timeseries data ,
    # as we have already processed and stored the file, we are using it

    df_train,df_test,train_min_max_list,test_min_max_list = Data_preprocessing()
    Data_timeseries(df_train, df_test)

    #need to comment below 3 lines of code if we want to Preprocess and generate timeseries data again
    #train_timeseries = pd.read_csv("./Data_Files/train_timeseries.csv")
    #test_timeseries = pd.read_csv("./Data_Files/test_timeseries.csv")
    #train_min_max_list = pd.read_csv("./Data_Files/train_min_max_list.csv")

    target = train_timeseries["Sales"]
    #dropping sales column from training set
    train_timeseries.drop("Sales", axis = 1, inplace  = True)

    #creating dataframe to store predicted data based on ID
    test_Id = test_timeseries["Id"]
    test_timeseries.drop("Id", axis=1, inplace=True)

    #converting the data to 3D arry to feed to LSTM model  5 as number of days considered=4 + 1(currentday)
    train_timeseries_final = np.array(train_timeseries).reshape(train_timeseries.shape[0], 5, -1)
    test_timeseries_final = np.array(test_timeseries).reshape(test_timeseries.shape[0], 5, -1)


    Model=Model()
    #Training the Model
    model_LSTM, history = Model.train_Model(train_timeseries_final,target)

    model_LSTM.save('Assignment2_Model_temp')
    model_LSTM.save_weights("Weights_ross_v.02_temp")

    #forecasting sales in test data
    df_forecast = pd.DataFrame(model_LSTM.predict(test_timeseries_final), columns=["Sales"], index=test_Id)
    df_test_forecast = Model.inverse_normalize(df_forecast, train_min_max_list)
    df_test_forecast['Sales'] = df_test_forecast['Sales'].apply(lambda x: 0 if x < 200 else x)
    df_test_forecast.sort_index()
    df_test_forecast.to_csv('submit.csv')

