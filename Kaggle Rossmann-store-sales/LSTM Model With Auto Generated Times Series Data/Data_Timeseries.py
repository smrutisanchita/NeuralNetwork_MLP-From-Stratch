import pandas as pd


class Data_Timeseries:

    def __init__(self):
        self.ordinal_columns = ['Assortment', 'PromoInterval', 'StoreType', 'Day', 'Month', 'Year']
        self.different_columns=[]


    def set_columns_diff_train(self,df_train, df_test):
        """
        sets the different_columns class variable based on differnce in columns in dataframes
        :param df_train: dataframe 1
        :param df_test: dataframe 2
        :return: Null
        """

        for col in df_train.columns:
            if col not in df_test.columns:
                self.different_columns.append(col)

        #df_train.drop(different_columns, axis=1, inplace=True)

        #return self.different_columns

    def series_to_supervised(self,df, Past_timesteps=1, sampling_rate=1, dropnan=True):
        """
        It Shifts the dataset based on the sampling rate and adds the column to the data
        :param df: Dataset
        :param Past_timesteps: Number of past days to include
        :param sampling_rate: number of days apart to consider
        :param dropnan: if we want to drop NA Values
        :return: transformed Dataset
        """

        cols, names = list(), list()
        for i in range(Past_timesteps, -1, -1):
            cols.append(df.shift(i * sampling_rate))
            names += [('%s(t-%d)' % (col, i)) for col in df.columns]
        Data_aggregated = pd.concat(cols, axis=1)
        Data_aggregated.columns = names
        if dropnan:
            Data_aggregated.dropna(inplace=True)

        return Data_aggregated

    def convert_to_timeseries(self,df, Past_timesteps, sampling_rate, mode):
        """
        This function converts appends the columns of previous Days based on Past_timesteps, sampling_rate
        :param df: Dataframe
        :param Past_timesteps: Number of past days to include
        :param sampling_rate: number of days apart to consider
        :param mode: Test/Train
        :return: dataframe with new added columns
        """

        if mode == "train":
            df_train = df.drop(self.different_columns, axis=1).sort_values(["Date", "Store"], ascending=True).copy()
            df_series = self.series_to_supervised(df_train, Past_timesteps, sampling_rate, dropnan=True)
        elif mode == "test":
            df_test = df.sort_values(["Date", "Store"], ascending=True).copy()
            df_series = self.series_to_supervised(df_test,Past_timesteps, sampling_rate, dropnan=True)

        return df_series

    def timeseries(self,df_train, df_test, Past_timesteps, sampling_rate, mode):
        """
        for Each Store it generates timeseries data
        :param df_train: Train Data
        :param df_test: Test Data
        :param Past_timesteps: Number of past days to include
        :param sampling_rate: number of days apart to consider
        :param mode: Test/Train
        :return: dataframe with new added columns
        """

        i = 0
        timeseries_data = pd.DataFrame()
        if mode == "train":
            for store_id in df_train.Store.unique():
                Dataset_train= df_train[df_train["Store"] == store_id].copy()
                Dataset_train = self.convert_to_timeseries(Dataset_train, Past_timesteps, sampling_rate, mode="train")
                if i == 0:
                    timeseries_data = Dataset_train
                    i = 1
                else:
                    timeseries_data = pd.concat([timeseries_data, Dataset_train])

        elif mode == "test":

            for s in df_test.Store.unique():
                Dataset_test = df_test[df_test["Store"] == s].copy()
                Dataset_train = df_train[df_train["Store"] == s].sort_values("Date", ascending=True).copy()
                Dataset_train["Id"] = 0
                Dataset_test = pd.concat([Dataset_train.drop(self.different_columns, axis=1).iloc[Dataset_train.shape[0] - (Past_timesteps * sampling_rate) - 1:-1, :], Dataset_test])
                Dataset_test = self.convert_to_timeseries(Dataset_test, Past_timesteps, sampling_rate, mode="test")
                if i == 0:
                    timeseries_data = Dataset_test
                    i = 1
                else:
                    timeseries_data = pd.concat([timeseries_data, Dataset_test])

        return timeseries_data

    def drop_columns(self,train_timeseries, test_timeseries,past_timesteps):
        """
        This function drops columns from Test and Train which are not common to both
        :param train_timeseries: Train Data
        :param test_timeseries: Test data
        :param past_timesteps: Number of past days included
        :return: Train and Test data with same columns
        """

        for i in range(past_timesteps, 0, -1):
            train_timeseries.drop(["Sales" + f"(t-{i})", f"Date(t-{i})"], axis=1, inplace=True)
        train_timeseries["Sales"] = train_timeseries["Sales(t-0)"]
        train_timeseries.drop(["Date(t-0)", "Sales(t-0)"], axis=1, inplace=True)

        for i in range(past_timesteps, 0, -1):
            test_timeseries.drop(["Id" + f"(t-{i})", f"Date(t-{i})"], axis=1, inplace=True)
        test_timeseries["Id"] = test_timeseries["Id(t-0)"]
        test_timeseries.drop(["Date(t-0)", "Id(t-0)"], axis=1, inplace=True)

        return train_timeseries,test_timeseries