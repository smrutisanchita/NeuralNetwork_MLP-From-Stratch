from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np

class NN_PreProcessing:

    def __init__(self, X):

        self.dataset = X
        self.norm_dataset = None
        data_min, data_max = None, None

    def prepare_data(self):
        #1. Clean the Data
        self.data_cleaning()

        #2. Normalizing the data
        scaler = MinMaxScaler()
        self.norm_dataset = scaler.fit_transform(self.dataset)

        #3. Deviding input X and output Y
        X = self.norm_dataset[:, :2]
        Y = self.norm_dataset[:, 2:]

        #4. Test and Validation set split
        X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=30)

        #saving min max values to be usined in game to normalize inputs in game
        self.data_min, self.data_max = scaler.data_min_, scaler.data_max_

        self.save_data_to_file(self.data_min, 'Traning_min_values')
        self.save_data_to_file(self.data_max, 'Traning_max_values')

        return X_train, X_val, Y_train, Y_val

    def data_cleaning(self):
        pass
        # self.dataset = np.unique(self.dataset, axis=0)              # remove any duplicate rows
        # self.dataset = self.dataset[self.dataset[:, 2] != 0]        # removing any rows which has 0 as new x velocity
        # self.dataset = self.dataset[self.dataset[:, 3] != 0]        # removing any rows which has 0 as new y velocity

    def save_data_to_file(self, X, filename):

        np.savetxt("{}.csv".format(filename), X, delimiter=",")

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler
    import numpy as np

    class NN_PreProcessing:

        def __init__(self, X):
            self.dataset = X
            self.norm_dataset = None
            data_min, data_max = None, None

        def prepare_data(self):
            # 1. Clean the Data
            self.data_cleaning()

            # 2. Deviding input X and output Y
            X = self.dataset[:, :2]
            Y = self.dataset[:, 2:]

            # 3. Test and Validation set split
            X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=30)

            # 4. Normalizing the data
            scalerx = MinMaxScaler()
            X_train = scalerx.fit_transform(X_train)
            X_val = scalerx.transform(X_val)

            scalery = MinMaxScaler()
            Y_train = scalery.fit_transform(Y_train)
            Y_val = scalery.transform(Y_val)

            self.data_min = np.concatenate([scalerx.data_min_, scalery.data_min_])
            self.data_max = np.concatenate([scalerx.data_max_, scalery.data_max_])

            self.save_data_to_file(self.data_min, 'Traning_min_values')
            self.save_data_to_file(self.data_max, 'Traning_max_values')

            return X_train, X_val, Y_train, Y_val

        def data_cleaning(self):
            """
            Function to clean the data (custom defined)
            """
            # all duplicate values for output is removed , even if input is different
            output_slice = self.dataset[:, [2, 3]]
            _, unq_row_indices = np.unique(output_slice, return_index=True, axis=0)
            self.dataset = self.dataset[unq_row_indices]

            self.dataset = self.dataset[self.dataset[:, 2] != 0]  # removing any rows which has 0 as new x velocity
            self.dataset = self.dataset[self.dataset[:, 3] != 0]  # removing any rows which has 0 as new y velocity

        def save_data_to_file(self, X, filename):
            np.savetxt("{}.csv".format(filename), X, delimiter=",")