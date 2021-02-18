from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

class Model:

    def __init__(self):
        self.epochs=4


    def inverse_normalize(self,Dataset, min_max_list):
        """
        Returns the De-Normalized values of the dataframe
        :param min_max_list: previously stored Min/Max values
        :return: De-Normalized Dataset
        """
        i = 0
        for col in Dataset.columns:
            max_value = min_max_list.iloc[i][0]
            min_value = min_max_list.iloc[i][1]
            i += 1
            Dataset[col] = (Dataset[col] * (max_value - min_value)) + min_value

        return Dataset

    def train_Model(self,X_train,y_train):
        """
        Train the Deep Neural Network with Layers of LSTM and Dense
        :param X_train: Input feature array
        :param y_train: Target Array
        :return: Model and History
        """
        model = Sequential()
        #Layes of LSTM along with Batch Normalization and dropouts
        model.add(LSTM(64, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
        model.add(layers.BatchNormalization())
        model.add(Dropout(0.2))
        model.add(LSTM(32, activation='relu', return_sequences=True))
        model.add(layers.BatchNormalization())
        model.add(Dropout(0.2))
        model.add(LSTM(16, activation='relu', return_sequences=False))
        model.add(layers.BatchNormalization())
        model.add(Dense(32, activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(Dropout(0.2))
        model.add(Dense(16, activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(Dropout(0.2))
        model.add(Dense(8, activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(Dense(1))

        #To print the summary of model
        model.summary()

        model.compile(loss="mean_squared_error", optimizer=keras.optimizers.Adam(0.0005),metrics=[keras.metrics.RootMeanSquaredError()])

        # have used two early stopping criteria
        reduce_lr_plateau = ReduceLROnPlateau(monitor='val_loss', factor=0.4, patience=3, verbose=1)
        early_stopping = EarlyStopping(monitor='val_loss', patience=4, verbose=1, mode='min', min_delta=0.0001)

        model.load_weights("Weights_ross_v.02")

        history = model.fit(X_train, y_train, epochs=self.epochs, batch_size=256, validation_split=0.1 ,callbacks=[early_stopping, reduce_lr_plateau])

        return model,history

    # def train_Model_OLD(self, X_train, y_train):
    #
    #     model = Sequential()
    #
    #     model.add(LSTM(64, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
    #     model.add(layers.BatchNormalization())
    #     model.add(Dropout(0.4))
    #     model.add(LSTM(32, activation='relu', return_sequences=True))
    #     model.add(Dropout(0.4))
    #     model.add(LSTM(32, activation='relu', return_sequences=True))
    #     model.add(Dropout(0.4))
    #     model.add(LSTM(16, activation='relu', return_sequences=True))
    #     model.add(layers.BatchNormalization())
    #     model.add(Dropout(0.4))
    #     model.add(LSTM(8, activation='relu', return_sequences=False))
    #     model.add(layers.BatchNormalization())
    #     model.add(Dense(1))
    #
    #     model.summary()
    #
    #     model.compile(loss="mean_squared_error", optimizer=keras.optimizers.Adam(0.0005),
    #                   metrics=[keras.metrics.RootMeanSquaredError()])
    #
    #     # have used two early stopping criteria
    #     reduce_lr_plateau = ReduceLROnPlateau(monitor='val_loss', factor=0.4, patience=3, verbose=1)
    #     early_stopping = EarlyStopping(monitor='val_loss', patience=4, verbose=1, mode='min', min_delta=0.0001)
    #
    #     model.load_weights("Weights_ross_v.01")
    #
    #     history = model.fit(X_train, y_train, epochs=self.epochs, batch_size=256, validation_split=0.1,
    #                         callbacks=[early_stopping, reduce_lr_plateau])
    #
    #     return model, history