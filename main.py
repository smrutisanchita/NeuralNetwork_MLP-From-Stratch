# This is a sample Python script.
import numpy as np
import PreProcessing
from AircraftANN import AircraftNN
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

# Press the green button in the gutter to run the script.
#if __name__ == '__main__':

if __name__ == '__main__':

    dataset = np.genfromtxt('./DataFiles/dataCollection.csv', delimiter=',')

    pp = PreProcessing.NN_PreProcessing(dataset)
    X_train, X_val, Y_train, Y_val = pp.prepare_data()


    ANN = AircraftNN(X_train.shape[1], Y_train.shape[1])

    # ANN=aircraftNN(inputnumber=items.shape[1],outputnumber=targets.shape[1],HiddenNeuron=5,epoch=100,l=0.7,lr=0.7,mt=0.1)
    ANN.train_model(X_train, Y_train, X_val, Y_val)

    weightHidden, weightOutput = ANN.get_weight_bias()
    print("weightHidden={}\nweightOutput={}".format(weightHidden, weightOutput))
    ANN.save_data_to_file("csv")