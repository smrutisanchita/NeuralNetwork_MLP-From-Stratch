import numpy as np
class AircraftNN:
    """
    A multilayer Neural Network with one hidden layer
    """

    def __init__(self, X_number, Y_number, hidden_neuron=8, epochs=10, lambda_val=0.6, learning_rate=0.3,
                 momentum_rate=0.2):
        """
        :param X_number: # of inputs - 2
        :param Y_number: # of outputs - 2
        :param hidden_neuron: # of hidden neuron - 6
        :param epochs: # of epochs - 6
        :param lambda_val: lambda value to be used in sigmoid activation function - 6
        :param learning_rate: learning rate of the model
        :param momentum_rate: momentum rate (for previous gradients)
        """

        # define hyper parameters
        self.vlambda = lambda_val  # 0.6
        self.learningRate = learning_rate  # 0.4
        self.hiddenLayerNeuron = hidden_neuron  # 0.6
        self.mrate = momentum_rate  # 0.2

        # define other given parameters
        self.inputLayerFeatures = X_number
        self.outputLayerNeuron = Y_number
        self.epochs = epochs
        self.train_rmse_error = []
        self.val_rmse_error = []

        # have used seed to keep the random weights initialization contact to test efficiency of hyperparameters
        np.random.seed(1)

        # random initialize the weights
        # size of weights to hidden layer = (hidden neuron * (inputs + 1 (for bias))
        # size of weights to output layer = (hidden neuron * (output + 1 (for bias))

        self.weightHidden = np.random.uniform(-1.22, 1.22, size=(self.hiddenLayerNeuron, self.inputLayerFeatures + 1))
        self.weightOutput = np.random.uniform(-1.22, 1.22, size=(self.outputLayerNeuron, self.hiddenLayerNeuron + 1))

        #         self.weightHidden = np.genfromtxt('.DataFiles/weightHidden.csv', delimiter=',')
        #         self.weightOutput = np.genfromtxt('.DataFiles/weightOutput.csv', delimiter=',')

        # intializing the gradients to 0 for the first epoch
        self.mGradWeightHidden = np.zeros((self.hiddenLayerNeuron, self.inputLayerFeatures + 1))
        self.mGradWeightOutput = np.zeros((self.outputLayerNeuron, self.hiddenLayerNeuron + 1))

        self.outputHidden = None

    def forward_propagation(self, x):
        """
        Forward Propagation function to calculate predicted output
        """
        # added 1 to input & output of hidden for bias
        input_hidden = self.matrix_dot_product(np.append(x, 1), self.weightHidden.T)
        self.outputHidden = self.sigmoid(input_hidden)
        input_output_layer = self.matrix_dot_product(np.append(self.outputHidden, 1), self.weightOutput.T)
        predicted_output = self.sigmoid(input_output_layer)

        return predicted_output

    def backward_propagation(self, error, x, output_predicted):
        """
            Backward Propagation using gradient descent to update the weights
        """
        gradient_output = self.vlambda * error * self.derivative_sigmoid(output_predicted)
        d_weight_output = (self.learningRate * self.matrix_dot_product(
            gradient_output.reshape(self.outputLayerNeuron, -1), np.append(self.outputHidden, 1).reshape(-1,
                                                                                                         self.hiddenLayerNeuron + 1))) + self.mrate * self.mGradWeightOutput
        gradient_hidden = self.vlambda * self.derivative_sigmoid(
            np.append(self.outputHidden, 1)) * self.matrix_dot_product(gradient_output, self.weightOutput)
        d_weight_hidden = self.learningRate * self.matrix_dot_product(
            gradient_hidden.reshape(self.hiddenLayerNeuron + 1, -1),
            np.append(x, 1).reshape(-1, self.inputLayerFeatures + 1))[:-1, :] + self.mrate * self.mGradWeightHidden

        self.weightOutput += d_weight_output
        self.weightHidden += d_weight_hidden

        self.mGradWeightOutput = d_weight_output
        self.mGradWeightHidden = d_weight_hidden

    def train_model(self, x_train, y_train, x_val, y_val):
        """
           Trains model running forward & backward propagation
        """

        for ep in range(0, self.epochs):
            train_mse_sum = 0
            val_mse_sum = 0
            val_error_inc_count = 0
            train_error_plateau_count = 0

            for i, (input, output) in enumerate(zip(x_train, y_train)):
                expected_output = output
                pred_output = self.forward_propagation(input)
                error = expected_output - pred_output
                self.backward_propagation(error, input, pred_output)

                train_mse = np.mean(error ** 2)
                train_mse_sum += train_mse

            for i, (input, output) in enumerate(zip(x_val, y_val)):
                val_output = self.forward_propagation(input)
                val_error = output - val_output
                val_mse = np.mean(val_error ** 2)
                val_mse_sum += val_mse

            train_mse_error = train_mse_sum / x_train.shape[0]
            self.train_rmse_error.append(np.sqrt(train_mse_error))

            val_mse_error = (val_mse_sum / x_val.shape[0])
            self.val_rmse_error.append(np.sqrt(val_mse_error))
            print("Error at epoch {} is {},Val={}".format(ep + 1, np.sqrt(train_mse_error), np.sqrt(val_mse_error)))

            if ep > 1:
                if (self.val_rmse_error[ep] - self.val_rmse_error[ep - 1]) > 0:
                    val_error_inc_count += 1
                if (self.train_rmse_error[ep - 1] - self.train_rmse_error[ep]) < 0.00000005:
                    train_error_plateau_count += 1

            if val_error_inc_count > 5 or train_error_plateau_count > 10:
                return self.weightHidden, self.weightOutput

        return self.weightHidden, self.weightOutput

    def rmse_error(total_error, size):
        pass

    def predict_output(self, x_val, y_val):
        """
        Predict Output for given inputs and calculate rmse
        :param x_val:
        :param y_val:
        :return:
        """
        predict_mse_sum = 0
        for i, (input, output) in enumerate(zip(x_val, y_val)):
            predicted_output = self.forward_propagation(input)
            print("input={}\nOutput={}\npredicted_output={}".format(input, output, predicted_output))
            predict_error = output - predicted_output
            predict_mse = np.mean(predict_error ** 2)
            predict_mse_sum += predict_mse

        predict_mse_error_mean = predict_mse_sum / x_val.shape[0]
        return np.sqrt(predict_mse_error_mean)

    def sigmoid(self, x):
        """
            This function returns sigmoid value of input
        """
        return 1 / (1 + np.exp(-x * self.vlambda))

    def derivative_sigmoid(self, x):
        """
            This function returns sigmoid derivative value of input
        """
        return x * (1 - x)

    def matrix_dot_product(self, X, Y):

        if len(X.shape) < 2:  # as my input is 1-D array , x.shape gives number of elements in it
            product = np.zeros((1, Y.shape[1]))
            if (X.shape[0] != Y.shape[0]):
                raise Exception("Matrix Dimention doesn't Match")
            i = 0
            for j in range(Y.shape[1]):
                product[i, j] = np.sum(X * Y[:, j].T)
            return product

        else:
            product = np.zeros((X.shape[0], Y.shape[1]))
            if (X.shape[1] != Y.shape[0]):
                raise Exception("Matrix Dimention doesn't Match")

            for i in range(X.shape[0]):

                for j in range(Y.shape[1]):
                    product[i, j] = np.sum(X[i] * Y[:, j].T)
        return product

    def get_weight_bias(self):
        """
            Getter function for weights
        """
        return self.weightHidden, self.weightOutput

    def save_data_to_file(self, file_format):
        """
        Save file in csv format
        """
        if file_format == "csv":
            np.savetxt(".DataFiles/weightHidden.csv", self.weightHidden, delimiter=",")
            np.savetxt(".DataFiles/weightOutput.csv", self.weightOutput, delimiter=",")

        elif file_format == "npy":
            np.save("weightHidden.npy", self.weightHidden)
            np.save("weightOutput.npy", self.weightOutput)