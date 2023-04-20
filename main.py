import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import MinMaxScaler, LabelEncoder


# ____________________________________________________________________________________________________
########################################### Preprocessing ###########################################
# ----------------------------------------------------------------------------------------------------
# Scale data using MinMaxScaler to scale features in the range [0, 1] since we don't have negative values, also help gradient decent to converge much faster
def scaling(df, features):
    scaler = MinMaxScaler()
    df[features] = scaler.fit_transform(df[features])
    return df


def split(data):
    Train = []
    Test = []
    for i in range(3):
        temp = data[i * 50:(i + 1) * 50, :]  # temp.shape(50,6)
        random.shuffle(temp)
        Train.append(temp[:30, :])
        Test.append(temp[30:, :])

    Train = np.array(Train).reshape((90, 6))
    Test = np.array(Test).reshape((60, 6))
    x_train = Train[:, 1:]
    y_train = Train[:, 0].reshape((90, 1))
    x_test = Test[:, 1:]
    y_test = Test[:, 0].reshape((60, 1))
    return x_train.T, y_train.T, x_test.T, y_test.T

def y_reshape(y, output_Dimen):
    m = y.shape[1]
    y_reshaped = np.zeros((output_Dimen, m))
    for i in range(m):
        label = int(y[0, i])
        y_reshaped[label, i] = 1
    return y_reshaped

def preprocessing():
    # read cvs file
    df = pd.read_csv('penguins.csv')

    # handling null values
    df = df.fillna(df.mode().iloc[0])

    # scaling data
    features = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
    df = scaling(df, features)

    # encode class label 'species' and 'gender'
    label_encoder = LabelEncoder()
    df['gender'] = label_encoder.fit_transform(df['gender'])
    df['species'] = label_encoder.fit_transform(df['species'])

    # splitting data
    return split(np.array(df))


def digits_preprocessing():
    Train = np.array(pd.read_csv('mnist_train.csv'))
    Test = np.array(pd.read_csv('mnist_test.csv'))
    #splitting
    X_train = Train[:, 1:]
    Y_train = Train[:, 0].reshape((60000, 1))
    X_test = Test[:, 1:]
    Y_test = Test[:, 0].reshape((10000, 1))
    #scaling
    X_train = X_train/255
    X_test = X_test/255
    return X_train.T, Y_train.T, X_test.T, Y_test.T


# ________________________________________________________________________________________________
########################################### Algorithm ###########################################
# ------------------------------------------------------------------------------------------------

def parameters_initialization(layers_dims):
    parameters = {}
    l = len(layers_dims)
    for i in range(1, l):
        parameters['W' + str(i)] = np.random.randn(layers_dims[i], layers_dims[i - 1])
        parameters['b' + str(i)] = np.zeros((layers_dims[i], 1))
    return parameters


def sigmoid(z):  # activation_flag = 0
    a = 1 / (1 + np.exp(-z))
    return a


def tanh(z):  # activation_flag = 1
    a = (1 - np.exp(-z)) / (1 + np.exp(-z))
    return a


def activations_derivative(a, activation_flag):
    if activation_flag:  # tanh
        dZ = (1 - a) * (1 + a)
    else:  # sigmoid
        dZ = a * (1 - a)
    return dZ


def BackPropagation(y_pred, Y, caches, activation_flag, layersN):
    grads = {}
    Y = Y.reshape(y_pred.shape)
    dA = -(Y - y_pred)  # since an ahna mshyen mn wara l odam f hakhod derivative y_hat bnzba l loss function al awl
    for l in range(layersN, 0, -1):
        current_cache = caches[l - 1]
        input, w, a = current_cache
        dZ = activations_derivative(a,activation_flag) # dZ = derivative al activation function used with respect to a (a is the output of the current layer)
        S = dA * dZ  # local gradient (S) = dA * derivative the activation function used(dZ)
        grads["dW" + str(l)] = np.dot(S, input.T)  # dW = local gradient (S) * input to the current layer
        grads["db" + str(l)] = np.sum(S, axis=1, keepdims=True)  # db = local gradient(S), keepdims: keep dimensions and don't convert it to rank 1 array, axis 1 = columns
        dA = np.dot(w.T, S)  # dA = local gradient of the previous layer(S) * weights
    return grads


def predict(X, params, L, activation_flag):
    A_in = X
    for l in range(L):
        Z = np.dot(params['W' + str(l + 1)], A_in) + params['b' + str(l + 1)]
        if activation_flag:
            A = tanh(Z)
        else:
            A = sigmoid(Z)
        A_in = A
    return A


def compute_cost(X, Y, params, L, activation_flag):
    m = X.shape[1]
    A = predict(X, params, L, activation_flag)
    cost = 1 / (2 * m) * np.sum(np.square(Y - A))
    return cost


def weightsUpdate(params, grads, learning_rate, add_bias, layersN):
    for l in range(1, layersN + 1):
        params["W" + str(l)] -= learning_rate * grads["dW" + str(l)]
        if add_bias:
            params["b" + str(l)] -= learning_rate * grads["db" + str(l)]
    return params


def gradient_decent(x_train, y_train, epochs, nlayers, activation_flag, parameters, learning_rate, add_bias):
    m = x_train.shape[1]
    for e in range(epochs):
        # print(f"####################################### EPOCH {e+1} \n")
        for i in range(m):
            # for each training example
            input = x_train[:, i].reshape(x_train.shape[0], 1)
            caches = []
            # 1- forward prop
            for l in range(1, nlayers + 1):
                z = np.dot(parameters['W' + str(l)], input) + parameters['b' + str(l)]
                if activation_flag:
                    a = tanh(z)
                else:
                    a = sigmoid(z)

                cache = (input, parameters['W' + str(l)], a)  # storing the input, output & W of the current layer (needed for back propagation)
                caches.append(cache)  # list of tuples
                input = a  # current output = next input
                # print(f'Layer {l} forward gives activation: \n {a}')
            # 2- back prop
            grads = BackPropagation(a, y_train[:, i], caches, activation_flag, nlayers)
            # print(f'Grads of epoch {e+1} are: \n {grads}')
            # 3- update weights
            parameters = weightsUpdate(parameters, grads, learning_rate, add_bias, nlayers)
            # print(f'Parameters after update in epoch{e+1}: \n {parameters}')
        # Compute total cost at the end of each epoch
        cost = compute_cost(x_train, y_train, parameters, nlayers, activation_flag)
        # if e % 1000 == 0:
        # print(f'Cost After Epoch {e+1}: {cost}')
    return parameters


# ________________________________________________________________________________________________
########################################### Evaluation ##########################################
# ------------------------------------------------------------------------------------------------
def confusion_matrix(X, Y, params, L, activation_flag, nNeurons): #nNeurons= number of neurons in output layer = number of classes
    A = predict(X, params, L, activation_flag)
    m = X.shape[1]
    matrix = np.zeros((nNeurons, nNeurons))
    for i in range(m):
        y = Y[:, i].reshape(nNeurons, 1)
        y_hat = A[:, i].reshape(nNeurons, 1)
        y = y.argmax()  # return index bt3 akbr value fl list
        y_hat = y_hat.argmax()
        matrix[y,y_hat] += 1
    tp = 0
    for i in range(nNeurons):
        tp += matrix[i, i]
    accuracy = (tp/m)*100
    return matrix, accuracy


# ________________________________________________________________________________________________
########################################### MAIN ###########################################
# ------------------------------------------------------------------------------------------------

def penguins_classification(layers_dims):
    x_train, y_train, x_test, y_test = preprocessing()
    y_train = y_reshape(y_train, 3)
    y_test = y_reshape(y_test, 3)
    layers_dims.insert(0, x_train.shape[0])
    layers_dims.append(3)
    return x_train, y_train, x_test, y_test

def digits_classification(layers_dims):
    x_Train, y_Train, x_Test, y_Test = digits_preprocessing()
    y_Train = y_reshape(y_Train, 10)
    y_Test = y_reshape(y_Test, 10)
    layers_dims.insert(0, x_Train.shape[0])
    layers_dims.append(10)
    return x_Train, y_Train, x_Test, y_Test


'''
# take inputs

def main_function(classifier,layers_dims, nEpochs, nLayers, activ, alpha, bias_flag):
    if classifier == 0:
        x_train, y_train, x_test, y_test = penguins_classification(layers_dims)
    else:
        x_train, y_train, x_test, y_test = digits_classification(layers_dims)

    params = parameters_initialization(layers_dims)
    # call gradient descent
    print(f"Cost on Train data before GD: ", compute_cost(x_train, y_train, params, nLayers, activ))
    params = gradient_decent(x_train, y_train, nEpochs, nLayers, activ, params, alpha, bias_flag)

    # calculate accuracy & confusion matrix
    matrix_test, accuracy_test = confusion_matrix(x_test, y_test, params, nLayers, activ, layers_dims[-1])
    matrix_train, accuracy_train = confusion_matrix(x_train, y_train, params, nLayers, activ, layers_dims[-1])
    print(f"Cost on test data after GD", compute_cost(x_test, y_test, params, nLayers, activ))
    print(f'Training Accuracy: {accuracy_train}%')
    print(f'Confusion Matrix on Test data: \n{matrix_test}')
    print(f'Testing Accuracy: {accuracy_test}%')

classifier = int(input("For Handwritten digits classification enter 1, for penguins classification enter 0: "))
nLayers = int(input("Number of hidden layers:"))
nLayers += 1  # for output layer that contain either 3 or 10 neurons
neurons = input("Enter number of neurons in each hidden layer respectively:")
nEpochs = int(input("Enter number of epochs: "))
activation = int(input("For Tanh activation function enter 1, for sigmoid enter 0:"))
alpha = float(input("Enter learning rate: "))
bias_flag = int(input("For bias enter 1 else enter 0: "))
layers_dims = neurons.split(" ")
layers_dims = list(map(int, layers_dims))

main_function(classifier, layers_dims, nEpochs, nLayers, activation, alpha, bias_flag)

'''