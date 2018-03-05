
import numpy as np
import matplotlib.pyplot as plt


class activate_tanh(object):
    """Tanh activation function"""
    def activate(self, z):
        return np.tanh(z)

    def activate_prime(self, z):
        return 1 - np.tanh(z) ** 2


class NeuralNetwork(object):
    
    def __init__(self, layers, activationFunction, weights=None, bias=None):
        
        self.layers = layers
        self.no_input_neurons = layers[0]
        self.no_output_neurons = layers[-1]
        self.no_hidden_layers = len(layers) - 2
        self.act_func = activationFunction
        self.weight_matrices = weights
        self.bias = bias
        self.loss = []

    def generate_Weight_Matrices(self, layers_list):
        
        layers = np.array(layers_list)
        matrix_idx = np.array([layers[:-1], layers[1:]])
        weights = [np.random.randn(idx[1], idx[0]) for idx in matrix_idx.T]
        return weights

    def train(self, X, Y, rate, epochs):
        
        m = Y.shape[1]
        
        for i in range(epochs):
            
            # do backprop
            dw, db, loss = self.back_pass(X, Y)

            # update matrices
            for i, _ in enumerate(self.weight_matrices):
                self.weight_matrices[i] = self.weight_matrices[i] - \
                    (rate / m) * dw[i]

            # update bias
            for i, _ in enumerate(self.bias):
                self.bias[i] = self.bias[i] - (rate / m) * db[i]

            # update loss
            self.loss.append(loss)

        self.print_loss()
        return self.weight_matrices, self.bias

    def predict(self, a):
        
        for i, b in enumerate(self.bias):
            z = np.dot(self.weight_matrices[i], a) + b
            a = self.act_func.activate(z)
        
        print('prediction is', a)

    def forward_pass(self, a):

        if self.weight_matrices is None:
            self.weight_matrices = self.generate_Weight_Matrices(self.layers)
        if self.bias is None:
            self.bias = np.random.randn(self.no_hidden_layers + 1)

        weighted_inputs = []
        activated_outputs = []
        activated_outputs.append(a)

        for i, b in enumerate(self.bias):
            z = np.dot(self.weight_matrices[i], a) + b
            a = self.act_func.activate(z)
            weighted_inputs.append(z)
            activated_outputs.append(a)
        
        return weighted_inputs, activated_outputs
    
    def back_pass(self, X, Y):
        
        dw = []
        db = []
        
        # do forward pass
        weighted_inputs, activated_outputs = self.forward_pass(X)
       
        # calculate output error
        dC_da = (activated_outputs[-1] - Y)
        loss = (0.5 / Y.shape[1]) * np.sum(dC_da**2)
        delta = dC_da * self.act_func.activate_prime(weighted_inputs[-1])

        # calculate weights and bias perturbations
        for i, _ in enumerate(self.bias):
            
            dw.append(np.dot(delta, activated_outputs[-(i+2)].T))
            db.append(np.sum(delta))
           
            if i is len(self.bias) - 1:
                break

            delta = np.dot(self.weight_matrices[-(i+1)].T, delta) * \
                self.act_func.activate_prime(weighted_inputs[-(i+2)])
            
        # return reversed list of dw and db also loss
        return dw[::-1], db[::-1], loss

    def print_loss(self):
        
        plt.plot(self.loss)
        plt.show()
      

if __name__ == '__main__':

    X = np.array([[0, 0],
                  [1, 0],
                  [0, 1],
                  [1, 1]])

    Y = np.array([[0],
                  [1],
                  [1],
                  [0]])
    tanhh = activate_tanh()
    NN = NeuralNetwork([2, 3, 3, 1], tanhh)

    w, b = NN.train(X.T, Y.T, 0.01, 10000)

    NN.predict(X.T)
  
