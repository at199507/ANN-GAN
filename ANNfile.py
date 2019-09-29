import numpy as np
import pandas as pd
import random
from math import exp
import mnist
import matplotlib.pyplot as plt

class ANN:
    def __init__(self,layers,X = [],y = [],loss = 'CV',activation_function = 'sigmoid',cost_function = 'cross_entropy',learning_rate = 0.01,batch_length = 1,randomise = True,final_activation_function = 'sigmoid',optimiser = 'SGD',X_test = [],y_test = []):
        self.cost_fun =  cost_function # quadratic,
        self.batch_length = batch_length
        self.act_fun = activation_function
        self.final_act_fun = final_activation_function
        self.optimiser = optimiser
        self.layers = layers
        self.loss = loss
        self.learning_rate = learning_rate
        self.X = X
        self.y = y
        self.init_weights()
        self.beta_1 = 0.9
        self.beta_2 = 0.999
        self.m = [0,0]
        self.v = [0,0]
        self.t = [0,0]
        self.epsilon = 0.0001
        rng_state = np.random.get_state()
        self.X_test = X_test
        self.y_test = y_test
        self.randomise = randomise

    #initialises the weights and biases of each layer to a random value
    def init_weights(self):
        #weights format = [layer][inputnode][outputnode]
        self.weights = np.array([np.random.random((self.layers[i + 1], self.layers[i])).transpose()/self.layers[i] for i in range(len(self.layers) - 1)])
        self.bias = np.array([np.random.random(self.layers[i+1]) for i in range(len(self.layers) - 1)])
        return [self.weights,self.bias]

    #given an input predicts
    def predict(self,input_array):
        x = np.array(input_array)
        for i in range(len(self.weights)):
            z = self.weights[i].transpose().dot(x)+self.bias[i]

            #if on last layer do final layer activation function
            if i == len(self.weights)-1:
                hold = self.act_fun
                self.act_fun = self.final_act_fun
                x = self.act_function(x = z, code = 0)
                self.act_fun = hold
            #else do the standard activation function
            else:
                x = self.act_function(x = z, code = 0)
        return x

    #backpropogation to retrieve gradients
    def backprop(self,x,y):

        #a_values holds the outputs of each layer (after the activation function)
        a_values  = []

        #the z_dash holds these values multiplied by differentiated activation function
        #z_dash = f_activation'(x)  this information is needed for the backward pass later
        z_dash = []
        a_values.append(x)
        z_dash.append(x)

        #the forward pass
        for i in range(len(self.weights)):
            #if at the final layer ensure the final activation function is used
            if i == len(self.weights)-1:
                hold = self.act_fun
                self.act_fun = self.final_act_fun

                #forward pass calculate a and z
                z = self.weights[i].transpose().dot(x) + self.bias[i]
                x = self.act_function(x=z, code=0)
                a_values.append(x)
                #z_dash_layer = self.act_function(x=x, code=1)

                # z_dash = f_activation'(x)  this information is needed for the backward pass later
                z_dash_layer = self.act_function(x=z, code=1)

                z_dash.append(z_dash_layer)
                self.act_fun = hold
            else:
                #retrieve the value at each node in the layer before the activation function
                z = self.weights[i].transpose().dot(x) + self.bias[i]
                #retrieve the value after the activation function
                x = self.act_function(x=z,code = 0)
                #add these values to the a_values
                a_values.append(x)

                #z_dash = f_activation'(x)  this information is needed for the backward pass later
                z_dash_layer = self.act_function(x=z, code = 1)

                z_dash.append(z_dash_layer)

        #lambda_vales is the change in cost function with respect to Z of a node, array populated during the backward pass
        self.lambda_values = np.empty_like(a_values)

        hold = self.act_fun
        self.act_fun = self.final_act_fun

        #create final lambda values to start backpropogation (code 2 means cost function takes activation function into account)
        self.lambda_values[-1] = np.ones_like(x)*self.cost_function(x=x, y=y, code=2)
        self.act_fun = hold

        #going backwards
        for i in range(len(self.weights)):

            #z = is the position counter of the layer, starting at the penultimate layer of the network and going backwards
            z = len(self.weights)-i-1

            #this handles the interactions of the two layers going backwards through the network
            #for each node in the layer lambda = Z_dash*sum(weight*corresponding lambda of layer in front) - (for each node infront)
            self.lambda_values[z] = np.array([z_dash[z][j]*(self.lambda_values[z+1].transpose().dot((self.weights[z][j]))) for j in range(len(self.weights[z]))])

        #weight gradients = a_values (of previous layer)* lambda
        weight_dash = np.array([(a_values[ii]*self.lambda_values[ii+1][:, np.newaxis]).transpose() for ii in range(len(self.weights))])

        #bias values = lambda
        bias_dash = np.array([self.lambda_values[i+1] for i in range(len(self.lambda_values)-1)])

        return weight_dash, bias_dash

    #update the weights
    def update_weights(self,learning_rate,weight_dash,bias_dash):

        #optimiser function takes the gradients and amends them as necessary for ADAM, SGD
        #i == 0 means weights, i == 1 means biases
        self.weights = self.weights - self.optimiser_func(weight_dash,i=0)*learning_rate
        self.bias = self.bias - self.optimiser_func(bias_dash,i=1)*learning_rate

    #for batch gradient descent calculate the average of the gradients
    def get_weight_bias_dash(self,input_array,y_array):
        total_weights = 0
        total_biases = 0
        for ii in range(len(input_array)):
            weights_dash, biases_dash = self.backprop(x = input_array[ii],y = self.label_to_y(y_array[ii],self.layers[-1]))
            total_weights += weights_dash
            total_biases += biases_dash
        av_weight_dash = total_weights/len(input_array)
        av_bias_dash = total_biases/len(input_array)
        return av_weight_dash, av_bias_dash

    #get cost function with respect to an input and a target
    def get_cost(self,input_array,y):
        x = self.predict(input_array =input_array)
        return self.cost_function(x = x,y = y, code = 0)


    def epoch_fun(self):
        if self.randomise:
            rng_state = np.random.get_state()
            np.random.shuffle(self.X)
            np.random.set_state(rng_state)
            np.random.shuffle(self.y)
        run_number = 0
        for ii in range(0,len(self.X),self.batch_length):

            run_number += 1

            #get array of input values
            input_array = self.X[ii:ii+self.batch_length]

            #get array of target values
            y_array = self.y[ii:ii+self.batch_length]

            #get the gradients averaged over the batch
            weight_dash,bias_dash = self.get_weight_bias_dash(input_array = input_array,y_array = y_array)

            #update the weights and biases
            self.update_weights(learning_rate=self.learning_rate,weight_dash = weight_dash,bias_dash = bias_dash)

            #these two lines exist to make a record of the cost function
            cost = self.get_cost(input_array=input_array[-1],y=self.label_to_y(y_array[-1],self.layers[-1]))
            self.cost_array.append(cost)

            #some validation to track the networks performance during training
            if run_number % 100 == 0:
                rate = self.random_predictions(1000)
                self.rate_array.append(rate)
                print("ii = ", ii, " /",len(self.X),"(train data) rate = ", rate,"%")

    def train(self,epochs):
        self.epochs = epochs
        self.cost_array = []
        self.rate_array = []
        for i in range(self.epochs):
            print("Epoch ",i+1,"/",self.epochs,":")
            self.epoch_fun()
            print("Epoch ", i+1, "/", self.epochs)
            print("Train data rate = ", self.random_predictions(5000),"%")
            if not self.X_test == []:
                print("Test data rate = ", self.predict_test(self.X_test,self.y_test),"%")

    #returns training accuracy using a number of random samples
    def random_predictions(self,no_preds):
        count = 0
        idxs = random.sample(range(0, len(self.X)), no_preds)
        for ii in idxs:
            predicted = self.predict(self.X[ii])
            if np.where(predicted==max(predicted))[0] == self.y[ii]:
                count += 1
        return count/no_preds*100

    #predict across the whole test set
    def predict_test(self,X_test,y_test):
        count = 0
        for ii in range(len(X_test)):
            predicted = self.predict(X_test[ii])
            if np.where(predicted == max(predicted))[0] == y_test[ii]:
                count += 1
        return count/len(X_test)*100

    @staticmethod
    def mnist_to_array(image_source, image_number):
        return np.array(image_source[image_number].flatten())

    #activation functions
    def act_function(self,x,code = 0):
        if self.act_fun == 'sigmoid':
            if code == 0:
                return np.array(1 / (1 + np.exp(x * -1)))
            elif code == 1:
                # same as:
                return np.exp(x*-1)/pow((1+np.exp(x*-1)),2)
                #return np.array(x * (1 - x))
        if self.act_fun == "ReLU":
            if code == 0:
                return np.array([x[a] if x[a] > 0 else 0 for a in range(len(x))])
            elif code == 1:
                return np.array([1 if x[a] > 0 else 0 for a in range(len(x))])

    #function to return the cost function
    #if code == 0, returns cost function
    #if code == 1, returns the result of the differentiated cost function
    #if code == 2, returns the result of the cost function and the differentiated activation function
    #this exists because the cross entropy cost function and sigmoid activation function simplifies to y - x during backpropogation
    def cost_function(self,x,y,code):
        if self.cost_fun == 'quadratic':
            if code == 0:
                return (sum(pow(y - x, 2))) / (2 * len(x))
            elif code == 1:
                return (x - y)
            elif code == 2:
                if self.act_fun == 'sigmoid':
                    return (x-y) * x* (1-x)
                else:
                    return (x - y) * self.act_function(x=x, code=1)
        elif self.cost_fun == 'cross_entropy':
            if code == 0:
                return np.array((-1 / len(x)) * (sum(y * np.log(x) + (1 - y) * np.log(1 - x))))
            elif code == 1:
                return np.array((-1 / len(x)) * (-1 * y * (1 / x) + 1 / (1 - x) - y / (1 - x)))
            elif code == 2:
                if self.act_fun == 'sigmoid':
                    return np.array((1 / len(x)) * (x - y))
                else:
                    return np.array((-1 / len(x)) * (-1*y*(1/x) + 1/(1-x) - y/(1-x)))*self.act_function(x=x, code=1)
        elif self.cost_fun == 'no_fun': #no cost function
            if code == 0:
                return x
            elif code == 1:
                return np.ones(len(x))
            elif code == 2:
                return np.ones(len(x))*self.act_function(x=x, code=1)

        #special case to enable the network to connect to others made using this class for GAN creation
        elif self.cost_fun == 'take_y':
            if code == 0 or code == 1:
                return y
            elif code == 2:
                if self.act_fun == 'sigmoid':
                    return np.array(y*x*(1-x))
                else:
                    return np.array(y)*self.act_function(x=x, code=1)

    #functionality to enable ADAM optimiser use
    def optimiser_func(self,gradients,i):
        if self.optimiser == 'ADAM':
            return self.run_ADAM(gradients,i = i)*gradients
        if self.optimiser == 'SGD':
            return gradients

    #function to calculate how ADAM changes the gradients
    def run_ADAM(self,gradients,i):
        self.t[i] += 1
        self.m[i] = self.beta_1 * self.m[i] + (1-self.beta_1)*gradients
        self.v[i] = self.beta_2 * self.v[i] + (1 - self.beta_2) * np.power(gradients,2)
        m_dash = self.m[i] / (1-np.power(self.beta_1, self.t[i]))
        v_dash = self.v[i] / (1 - np.power(self.beta_2, self.t[i]))
        return m_dash /(np.power(v_dash,0.5)+self.epsilon)

    #create target array from one number (one hot encode)
    @staticmethod
    def label_to_y(label,length):
        out = np.zeros(length)
        out[label] = 1
        return out



def main():
    route = 1
    X = np.array([x.flatten() / 256 for x in mnist.train_images()])
    y = mnist.train_labels()
    X_test = np.array([x.flatten() / 256 for x in mnist.test_images()])
    y_test = np.array(mnist.test_labels())

    if route == 1:
        b = ANN(layers = [784, 30, 10],
                X=X,
                y=y,
                cost_function='cross_entropy',
                activation_function='ReLU',
                #activation_function='sigmoid',
                #optimiser='ADAM',
                batch_length = 15,
                learning_rate = 1,
                randomise = True,
                X_test=X_test,
                y_test=y_test,
                )
        b.train(epochs = 1)
        b.predict_test(X_test = X_test,
                       y_test = y_test
                       )

if __name__ == "__main__":
    main()