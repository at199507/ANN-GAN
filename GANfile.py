import numpy as np
import pandas as pd
import random
import math
import mnist
import matplotlib.pyplot as plt
from datetime import datetime
from ANNfile import ANN
import pickle
import os
import pickle

class GAN:
    def __init__(self,X, generator, discriminator, noise_generator = "noise_generator_1", cost_function = "self.cross_entropy", discriminator_learning_rate = 1, generator_learning_rate=1, save=False, number = 2000):
        self.save = save
        self.a = 5
        self.discriminatorANN = ANN(discriminator,
                                    cost_function='take_y',
                                    activation_function='ReLU'
                                    )
                                    #optimiser='ADAM')
        self.generatorANN = ANN(generator,
                                cost_function='take_y',
                                activation_function='ReLU',
                                )
                                #optimiser='ADAM')

        self.noise_generator = "self."+noise_generator
        self.discriminator_learning_rate = discriminator_learning_rate
        self.generator_learning_rate = generator_learning_rate*-1
        self.generated_digit_list = []
        self.number = number
        self.input = self.get_random_input()
        self.X = X

        rng_state = np.random.get_state()
        self.train_images = mnist.train_images() / 256
        self.train_labels = mnist.train_labels()
        self.test_images = mnist.test_images() / 256
        self.test_labels = mnist.test_labels()
        self.startTime = datetime.now()
        self.cost_plot_real = []
        self.cost_plot_fake = []
        self.dis_prediction_real = []
        self.dis_prediction_fake = []
    def get_random_input(self):
        return eval(self.noise_generator)(number = self.generatorANN.layers[0],factor = 1)

    def fake_input_run(self):
        y = np.array([0, 1])
        input = self.get_random_input()
        input = self.input
        #input = [0,1,2,3,4]
        fake = self.generatorANN.predict(input_arr = input)
        self.save_generated_digits(fake)
        # the get cost function increase the predicting part
        self.cost = self.discriminatorANN.get_cost(input_array = fake,y = y)
        self.discriminator_weight_dash, self.descriminator_bias_dash = self.discriminatorANN.backprop(x = fake,y = self.cost_fun(y=y,code=0))

        #link to backprop through generator
        #the weight/bias dash above does not give us enough information to start backpropogation in the generator
        starting_values_for_GAN_backprop = self.discriminatorANN.lambda_values[0]
        self.generator_weight_dash, self.generator_bias_dash = self.generatorANN.backprop(x = input,y = starting_values_for_GAN_backprop)
        return self.generator_weight_dash, self.generator_bias_dash, self.discriminator_weight_dash, self.descriminator_bias_dash, self.cost


    def disc_get_real_weights(self,input):
        prediction = self.discriminatorANN.predict(input)
        y = self.dis_cost_fun_real(x = prediction,code=1)
        self.dis_prediction_real.append(prediction)
        self.descriminator_weight_dash, self.descriminator_bias_dash = self.discriminatorANN.backprop(x = input,y = y)
        return self.descriminator_weight_dash, self.descriminator_bias_dash

    def disc_get_fake_weights(self, input):
        prediction = self.discriminatorANN.predict(input)
        self.dis_prediction_fake.append(prediction)
        y = self.dis_cost_fun_fake(x = prediction, code=1)
        self.descriminator_weight_dash, self.descriminator_bias_dash = self.discriminatorANN.backprop(x=input, y=y)
        return self.descriminator_weight_dash, self.descriminator_bias_dash

    def gen_get_fake_weights(self, input):
        fake = self.generatorANN.predict(input)
        prediction = self.discriminatorANN.predict(fake)
        self.dis_prediction_fake.append(prediction)
        y = self.gen_cost_fun(x = prediction, code=1)
        self.discriminatorANN.backprop(x=fake, y=y)
        starting_values_for_GAN_backprop = self.discriminatorANN.lambda_values[0]
        self.generator_weight_dash, self.generator_bias_dash = self.generatorANN.backprop(x=input,
                                                                                          y=starting_values_for_GAN_backprop)
        return self.generator_weight_dash, self.generator_bias_dash

    def fake_batch(self,number = 1):
        #takes a number, runs SGD for that number of generated fake inputs
        gen_total_weights_dash = 0
        gen_total_bias_dash = 0
        disc_total_weights_dash = 0
        disc_total_bias_dash = 0
        total_cost = 0
        for i in range(number):
            gen_weight_dash,gen_bias_dash,disc_weight_dash,disc_bias_dash,cost = self.fake_input_run()
            gen_total_weights_dash += gen_weight_dash
            gen_total_bias_dash += gen_bias_dash
            disc_total_weights_dash += disc_weight_dash
            disc_total_bias_dash += disc_bias_dash
            total_cost += cost
        cost = total_cost/ number
        self.cost_plot_fake.append(self.cost)

        gen_av_weight_dash = gen_total_weights_dash/number
        gen_av_bias_dash = gen_total_bias_dash/number
        disc_av_weight_dash = disc_total_weights_dash / number
        disc_av_bias_dash = disc_total_bias_dash / number
        self.generatorANN.update_weights(learning_rate=self.generator_learning_rate,
                                         weight_dash=gen_av_weight_dash,
                                         bias_dash=gen_av_bias_dash)
        self.discriminatorANN.update_weights(learning_rate=self.discriminator_learning_rate,
                                             weight_dash=disc_av_weight_dash,
                                             bias_dash=disc_av_bias_dash)
    def real_batch(self,number = 1):
        #takes a number, runs SGD for that number of generated real inputs
        disc_total_weights_dash = 0
        disc_total_bias_dash = 0
        total_cost = 0
        for i in range(number):
            disc_weight_dash,disc_bias_dash,cost = self.real_input_run()
            disc_total_weights_dash += disc_weight_dash
            disc_total_bias_dash += disc_bias_dash
            total_cost += cost
        cost = total_cost/ number
        self.cost_plot_real.append(self.cost)

        disc_av_weight_dash = disc_total_weights_dash / number
        disc_av_bias_dash = disc_total_bias_dash / number

        self.discriminatorANN.update_weights(learning_rate=self.discriminator_learning_rate,
                                             weight_dash=disc_av_weight_dash,
                                             bias_dash=disc_av_bias_dash)

    def update_dicriminator(self,no_fake = 10,no_real = 10):
        fake_data = [self.generatorANN.predict(self.get_random_input()) for i in range(no_fake)]
        real_data = [self.get_real_input() for i in range(no_real)]
        total_weight_dash = np.zeros_like(self.discriminatorANN.weights)
        total_bias_dash = np.zeros_like(self.discriminatorANN.bias)
        for ii in range(len(real_data)):
            a,b = self.disc_get_real_weights(input = real_data[ii])
            total_weight_dash += a
            total_bias_dash += b
        av_weight_dash_real = total_weight_dash/len(real_data)
        av_bias_dash_real = total_bias_dash /len(real_data)

        total_weight_dash = np.zeros_like(self.discriminatorANN.weights)
        total_bias_dash = np.zeros_like(self.discriminatorANN.bias)
        for ii in range(len(fake_data)):
            a, b = self.disc_get_fake_weights(fake_data[ii])
            total_weight_dash += a
            total_bias_dash += b
        av_weight_dash_fake = total_weight_dash / len(fake_data)
        av_bias_dash_fake = total_bias_dash / len(fake_data)

        av_weight_dash = av_weight_dash_real + av_weight_dash_fake
        av_bias_dash = av_bias_dash_real + av_bias_dash_fake

        self.discriminatorANN.update_weights(learning_rate = -self.discriminator_learning_rate, weight_dash = av_weight_dash, bias_dash = av_bias_dash)

    def update_generator(self, no_fake=10):
        noise = [self.get_random_input() for i in range(no_fake)]
        total_weight_dash = np.zeros_like(self.generatorANN.weights)
        total_bias_dash = np.zeros_like(self.generatorANN.bias)
        for ii in range(len(noise)):
            a, b = self.gen_get_fake_weights(noise[ii])
            total_weight_dash += a
            total_bias_dash += b
        av_weight_dash = total_weight_dash / len(noise)
        av_bias_dash = total_bias_dash / len(noise)
        self.generatorANN.update_weights(learning_rate = self.generator_learning_rate, weight_dash = av_weight_dash, bias_dash = av_bias_dash)

    def run(self,no = 1000,disc_steps = 2,gen_steps = 1):
        for j in range(no):
            for i in range(disc_steps):
                self.update_dicriminator()
            for ii in range(gen_steps):
                self.update_generator()

    def get_real_input(self):
        return np.array(self.X[np.random.randint(0, len(self.X))])
        #return np.array(self.X[0].flatten())

    def get_cost(self, input = [0, 1, 2, 3, 4],y = np.array([0, 1])):
        fake = self.generatorANN.predict(input_arr=input)
        return self.discriminatorANN.get_cost(input_array=fake,y = y)

    def gen_cost_fun(self,x,code=0): #this is trying to min
        if code == 0:
            return math.log(1-x)
        if code == 1:
            return -1/(1-x)

    def dis_cost_fun_real(self,x,code=0): #this is trying to max, not min
        if code == 0:
            return math.log(x)
        if code == 1:
            return 1/x

    def dis_cost_fun_fake(self,x,code=0): #this is trying to max, not min
        if code == 0:
            return math.log(1-x)
        if code == 1:
            return -1/(1-x)

    def save_func(self):

        dt = datetime.today()
        day = dt.day
        month = dt.month

        name = str(month) + "/" + str(day)
        if not os.path.exists(name):
            os.makedirs(name)

        filehandler = open(name+'/'+str(dt.hour)+str(dt.minute)+'.'+str(self.number)+'.obj', 'wb')
        pickle.dump(self, filehandler)

        # go to month
        # if day not exist create
        # go to day
        # save file as time + 'number' + number

    def graphs(self):
        if self.save:
            self.save_func()
        plt.figure()
        plt.plot(self.dis_prediction_real,label = 'real')
        plt.show
        plt.plot(self.dis_prediction_fake,label = 'fake')
        plt.show
        plt.legend()
        print("End of run")
        print("wall time: ",datetime.now()-self.startTime)

    @staticmethod
    def noise_generator_1(number,factor):
        return np.random.random(number)*factor

    def save_generated_digits(self, digit):
        self.generated_digit_list.append(digit)

    @staticmethod
    def visualise_mnist(input):
        pixels = input.reshape(28, 28)
        plt.figure()
        plt.imshow(pixels, cmap='gray')
        plt.show(block = False)
    @staticmethod
    def mnist_to_array(image_source, image_number):
        return np.array(image_source[image_number].flatten())

    @staticmethod
    def mnist_to_cost(train_labels, image_number):
        out = np.zeros(10)
        out[train_labels[image_number]] = 1
        return out
def main():
    route = 2
    if route == 1:
        gan = GAN(generator = [5, 6, 5, 7], discriminator  = [7, 3, 2])
        gan.fake_input_run()
        cost = gan.cost
        print("Initial cost: ",cost)
        diff = gan.generator_weight_dash[1][5,4]
        print("weight dash: ",diff)
        wei = gan.generatorANN.weights[1][5,3]
        print("the actual weight",wei)
        gan.generatorANN.weights[1][5,3] *= 1.1
        newwei = gan.generatorANN.weights[1][5,3]
        print("the new weight", newwei)
        gan.fake_input_run()
        costnew = gan.get_cost()
        print("new cost = ",costnew)
        print("change = ",(costnew-cost)/(newwei-wei))
        #b = ANN(layers = di,
         #       cost_function = 'self.cross_entropy',
         #       batch_length = 10,
          #      learning_rate= 5,
         #       randomise = False)


        cost = gan.cost
        print("Initial cost: ", cost)
        print(gan.discriminator_weight_dash[0][5,2])
        diff = gan.discriminator_weight_dash[0][5,2]
        print("weight dash: ", diff)
        wei = gan.discriminatorANN.weights[0][5,2]
        print("the actual weight", wei)
        gan.discriminatorANN.weights[0][5, 2] += 0.2
        newwei = gan.discriminatorANN.weights[0][5,2]
        print("the new weight", newwei)
        gan.fake_input_run()
        costnew = gan.get_cost()
        print("new cost = ", costnew)
        print("change = ", (costnew - cost) / (newwei - wei))
    elif route == 2:
        X = np.array([x.flatten()/256 for x in mnist.train_images()])
        pos = np.where(mnist.train_labels() == 7)
        gan = GAN(generator=[100, 300, 500, 784],
                  discriminator=[784, 30, 1],
                  discriminator_learning_rate = 1,
                  generator_learning_rate = 1.5,
                  #X=X,
                  X=X[pos],
                  save = False)

        #gan.visualise_mnist(gan.train_images[64])
        a = gan.run(no = 300,disc_steps=1)
        #filehandler = open('testsave.obj', 'wb')
        #pickle.dump(gan, filehandler)
        gan.graphs()
        for ii in range(3):
            gan.visualise_mnist(gan.generatorANN.predict(gan.get_random_input()))
        #gan.visualise_mnist(gan.generated_digit_list[-1])
        plt.show()
if __name__ == "__main__":
    main()