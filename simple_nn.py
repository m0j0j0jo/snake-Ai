import pickle
from random import randint
import os
import numpy as np
import sys
sys.path.append(".")
# from .modle_name import modle_names
# from .ml_model import MLModle
# from .snake_game import SnakeGame

from modle_name import modle_names
from ml_model import MLModle
from snake_game import SnakeGame

from tflearn.layers.core import input_data, fully_connected
from tflearn.layers.estimator import regression
import tflearn


class SimpleNN(MLModle):

    def __init__(self, game: SnakeGame):
        super().__init__(game)
        self.size_x = 4
        self.filename_survive = modle_names.survive.name + '.txt'
        self.filename_random_input = modle_names.random.name + '.txt'
        self.data = []
        self.game = game
        self.input_data_shape = [None, self.size_x, 1]
        self.input_x = np.array([])
        self.input_y = np.array([])
        self.load_model_and_train()


    def get_model(self):
        network = input_data(shape=self.input_data_shape, name='input')
        network = fully_connected(network, 25, activation='relu')
        network = fully_connected(network, 1, activation='linear')
        network = regression(network, optimizer='adam', learning_rate=0.1, loss='mean_square', name='target')
        self.model = tflearn.DNN(network, tensorboard_dir='log')

    def save_mod(self):
        self.model.save(modle_names.survive.name + 'model')


    def calculate_err(self, key):
        pass

    def load_model_and_train(self):
        self.get_model()
        if os.path.exists(modle_names.survive.name + 'model'):
            self.model.load(modle_names.survive.name + 'model')
        self.input_data()
        self.model.fit(self.input_X, self.input_Y, n_epoch=1, shuffle=True, run_id=self.filename_survive)

    def get_data(self, action):
        return np.append([action], self.generate_observation_survive(self.game.snake))

    def predict_move(self):
        predictions = []
        for action in range(-1, 2):
            predictions.append(
                self.model.predict((self.get_data(action)).reshape(1, self.size_x, -1)))
        move = np.argmin(np.array(predictions)) -1
        key = self.get_key_from_direction(self.get_snake_direction_vector(self.game.snake), move)
        self.data_survive = self.data_survive + [list(self.get_data(move)) + [self.calculate_err(key)]]
        return key

    def input_data(self):
        input_file = pickle.load(open(self.filename_survive, 'rb'))
        temp_x = list()
        tempY = list()
        for line in input_file:
            temp_x = temp_x + ([list(line[:-1])])
            tempY = tempY + [line[-1]]
        # print('np.shape(self.input_X): {}'.format(temp_x))
        # print('np.shape(self.input_Y): {}'.format(tempY))
        self.input_X = np.array(temp_x).reshape(-1, self.size_x, 1)
        self.input_Y = np.array(tempY).reshape(-1, 1)
        # print('self.input_x: {}, self.input_y: {}'.format(self.input_X, self.input_Y))
        # print('np.shape(self.input_X): {}'.format(np.shape(self.input_X)))
        # print('np.shape(self.input_Y): {}'.format(np.shape(self.input_Y)))

    def run(self, number_of_itertion=1):
        self.run_game(number_of_itertion)
        # self.save_data()
        self.save_mod()

    def run_game(self, number_of_itertion):
        for _ in range(number_of_itertion):
            self.game.start()
            for _ in range(1000):
                key = self.predict_move()
                print(key)
                try:
                    self.game.step(key)
                except:
                    pass
                if self.snake_die(key):
                    self.game.render_destroy()
                    break

