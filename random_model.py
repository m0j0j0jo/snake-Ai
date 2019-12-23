import pickle
from random import randint

import numpy as np

from modle_name import modle_names
from ml_model import MLModle
from snake_game import SnakeGame


# from .modle_name import modle_names
# from .ml_model import MLModle
# from .snake_game import SnakeGame


class RandomModel(MLModle):

    def __init__(self, game: SnakeGame):
        super().__init__(game)
        self.game = game

    def train_model(self):
        pass

    def save_mod(self):
        pass

    def load_model(self):
        pass

    def predict_move(self):
        move = randint(0, 2) -1
        key = self.update_date(move)
        return key

    def run(self, number_of_itertion):
        self.run_game(number_of_itertion)
        self.save_data()

    def update_date(self, move):
        key = self.get_key_from_direction(self.get_snake_direction_vector(self.game.snake), move)
        # print(' self.data_survive {}'.format([[move] + self.generate_observation_survive(self.game.snake) + [self.calculate_err(key)]]))
        self.data_survive = self.data_survive + [[move] + list(self.generate_observation_survive(self.game.snake)) + [self.calculate_err_survive(key)]]
        self.data_feed_the_snake = self.data_feed_the_snake + [[move] + [self.generate_observation_feed_the_snake(self.game.snake, self.game.food)] + [self.calculate_err_feed_the_snake(key)]]
        self.data_full_screen = self.data_full_screen + [[move] + list(self.generate_observation_full_screen(self.game.snake, self.game.food)) + [self.calculate_err_full_screen(key)]]
        return key








