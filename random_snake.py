from random import randint

import keras
import numpy as np
from base_model import BaseModel
from snake_game import SnakeGame
from modle_name import modle_names
from utils import _snake_utils

class Random_Snake(BaseModel):

    def __init__(self, game: SnakeGame):
        super().__init__(game, modle_names.random_snake.name)



    def get_model(self):
        pass

    def calculate_err(self, key):
        pass

    def predict_move(self):
        action = randint(-1, 1)
        key = self.get_key_from_direction(self.get_snake_direction_vector(self.game.snake), action)
        return key

    def update_date(self, move):
        pass

    def get_data(self, action, snake):
       pass


    def run(self, number_of_itertion=1):
        for _ in range(number_of_itertion):
            self.run_game()
