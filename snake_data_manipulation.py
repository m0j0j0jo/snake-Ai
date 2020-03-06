import keras
import numpy as np
from base_model import BaseModel
from snake_game import SnakeGame
from modle_name import modle_names
from utils import _snake_utils
from numpy.random import choice
import pickle

class Snake_Data_Manipulation(BaseModel):

    def __init__(self, game: SnakeGame, mode=modle_names.survivor_snake__food_dist):
        super().__init__(game, mode.name)
        self.scores = []
        self.err = []
        self.size_x = 6
        if mode == modle_names.survivor_snake__food_dist.name:
            self.get_data_function = self.get_data_future_distance
        elif mode == modle_names.survivor_snake__food_dist__current_food_dist.name:
            self.size_x = 7
            self.get_data_function = self.get_data_current_future_distance
        elif mode == modle_names.survivor_snake__food_angle.name:
            print(modle_names.survivor_snake__food_angle.name)
            self.get_data_function = self.get_data_angle
        elif mode == modle_names.survivor_snake.name:
            self.size_x = 5
            self.get_data_function = self.get_data_defult



    def get_model(self):
        self.model = keras.models.Sequential()
        self.model.add(keras.layers.Dense(25, input_dim=self.size_x, activation='relu'))
        self.model.add(keras.layers.Dense(10, activation='relu'))
        self.model.add(keras.layers.Dense(1, activation='linear'))
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def calculate_err(self, key):
        snake = _snake_utils.create_new_point(key, self.game.snake)
        if _snake_utils.snake_die(self.game, key):
            return 3
        elif self.game.food == snake[0]:
            return 0
        else:
            return _snake_utils.distance_from_food(snake, self.game.food, self.game)

    def predict_move(self):
        predictions = []
        for action in range(-1, 2):
            predictions.append(
                    self.model.predict((self.get_data_from_action(action)).reshape(1, self.size_x)))
            self.err = self.err + [self.calculate_err(action) - predictions[action+1]]
        move = np.argmin(np.array(predictions)) - 1
        key = self.get_key_from_direction(self.get_snake_direction_vector(self.game.snake), move)
        self.data = self.data + [np.append(self.get_data_from_action(move), [self.calculate_err(key)])]
        return key

    def get_data_from_action(self, action):
        key = self.get_key_from_direction(self.get_snake_direction_vector(self.game.snake), action)
        snake = _snake_utils.create_new_point(key, self.game.snake)
        return self.get_data(key, self.game.snake, snake, self.game.food)

    def update_date(self, move):
        pass

    def get_data(self, action, current_snake, future_snake, food):
        return self.get_data_function(action, current_snake, future_snake, food)

    def get_data_future_distance(self, action, current_snake, future_snake, food):
        return np.append(
            [action, _snake_utils.distance_from_food(future_snake, food, self.game), self.is_snake_hit_himself(future_snake)],
            self.generate_observation_survive(future_snake))

    def get_data_angle(self, action, current_snake, future_snake, food):
        return np.append([action, _snake_utils.get_angle_snake_to_food(future_snake, food),
                          self.is_snake_hit_himself(future_snake)], self.generate_observation_survive(future_snake))

    def get_data_current_future_distance(self, action, current_snake, future_snake, food):
        return np.append([action, _snake_utils.distance_from_food(current_snake, food,self.game), _snake_utils.distance_from_food( future_snake, food,self.game),
                          self.is_snake_hit_himself(future_snake)], self.generate_observation_survive(future_snake))

    def get_data_defult(self, action, current_snake, future_snake, food):
        return np.append([action, self.is_snake_hit_himself(future_snake)], self.generate_observation_survive(future_snake))

    def is_snake_hit_himself(self, snake):
        if snake[0] in snake[1:]:
            return 1
        return 0

    def run(self, number_of_itertion=1):
        print('start running {} '.format(self.name))
        self.get_model()
        self.load_model()
        self.old_weights = self.model.get_weights()
        for _ in range(number_of_itertion):
            self.run_game()
            self.scores = self.scores + [self.game.score]
            self.train_model()
            self.refresh_game()
        self.save_model()
