import os
import keras
import xgboost
import numpy as np
from base_model import BaseModel
from snake_game import SnakeGame
from modle_name import modle_names
from utils import _snake_utils

class Boosting(BaseModel):

    def __init__(self, game: SnakeGame, name=modle_names.boost_survivor_snake__food_dist__current_food_dist.name,mode=''):
        super().__init__(game, name)
        self.size_x = 6
        self.get_data_function = self.get_data_defult
        self.collect_data = True

        if mode == modle_names.boost_survivor_snake__food_dist:
            self.get_data_function = self.get_data_future_distance
        elif mode == modle_names.boost_survivor_snake__food_dist__current_food_dist.name:
            self.size_x = 7
            self.get_data_function = self.get_data_current_future_distance
        elif mode == modle_names.boost_survivor_snake__food_angle.name:
            self.get_data_function = self.get_data_angle

        self.get_data_survivor_snake__food_dist(str(name).replace('boost_', ''))
        self.collect_data = False
        self.game = SnakeGame(gui=True, board_height=self.game.board['height'], board_width=self.game.board['width'])

    def get_data_survivor_snake__food_dist(self,name):
        self.load_data_model(name)
        self.play_games(200)


    def play_games(self, number_of_games):
        for game_number in range(number_of_games):
            print(game_number)
            self.refresh_game()
            self.run_game()

    def load_data_model(self, name):
        self.model = keras.models.Sequential()
        self.model.add(keras.layers.Dense(25, input_dim=self.size_x, activation='relu'))
        self.model.add(keras.layers.Dense(10, activation='relu'))
        self.model.add(keras.layers.Dense(1, activation='linear'))
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        if os.path.exists(name + '_weights.h5'):
            self.model.load_weights(name + '_weights.h5')

    def get_model(self):
        self.model = xgboost.XGBRegressor(max_depth=8)

    def save_model(self):
        self.model.save_model(self.name + '.txt')

    def load_model(self):
        self.get_model()
        if os.path.exists(self.name + '.txt'):
            self.model.load_model(self.name + '.txt')

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
        move = np.argmin(np.array(predictions)) - 1

        key = self.get_key_from_direction(self.get_snake_direction_vector(self.game.snake), move)
        if self.collect_data:
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

    def get_data_current_future_distance(self, action, current_snake, future_snake, food):
        return np.append([action, _snake_utils.distance_from_food(current_snake, food,self.game), _snake_utils.distance_from_food( future_snake, food,self.game),
                          self.is_snake_hit_himself(future_snake)], self.generate_observation_survive(future_snake))

    def get_data_future_distance(self, action, current_snake, future_snake, food):
        return np.append(
            [action, _snake_utils.distance_from_food(future_snake, food, self.game), self.is_snake_hit_himself(future_snake)],
            self.generate_observation_survive(future_snake))

    def get_data_angle(self, action, current_snake, future_snake, food):
        return np.append([action, _snake_utils.get_angle_snake_to_food(future_snake, food),
                          self.is_snake_hit_himself(future_snake)], self.generate_observation_survive(future_snake))

    def get_data_defult(self, action, current_snake, future_snake, food):
        return np.append([action, _snake_utils.get_angle_snake_to_food(current_snake, food),
                          self.is_snake_hit_himself(future_snake)], self.generate_observation_survive(future_snake))


    def is_snake_hit_himself(self, snake):
        if snake[0] in snake[1:]:
            return 1
        return 0

    def run(self, number_of_itertion=1):
        print('start running {} '.format(self.name))
        self.load_model()
        self.train_model()
        self.save_model()
        for _ in range(number_of_itertion):
            self.run_game()

