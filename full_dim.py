import keras
import numpy as np
from base_model import BaseModel
from snake_game import SnakeGame
from modle_name import modle_names
from utils import _snake_utils

class Full_Dim(BaseModel):

    def __init__(self, game: SnakeGame,mode=modle_names.full_screen_snake):
        super().__init__(game, mode.name)
        if mode == modle_names.full_screen_snake__current_full_screen_snake.name:
            self.size_x = (self.game.board['width'] + 1) * (self.game.board['height'] + 1)*2 + 1
            self.get_data_function =self.get_data_current_future
        elif mode == modle_names.full_screen_snake.name:
            self.size_x = (self.game.board['width'] + 1) * (self.game.board['height'] + 1) + 1
            self.get_data_function =self.get_data_future

    def get_model(self):
        self.model = keras.models.Sequential()
        self.model.add(keras.layers.Dense(100, input_dim=self.size_x, activation='relu'))
        self.model.add(keras.layers.Dense(50, input_dim=self.size_x, activation='relu'))
        self.model.add(keras.layers.Dense(30, input_dim=self.size_x, activation='relu'))
        self.model.add(keras.layers.Dense(5, activation='relu'))
        self.model.add(keras.layers.Dense(1, activation='linear'))
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


    def calculate_err(self, key):
        snake = _snake_utils.create_new_point(key, self.game.snake)
        if _snake_utils.snake_die(self.game, key):
            return 1
        elif self.game.food == snake[0]:
            self.num_of_step_to_apple = 0
            return 0
        else:
            return _snake_utils.distance_from_food(snake, self.game.food, self.game)


    def predict_move(self):
        predictions = []
        tmp_data = []
        for action in range(-1, 2):
            tmp_data.append(self.get_data(action))
            predictions.append(
                    self.model.predict((tmp_data[action+1]).reshape(1, self.size_x)))

        move = np.argmin(np.array(predictions)) - 1
        key = self.get_key_from_direction(self.get_snake_direction_vector(self.game.snake), move)
        self.data = self.data + [np.append(tmp_data[move+1],[self.calculate_err(key)])]
        return key

    def update_date(self, move):
        pass

    def get_data(self, action):
        key = self.get_key_from_direction(self.get_snake_direction_vector(self.game.snake), action)
        future_snake = _snake_utils.create_new_point(key, self.game.snake)
        return self.get_data_function(action, self.game.snake, future_snake, self.game.food)

    def get_data_future(self,action, current_snake, future_snake, food):
        return np.append([action], self.generate_observation_full_screen(future_snake, food))

    def get_data_current_future(self,action, current_snake, future_snake, food):
        return np.append(np.append([action], self.generate_observation_full_screen(current_snake, food)),self.generate_observation_full_screen(future_snake, food))


    def run(self, number_of_itertion=1):
        self.get_model()
        self.load_model()
        for _ in range(number_of_itertion):
            self.run_game()
            self.train_model()
            self.refresh_game()
        self.save_model()

    def generate_observation_full_screen(self, snake, food):
        board_vector = np.zeros((self.game.board['width'] + 1) * (self.game.board['height'] + 1))
        snake_head = snake[0]
        for s in snake:
            board_vector[s[0] * self.game.board['width'] + s[1]] = 1
        board_vector[snake_head[0] * self.game.board['width'] + snake_head[1]] += 1
        board_vector[food[0] * self.game.board['width'] + food[1]] = -1
        return np.array(board_vector)