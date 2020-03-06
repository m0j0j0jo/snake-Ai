import pickle

import keras
import numpy as np
from base_model import BaseModel
from snake_game import SnakeGame
from modle_name import modle_names
from utils import _snake_utils

class Conv2_Full_Dim(BaseModel):

    def __init__(self, game: SnakeGame, mode=modle_names.cnn_full_screen_snake):
        super().__init__(game, mode.name)
        if mode == modle_names.cnn_full_screen_snake__current_full_screen_snake.name:
            self.size_x = (self.game.board['width'] + 1) * (self.game.board['height'] + 1)*2
            self.get_data_function = self.get_data_current_future
            self.input_shape = ((self.game.board['width'] + 1)*2, self.game.board['height'] + 1, 1)
            self.input_reshape = self.input_reshape_current_future
        elif mode == modle_names.cnn_full_screen_snake.name:
            self.size_x = (self.game.board['width'] + 1) * (self.game.board['height'] + 1)
            self.input_shape = (self.game.board['width']+1, self.game.board['height']+1, 1)
            self.input_reshape = self.input_reshape_future
            self.get_data_function = self.get_data_future

    def input_reshape_future(self,count=1):
        return  (count, self.game.board['width'] + 1, self.game.board['height'] + 1, 1)

    def input_reshape_current_future(self,count=1):
        return  (count, (self.game.board['width'] + 1)*2, self.game.board['height'] + 1, 1)

    def get_model(self):
        self.model = keras.models.Sequential()
        self.model.add(keras.layers.Conv2D(5, (2,2), activation='relu', input_shape=self.input_shape))
        self.model.add(keras.layers.MaxPooling2D(pool_size=2))
        self.model.add(keras.layers.Flatten())
        self.model.add(keras.layers.Dense(50,  activation='relu'))
        self.model.add(keras.layers.Dense(30, activation='relu'))
        self.model.add(keras.layers.Dense(1, activation='linear'))
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def calculate_err(self, key):
        snake = _snake_utils.create_new_point(key, self.game.snake)
        if _snake_utils.snake_die(self.game, key):
            return 3
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
                    self.model.predict((tmp_data[action+1])))

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

    def get_data_future(self, action, current_snake, future_snake, food):
        return self.generate_observation_full_screen(future_snake, food).reshape(
            self.input_reshape())

    def get_data_current_future(self, action, current_snake, future_snake, food):
        return np.append(self.generate_observation_full_screen(current_snake, food),
                         self.generate_observation_full_screen(future_snake, food)).reshape(self.input_reshape())


    def run(self, number_of_itertion=1):
        self.get_model()
        self.load_model()
        self.old_weights = self.model.get_weights()
        for _ in range(number_of_itertion):
            self.run_game()
            self.train_model()
            self.refresh_game()
        self.save_model()

    def generate_observation_full_screen(self, snake, food):
        board_metrix = np.zeros((self.game.board['width']+1,self.game.board['height']+1))
        snake_head = snake[0]
        snake_head_x = board_metrix.shape[0] - 1 if (snake[0][0] >= board_metrix.shape[0]) else snake[0][0]
        snake_head_y = board_metrix.shape[1] - 1 if (snake[0][1] >= board_metrix.shape[1]) else snake[0][1]
        for s in snake:
            snake_x = (board_metrix.shape[0] - 1) if (s[0] >= board_metrix.shape[0]) else s[0]
            snake_y = (board_metrix.shape[1] - 1) if (s[1] >= board_metrix.shape[1]) else s[1]
            board_metrix[snake_x, snake_y] = 1

        board_metrix[snake_head_x,snake_head_y] += 1
        board_metrix[food[0]-1, food[1]-1] = -1
        return board_metrix

    def input_data(self):
        input_file = self.data
        collect_x = np.array([])
        collect_y = np.array([])
        count = 0
        for i in range(len(input_file)):
            line = input_file[i]
            count += 1
            collect_x = np.append(collect_x, np.array([line[0:self.size_x]]).reshape(self.input_reshape()))
            collect_y = np.append(collect_y, [line[self.size_x]])
        self.input_X = np.array(collect_x).reshape(self.input_reshape(count=count))
        self.input_Y = np.array(collect_y).reshape(-1, 1)
        self.data = []