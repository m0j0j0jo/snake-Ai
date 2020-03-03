
import os
from abc import abstractmethod
from utils import _snake_utils
from snake_game import SnakeGame
import numpy as np

class BaseModel:

    def __init__(self, game: SnakeGame, name: str):
        self.game = game
        self.name = name
        self.vectors_and_keys = [
                                [-1, 0],
                                [0, 1],
                                [1, 0],
                                [0, -1]
                                ]
        self.Rotation_matrix = np.array([[0, -1], [1, 0]])
        self.prev_score = 0
        self.model = None
        self.size_x = 0
        self.input_X = []
        self.input_Y = []
        self.data = []
        self.num_of_step_to_apple = 0
        self.old_weights = None

    @abstractmethod
    def get_model(self):
        pass

    @abstractmethod
    def calculate_err(self, key):
        pass

    @abstractmethod
    def predict_move(self):
        pass

    @abstractmethod
    def run(self, number_of_itertion=1):
        pass

    @abstractmethod
    def update_date(self, move):
        pass

    def save_model(self):
        model_json = self.model.to_json()
        with open(self.name + '_model.json', "w") as json_file:
            json_file.write(model_json)
        self.model.save_weights(self.name + '_weights.h5')

    def load_model(self):
        if os.path.exists(self.name + '_weights.h5'):
            self.model.load_weights(self.name + '_weights.h5')

    def train_model(self):
        self.input_data()
        self.model.fit(self.input_X, self.input_Y)

    def input_data(self):
        input_file = self.data
        # input_file = pickle.load(open(self.name + '.txt', 'rb'))
        collect_x = np.array([])
        collect_y = np.array([])
        for line in input_file:
            collect_x = np.append(collect_x, [line[0:self.size_x]])
            collect_y = np.append(collect_y, [line[self.size_x]])
        self.input_X = np.array(collect_x).reshape(-1, self.size_x)
        self.input_Y = np.array(collect_y).reshape(-1, 1)
        self.data = []

    def save_data(self):
        pass

    def run_game(self):
        self.game.start()
        self.num_of_step_to_apple = 0
        for _ in range(200):
            self.num_of_step_to_apple += 1
            key = self.predict_move()
            try:
                if _snake_utils.snake_die(self.game, key):
                    self.game.done = True
                self.game.step(key)
            except:
                break

    def get_key_from_direction(self, direction, move):
        """
        move right -1
        move forward 0
        move left 1
        """

        if move == 0:
            return self.vectors_and_keys.index(list(direction))
        L = np.ndarray.tolist(np.matmul(move * self.Rotation_matrix, direction))
        return self.vectors_and_keys.index(L)

    def refresh_game(self):
        self.game = SnakeGame(gui=self.game.gui, board_height=self.game.board['height'], board_width=self.game.board['width'])

    def generate_observation_survive(self, snake):
        snake_direction = self.get_snake_direction_vector(snake)
        barrier_left = self.is_direction_blocked(snake, self.turn_vector_to_the_left(snake_direction))
        barrier_front = self.is_direction_blocked(snake, snake_direction)
        barrier_right = self.is_direction_blocked(snake, self.turn_vector_to_the_right(snake_direction))
        return np.array([int(barrier_left), int(barrier_front), int(barrier_right)])

    def add_action_to_observation(self, observation, action):
        return np.append([action], observation)

    def get_snake_direction_vector(self, snake):
        return np.array(snake[0]) - np.array(snake[1])

    def is_direction_blocked(self, snake, direction):
        point = np.array(snake[0]) + np.array(direction)
        return point.tolist() in snake[:-1] or point[0] == 0 or point[1] == 0 or point[0] == 21 or point[1] == 21

    def turn_vector_to_the_left(self, vector):
        return np.array([-vector[1], vector[0]])

    def turn_vector_to_the_right(self, vector):
        return np.array([vector[1], -vector[0]])