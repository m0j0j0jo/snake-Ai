# from .snake_game import SnakeGame
import math
import pickle

from modle_name import modle_names
from snake_game import SnakeGame

# from snake_project.modle_name import modle_names
# from snake_project.snake_game import SnakeGame
import numpy as np

class MLModle():

    def __init__(self, game: SnakeGame):
        self.game = game
        self.vectors_and_keys = [
                                [-1, 0],
                                [0, 1],
                                [1, 0],
                                [0, -1]
                                ]
        self.Rotation_matrix = np.array([[0, -1], [1, 0]])
        self.filename_survive = modle_names.survive.name + '.txt'
        self.filename_feed_the_snake = modle_names.feed_the_snake.name + '.txt'
        self.filename_full_screen = modle_names.full_screen.name + '.txt'
        self.data_survive = []
        self.data_feed_the_snake = []
        self.data_full_screen = []
        self.prev_dis = 4000
        self.prev_score = 0

    def train_model(self):
        pass

    def save_model(self):
        pass

    def calculate_err_survive(self, key):
        return 1 if self.snake_die(key) else 0

    def calculate_err_feed_the_snake(self, key):
        food_dis = self.distance_from_food(self.game.snake, self.game.food)
        if self.snake_die(key):
            return 1 + food_dis
        else:

            # print('self.prev_dis > food_dis {}'.format(self.prev_dis > food_dis))
            # print('self.game.score > self.prev_score {}'.format(self.game.score > self.prev_score))
            if self.prev_dis > food_dis or self.game.score > self.prev_score:
                self.prev_dis = food_dis
                self.prev_score = self.game.score
                return -1
            else:
                return food_dis

    def calculate_err_full_screen(self, key):
        return self.calculate_err_feed_the_snake(key)

    def load_model(self):
        pass

    def predict_move(self):
        pass

    def run_game(self, number_of_itertion):
        for num in range(number_of_itertion):
            # print('game number {}'.format(num))
            self.game.start()
            for _ in range(1000):
                key = self.predict_move()
                try:
                    if self.snake_die(key):
                        self.done = True
                    self.game.step(key)
                except:
                    # print('game number {} end '.format(num))
                    break


    def run(self, number_of_itertion=1):
        pass


    def snake_die(self, key):
        snake_head = [0, 0]
        snake_head[0] = self.game.snake[0][0]
        snake_head[1] = self.game.snake[0][1]
        if key == 0:
            snake_head[0] -= 1
        elif key == 1:
            snake_head[1] += 1
        elif key == 2:
            snake_head[0] += 1
        elif key == 3:
            snake_head[1] -= 1
        if (self.game.snake[0][0] <= 0 or
                snake_head[0] >= self.game.board["width"] + 1 or
                snake_head[1] <= 0 or
                snake_head[1] >= self.game.board["height"] + 1 or
                snake_head in self.game.snake[1:-1]):
            return True
        return False

    def generate_observation_full_screen(self, snake, food):
        board_vector = np.zeros((self.game.board['width']+1)*(self.game.board['height']+1))
        for s in snake:
            board_vector[s[0]*self.game.board['width'] + s[1]] = 1
        board_vector[food[0] * self.game.board['width'] + food[1]] = -1
        snake_direction = self.get_snake_direction_vector(snake)
        return np.append(self.generate_observation_survive(snake),np.array(board_vector))



    def generate_observation_feed_the_snake(self, snake, food):
        food_vector = self.get_food_direction_vector(snake, food)
        snake_direction = self.get_snake_direction_vector(snake)
        return self.distance_from_food(snake, food)
        # return np.append(self.generate_observation_survive(snake),self.distance_from_food(snake, food))#self.get_angle(food_vector,snake_direction))

    def distance_from_food(self, snake, food):
        snake_head = snake[0]
        return (((snake_head[0] - food[0])**2 + (snake_head[1] - food[1])**2)**0.5)/30

    def get_angle(self, a, b):
        a = a / np.linalg.norm(a)
        b = b / np.linalg.norm(b)
        return math.atan2(a[0] * b[1] - a[1] * b[0], a[0] * b[0] + a[1] * b[1]) / math.pi

    def get_food_direction_vector(self, snake, food):
        return np.array(food) - np.array(snake[0])

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

    def save_data(self):
        with open(self.filename_survive, 'wb') as file_survive:
            pickle.dump(self.data_survive, file_survive)
        with open(self.filename_feed_the_snake, 'wb') as file_feed_the_snake:
            pickle.dump(self.data_feed_the_snake, file_feed_the_snake)
            # print(self.data_feed_the_snake)
        with open(self.filename_full_screen, 'wb') as file_full_screen:
            pickle.dump(self.data_full_screen, file_full_screen)

    def update_date(self, move):
        pass