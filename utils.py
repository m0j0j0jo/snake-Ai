import math
import numpy as np

class snake_utils:

    def __init__(self):
        pass

    def snake_die(self, game, key):
        snake_head = [0, 0]
        snake_head[0] = game.snake[0][0]
        snake_head[1] = game.snake[0][1]
        if key == 0:
            snake_head[0] -= 1
        elif key == 1:
            snake_head[1] += 1
        elif key == 2:
            snake_head[0] += 1
        elif key == 3:
            snake_head[1] -= 1
        if (game.snake[0][0] <= 0 or
                snake_head[0] >= game.board["width"] + 1 or
                snake_head[1] <= 0 or
                snake_head[1] >= game.board["height"] + 1 or
                snake_head in game.snake[1:-1]):
            return True
        return False

    def create_new_point(self, key, tmp_snake: list):
        snake = tmp_snake.copy()
        new_point = [snake[0][0], snake[0][1]]
        if key == 0:
            new_point[0] -= 1
        elif key == 1:
            new_point[1] += 1
        elif key == 2:
            new_point[0] += 1
        elif key == 3:
            new_point[1] -= 1
        snake.insert(0, new_point)
        return snake

    def food_eaten(self, game, snake):
        return snake[0] == game.food

    def snake_step(self, game, key):
        # 0 - UP
        # 1 - RIGHT
        # 2 - DOWN
        # 3 - LEFT
        snake = self.create_new_point(key, game.snake)
        if self.food_eaten(game, snake):
            pass
        else:
            snake.pop()
        return snake


    def distance_from_food(self, snake, food, game):
        snake_head = snake[0]
        return ((snake_head[0] - food[0])**2 + (snake_head[1] - food[1])**2)**0.5/((game.board['width']**2+game.board['height']**2)**0.5)

    def get_angle(self, a, b):
        if np.linalg.norm(a) != 0 and np.linalg.norm(b) != 0:
            a = a / np.linalg.norm(a)
            b = b / np.linalg.norm(b)
            return math.atan2(a[0] * b[1] - a[1] * b[0], a[0] * b[0] + a[1] * b[1]) / math.pi
        else:
            return 0

    def get_food_direction_vector(self, snake, food):
        return np.array(food) - np.array(snake[0])


    def get_snake_direction_vector(self, snake):
        return np.array(snake[0]) - np.array(snake[1])

    def get_angle_snake_to_food(self, snake, food):
        angle = self.get_angle(self.get_snake_direction_vector(snake), self.get_food_direction_vector(snake, food))
        return 0 if angle is None else angle


_snake_utils = snake_utils()