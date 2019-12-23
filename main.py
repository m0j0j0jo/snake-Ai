from random import randint

from modle_name import modle_names
from random_model import RandomModel
from snake_game import SnakeGame
from simple_nn import SimpleNN

from feed_the_snake_modle import Feed_The_Snake_Modle
from full_board import Full_Board
# from .modle_name import modle_names
# from .random_model import RandomModel
# from .snake_game import SnakeGame
# from .simple_nn import SimpleNN
# from feed_the_snake_modle import Feed_The_Snake_Modle


def main_run(model_name, gui=False, number_of_itertion=1):
    game = SnakeGame(gui=gui)
    model = None
    if model_name == modle_names.random:
        model = RandomModel(game)
    elif model_name == modle_names.survive:
        model = SimpleNN(game)
    elif model_name == modle_names.feed_the_snake:
        model = Feed_The_Snake_Modle(game)
    elif model_name == modle_names.full_screen:
        model = Full_Board(game)
    model.run(number_of_itertion)

def with_gui():
    ans = input('with gui?')
    if ans == 'Y' or ans == 'y':
        return True
    else:
        return False

def number_of_games():
    ans = int(input('number of games?'))
    return ans

if __name__ == "__main__":

    while(True):
        gui = with_gui()
        num_games = number_of_games()
        num = str(input('choose \n1. random. \n2. Survive \n3. Feed the snake\n4. full screen \n'))
        # gui = False
        # num_games = 1
        # num = 1
        if num == '1':
            main_run(modle_names.random, gui=gui, number_of_itertion=num_games)

        elif num == '2':
            main_run(modle_names.survive, gui=gui, number_of_itertion=num_games)

        elif num == '3':
            main_run(modle_names.feed_the_snake, gui=gui, number_of_itertion=num_games)

        elif num == '4':
            main_run(modle_names.full_screen, gui=gui, number_of_itertion=num_games)

        else :
            break