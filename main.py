from modle_name import modle_names
from snake_game import SnakeGame
from snake_data_manipulation import Snake_Data_Manipulation
from random_snake import Random_Snake
from conv2_full_dim import Conv2_Full_Dim
from full_dim import Full_Dim
from boost_snake import Boosting

def main_run(model_name, gui=False, number_of_itertion=1):
    game = SnakeGame(gui=gui, board_height=10, board_width=10)
    model = None

    if model_name == modle_names.random_snake:
        model = Random_Snake(game)
    elif model_name in [modle_names.survivor_snake, modle_names.survivor_snake__food_dist, modle_names.survivor_snake__food_angle,modle_names.survivor_snake__food_dist__current_food_dist]:
        model = Snake_Data_Manipulation(game, mode=model_name.name)

    elif model_name in [modle_names.full_screen_snake,modle_names.full_screen_snake__current_full_screen_snake]:
        model = Full_Dim(game, mode=model_name.name)

    elif model_name in [modle_names.cnn_full_screen_snake__current_full_screen_snake, modle_names.cnn_full_screen_snake]:
        model = Conv2_Full_Dim(game, mode=model_name.name)

    elif model_name in [modle_names.boost_survivor_snake__food_dist,modle_names.boost_survivor_snake__food_angle,modle_names.boost_survivor_snake__food_dist__current_food_dist]:
        model = Boosting(game, mode=model_name.name)

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


def get_enum_map():
    return {
        modle_names.random_snake.name : modle_names.random_snake.value,
        modle_names.survivor_snake.name : modle_names.survivor_snake.value,
        modle_names.survivor_snake__food_dist.name : modle_names.survivor_snake__food_dist.value,
        modle_names.survivor_snake__food_dist__current_food_dist.name : modle_names.survivor_snake__food_dist__current_food_dist.value,
        modle_names.survivor_snake__food_angle.name : modle_names.survivor_snake__food_angle.value,
        modle_names.full_screen_snake.name : modle_names.full_screen_snake.value,
        modle_names.full_screen_snake__current_full_screen_snake.name : modle_names.full_screen_snake__current_full_screen_snake.value,
        modle_names.cnn_full_screen_snake.name : modle_names.cnn_full_screen_snake.value,
        modle_names.cnn_full_screen_snake__current_full_screen_snake.name : modle_names.cnn_full_screen_snake__current_full_screen_snake.value,
        modle_names.boost_survivor_snake__food_dist.name : modle_names.boost_survivor_snake__food_dist.value,
        modle_names.boost_survivor_snake__food_dist__current_food_dist.name : modle_names.boost_survivor_snake__food_dist__current_food_dist.value,
        modle_names.boost_survivor_snake__food_angle.name : modle_names.boost_survivor_snake__food_angle.value,
        }
if __name__ == "__main__":
    while(True):

        gui = with_gui()
        num_games = number_of_games()
        line = 'choose \n'
        for key, value in get_enum_map().items():
            line = line + "{}. {} \n".format(value[0], key)
        num = int(input(line))

        if num == modle_names.random_snake.value[0]:
            main_run(modle_names.random_snake, gui=gui, number_of_itertion=num_games)
        elif num == modle_names.survivor_snake__food_dist.value[0]:
            main_run(modle_names.survivor_snake__food_dist, gui=gui, number_of_itertion=num_games)
        elif num == modle_names.survivor_snake.value[0]:
            main_run(modle_names.survivor_snake, gui=gui, number_of_itertion=num_games)
        elif num == modle_names.survivor_snake__food_angle.value[0]:
            main_run(modle_names.survivor_snake__food_angle, gui=gui, number_of_itertion=num_games)
        elif num == modle_names.survivor_snake__food_dist__current_food_dist.value[0]:
            main_run(modle_names.survivor_snake__food_dist__current_food_dist, gui=gui, number_of_itertion=num_games)
        elif num == modle_names.full_screen_snake.value[0]:
            main_run(modle_names.full_screen_snake, gui=gui, number_of_itertion=num_games)
        elif num == modle_names.full_screen_snake__current_full_screen_snake.value[0]:
            main_run(modle_names.full_screen_snake__current_full_screen_snake, gui=gui, number_of_itertion=num_games)
        elif num == modle_names.cnn_full_screen_snake.value[0]:
            main_run(modle_names.cnn_full_screen_snake, gui=gui, number_of_itertion=num_games)
        elif num == modle_names.boost_survivor_snake__food_angle.value[0]:
            main_run(modle_names.boost_survivor_snake__food_angle, gui=gui, number_of_itertion=num_games)
        elif num == modle_names.boost_survivor_snake__food_dist.value[0]:
            main_run(modle_names.boost_survivor_snake__food_dist, gui=gui, number_of_itertion=num_games)
        elif num == modle_names.boost_survivor_snake__food_dist__current_food_dist.value[0]:
            main_run(modle_names.boost_survivor_snake__food_dist__current_food_dist, gui=gui, number_of_itertion=num_games)
        elif num == modle_names.cnn_full_screen_snake__current_full_screen_snake.value[0]:
            main_run(modle_names.cnn_full_screen_snake__current_full_screen_snake, gui=gui, number_of_itertion=num_games)
        else :
            break

