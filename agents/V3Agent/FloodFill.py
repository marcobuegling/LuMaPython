import math
from queue import Queue
from typing import Set
import numpy as np
from typing import List, Tuple
from environment.Battlesnake.model.Direction import Direction

from environment.Battlesnake.model.Position import Position
from environment.Battlesnake.model.Snake import Snake
from environment.Battlesnake.model.board_state import BoardState

from numba import jit, njit
from numba import vectorize

@njit
def _in(value:tuple, x_list:list, y_list:list) -> bool:
    i = 0
    while i < len(x_list):
        if(x_list[i] == value[0] and y_list[i] == value[1]): return True
        i += 1
    
    return False

@njit
def _unravel(a:np.ndarray) -> Tuple:
    for y in range(a.shape[0]):
        for x in range(a.shape[1]):
            if(a[y,x] == 0):
                return (y,x)
    
    return None

@njit
def _to_string(a:np.ndarray) -> str:
    ret = "["
    
    i = 0
    while i < len(a):
        ret += str(a[i]) + ","
        
        i += 1
        
    if(len(ret) > 1):
        ret = ret[0: len(ret)-1] 
        
    return ret + "]"

@njit
def _calc(blocked_positions_array:np.ndarray) -> np.ndarray:
    '''Creates a ndarray storing the size of the are of free space connected to the square in each corresponding matrix cell. 
    The cells representing occupied squares have a value of 0.'''

    result = np.zeros(blocked_positions_array.shape)
    a = blocked_positions_array.copy()
    while 0 == np.min(a):
        first_free_x, first_free_y = _unravel(a)
        a[first_free_x, first_free_y] = 2
        
        to_expand_x = np.array([first_free_x])
        to_expand_y =  np.array([first_free_y])

        index_list_x = np.array([0])
        index_list_x = np.delete(index_list_x, 0)
        index_list_y = np.array([0])
        index_list_y = np.delete(index_list_y, 0)
        
        while len(to_expand_x) > 0:
            x = to_expand_x[0]
            y = to_expand_y[0]
            a[x, y] = 1
            
            if (x - 1) >= 0 and a[x - 1, y] == 0 and not _in((x - 1, y), to_expand_x, to_expand_y):
                to_expand_x = np.append(to_expand_x, np.array([(x - 1)]))
                to_expand_y = np.append(to_expand_y, np.array([y]))
            if (y - 1) >= 0 and a[x, y - 1] == 0 and not _in((x, y - 1), to_expand_x, to_expand_y):
                to_expand_x = np.append(to_expand_x, np.array([x]))
                to_expand_y = np.append(to_expand_y, np.array([(y-1)]))
            if (x + 1) < a.shape[0] and a[x + 1, y] == 0 and not _in((x + 1, y), to_expand_x, to_expand_y):
                to_expand_x = np.append(to_expand_x, np.array([(x + 1)]))
                to_expand_y = np.append(to_expand_y,  np.array([y]))
            if (y + 1) < a.shape[1] and a[x, y + 1] == 0 and not _in((x, y + 1), to_expand_x, to_expand_y):
                to_expand_x = np.append(to_expand_x, np.array([x]))
                to_expand_y = np.append(to_expand_y, np.array([(y + 1)]))
            if not _in((to_expand_x[0], to_expand_y[0]), index_list_x, index_list_y):
                index_list_x = np.append(index_list_x, np.array([to_expand_x[0]]))
                index_list_y = np.append(index_list_y, np.array([to_expand_y[0]]))
                
            to_expand_x = to_expand_x[1:]
            to_expand_y = to_expand_y[1:]
            size = len(index_list_x)
            
            #print(f"To expand: ({_to_string(to_expand_x)},{_to_string(to_expand_y)})")
            #print(f"Index list: ({_to_string(index_list_x)},{_to_string(index_list_y)})")
            
            i = 0
            while i < (len(index_list_x)):
                result[index_list_x[i], index_list_y[i]] = size
                i += 1
        
            
    return result

@njit
def _min_reduce(array1:np.ndarray, array2:np.ndarray):
    
    if(array1.shape != array2.shape):
        raise ValueError("The dimensions of the matrices do not match!")
    
    min_array = np.zeros(shape=array1.shape)
    
    height, width = min_array.shape
    
    y = 0
    while y < height:
        
        x = 0
        while x < width:
            
            option1 = array1[y,x]
            option2 = array2[y,x]
            
            min_array[y,x] = min(option1, option2)
            x += 1
        
        y += 1
    
    return min_array

@njit
def _calc_extended_worst_case(blocked_positions_array:np.ndarray) -> np.ndarray:
    '''Creates a ndarray storing the minimum size of the are of free space connected to the square 
    in each corresponding matrix cell when itroducing the worst possible blocked positions around the cell, but only one blocked cell is added at a time. 
    The cells representing occupied squares have a value of 0.'''

    min_aggregation = np.full(shape=blocked_positions_array.shape, fill_value=math.inf)

    height, width = blocked_positions_array.shape

    for y in range(height):
        for x in range(width):

            if(blocked_positions_array[y,x] == 0):

                blocked_positions_array [y,x] = 1
                calculated_flood_fill = _calc(blocked_positions_array=blocked_positions_array)
                calculated_flood_fill[y,x] = math.inf
                min_aggregation = _min_reduce(min_aggregation, calculated_flood_fill)
                blocked_positions_array[y,x] = 0

    return min_aggregation


class FloodFill:


    @staticmethod
    @njit
    def calc(blocked_positions_array:np.ndarray) -> np.ndarray:
        '''Creates a ndarray storing the size of the are of free space connected to the square in each corresponding matrix cell. 
        The cells representing occupied squares have a value of 0.'''

        return _calc(blocked_positions_array=blocked_positions_array)

    
    @staticmethod
    @njit
    def calc_extended_average(blocked_positions_array:np.ndarray) -> np.ndarray:
        '''Creates a ndarray storing the average size of the are of free space connected to the square 
        in each corresponding matrix cell when itroducing newly blocked positions around the cell. 
        The cells representing occupied squares have a value of 0.'''

        aggregation = np.zeros(shape=blocked_positions_array.shape)

        height, width = blocked_positions_array.shape

        counter = 0

        for y in range(height):
            for x in range(width):

                if(blocked_positions_array[y,x] == 0):
                    blocked_positions_array [y,x] = 1
                    aggregation = aggregation + _calc(blocked_positions_array=blocked_positions_array)
                    blocked_positions_array[y,x] = 0
                    counter += 1
        
        aggregation = aggregation / max(counter,1)

        return aggregation

    @staticmethod
    @njit
    def calc_extended_worst_case(blocked_positions_array:np.ndarray) -> np.ndarray:
        '''Creates a ndarray storing the minimum size of the are of free space connected to the square 
        in each corresponding matrix cell when itroducing the worst possible blocked positions around the cell, but only one blocked cell is added at a time. 
        The cells representing occupied squares have a value of 0.'''

        return _calc_extended_worst_case(blocked_positions_array=blocked_positions_array)


    @staticmethod
    @njit
    def calc_trap_heuristic(blocked_positions_array:np.ndarray) -> np.ndarray:
        '''Creates a ndarray storing the minimum size of the are of free space connected to the square 
        in each corresponding matrix cell when itroducing the worst possible blocked positions around the cell, but only one blocked cell is added at a time. 
        The cells representing occupied squares have a value of 0.'''

        space_loss = np.zeros(shape=blocked_positions_array.shape)
        
        
        height, width = blocked_positions_array.shape

        base_heuristic = _calc(blocked_positions_array=blocked_positions_array)
        base_free_space = base_heuristic.sum()

        # find the position that causes the biggest decrease in overall connected space
        for y in range(height):
            for x in range(width):

                if(blocked_positions_array[y,x] == 0):

                    blocked_positions_array [y,x] = 1
                    calculated_index_heuristic = _calc(blocked_positions_array)
                    
                    free_space = calculated_index_heuristic.sum()
                    
                    space_loss[y,x] = base_free_space - free_space
                    
                    blocked_positions_array[y,x] = 0

        return space_loss


#######################################################################################################################################################
    # test implementation


    @staticmethod
    def generate_examples() -> List[np.ndarray]: 
        state1 = np.zeros(shape=(8,6))

        state1[0,1] = 1
        state1[1,1] = 1
        state1[2,1] = 1
        state1[3,1] = 1
        state1[4,0] = 1
        state1[4,1] = 1
        state1[4,2] = 1
        state1[4,3] = 1
        state1[4,4] = 1
        state1[4,5] = 1

        state2 = np.zeros(shape=(8,6))

        state2[1,1] = 1
        state2[2,1] = 1
        state2[3,1] = 1
        state2[4,1] = 1
        state2[5,1] = 1
        state2[5,2] = 1
        state2[4,2] = 1
        state2[4,3] = 1
        state1[4,4] = 1
        state2[5,4] = 1
        state2[0,2] = 1
        state2[0,3] = 1
        state2[0,4] = 1
        state2[0,5] = 1
        state2[1,5] = 1
        state2[2,5] = 1
        state2[3,5] = 1
        state2[4, 4] = 1
        state2[4,5] = 1

        state3 = np.zeros(shape=(8,6))

        state3[1,1] = 1
        state3[2,0] = 1
        state3[2,2] = 1
        state3[3,1] = 1

        return [state3, state1, state2]
    

    @staticmethod
    def test_calc():
        print("\nTesting FloodFill!\n")
        
        expected_heuristic_1 = np.array(
            [
                [4,0,16,16,16,16],
                [4,0,16,16,16,16],
                [4,0,16,16,16,16],
                [4,0,16,16,16,16],
                [0,0,0,0,0,0],
                [18,18,18,18,18,18],
                [18,18,18,18,18,18],
                [18,18,18,18,18,18],
            ], dtype=float)
        
        expected_heuristic_2 = np.array(
            [
                [21,21,0,0,0,0],
                [21,0,9,9,9,0],
                [21,0,9,9,9,0],
                [21,0,9,9,9,0],
                [21,0,0,0,0,0],
                [21,0,0,21,0,21],
                [21,21,21,21,21,21],
                [21,21,21,21,21,21],
            ], dtype=float)
        
        expected_heuristic_3 = np.full(shape=(8,6), fill_value=43.0, dtype=float)
        expected_heuristic_3[2,0] = 0
        expected_heuristic_3[1,1] = 0
        expected_heuristic_3[2,2] = 0
        expected_heuristic_3[3,1] = 0
        expected_heuristic_3[2,1] = 1
        
        expected_heuristics = [expected_heuristic_3, expected_heuristic_1, expected_heuristic_2]
        examples_states = FloodFill.generate_examples()
        
        for i in range(len(examples_states)):
            current_state = examples_states[i]
            current_expected_heuristic = expected_heuristics[i]
            
            print("====================================================================================================")
            print(f"Calculating the floodfill heuristic for the following state:\n{current_state}")
            
            print(f"\nExpecting the following heuristic:\n{current_expected_heuristic}")
            current_calculated_heuristic = FloodFill.calc(current_state)
            print(f"\nCalculated heuristic:\n{current_calculated_heuristic}")
            
            assert(np.array_equal(current_expected_heuristic, current_calculated_heuristic))

    @staticmethod
    def test_extended_average():
        print("\nTesting Extended Average FloodFill!\n")
    

        examples_states = FloodFill.generate_examples()
        
        for i in range(len(examples_states)):
            current_state = examples_states[i]
            
            print("====================================================================================================")
            print(f"Calculating the floodfill heuristic for the following state:\n{current_state}")
            
            current_calculated_heuristic = FloodFill.calc_extended_average(current_state)
            print(f"\nCalculated heuristic:\n{current_calculated_heuristic}")

    @staticmethod
    def test_extended_worst_case():
        print("\nTesting Extended Worst Case FloodFill!\n")
    

        examples_states = FloodFill.generate_examples()
        
        for i in range(len(examples_states)):
            current_state = examples_states[i]
            
            print("====================================================================================================")
            print(f"Calculating the floodfill heuristic for the following state:\n{current_state}")
            
            current_calculated_heuristic = FloodFill.calc_extended_worst_case(current_state)
            print(f"\nCalculated heuristic:\n{current_calculated_heuristic}")
            
    @staticmethod
    def test_trap_heuristic():
        print("\nTesting Trap Heuristic!\n")
    

        examples_states = FloodFill.generate_examples()
        
        for i in range(len(examples_states)):
            current_state = examples_states[i]
            
            print("====================================================================================================")
            print(f"Calculating the Trap Heuristic for the following state:\n{current_state}")
            
            current_calculated_heuristic = FloodFill.calc_trap_heuristic(current_state)
            print(f"\nCalculated heuristic:\n{current_calculated_heuristic}")