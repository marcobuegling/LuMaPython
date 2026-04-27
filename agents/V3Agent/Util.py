import math
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
from numba import njit
from environment.Battlesnake.model.Direction import Direction
from environment.Battlesnake.model.Position import Position
from environment.Battlesnake.model.RulesetSettings import RulesetSettings
from environment.Battlesnake.model.Snake import Snake
from environment.Battlesnake.model.board_state import BoardState
from environment.Battlesnake.util.kl_priority_queue import KLPriorityQueue


class Util:
    
    '''Dummy rule settings used to simulate very basic games in which only the snake movements matter.'''
    DUMMY_RULE_SETTINGS:RulesetSettings = RulesetSettings(foodSpawnChance=0, minimumFood=0, hazardDamagePerTurn=0, royale_shrinkEveryNTurns=0, 
                                       squad_allowBodyCollisions=False, squad_sharedElimination=False, squad_sharedHealth=False,
                                       squad_sharedLength=False, viewRadius=-1)
    # TODO: check if royale_shrinkEveryNTurns=0 causes bugs, maybe it has to be -1
    
    @staticmethod
    def a_star_search(
        start_field: Position,
        search_field: Position,
        board: BoardState,
        blocked_positions:List[Position]
    ) -> Tuple[int, List[Tuple[Position, Direction]]]:
        queue = KLPriorityQueue()
        came_from = {}
        cost_so_far = {}
        
        queue.put(0, start_field)
        
        explored_positions = []
        
        cost_so_far[start_field] = 0
        came_from[start_field] = (None, None)
        
        while(not queue.empty()):
            
            explore_pos: Position = queue.get()
            explored_positions.append(explore_pos)
            
            # end of search
            if(explore_pos == search_field):
                # the path is calculated afterwards
                break
                
            for dir in Direction: 
                neighbour_pos = explore_pos.advanced(dir)
                
                # ignore unreachable
                if (board.is_out_of_bounds(neighbour_pos) or neighbour_pos in blocked_positions): continue
                
                goal_pos_tuple = (search_field.x, search_field.y)
                neighbour_tuple = (neighbour_pos.x, neighbour_pos.y)
                
                new_cost = cost_so_far[explore_pos] + 1
                
                # add path if this is the first time this position was visited or update path if the costs are smaller
                # notice that the check for the existance of the position in the dict is necessary to avoid a missing key error
                if( not neighbour_pos in cost_so_far or 
                    (neighbour_pos in cost_so_far and new_cost < cost_so_far[neighbour_pos])):
                
                    cost_so_far[neighbour_pos] = new_cost
                    came_from[neighbour_pos] = (explore_pos, dir)
                
                    # don't re-explore already explored positions
                    if(neighbour_pos not in explored_positions):
                        heuristic_value = math.dist(goal_pos_tuple, neighbour_tuple)
                        queue.put(cost_so_far[neighbour_pos] + heuristic_value, neighbour_pos)
                
        # no path possible
        if(explore_pos != search_field):
            return None
        
        # calculate the final cost and path
        cost = cost_so_far[search_field]
        path: List[Tuple[Position, Direction]] = []
                
        # build the path from the dictionary
        temp_pos, temp_dir = came_from[search_field]
        while(temp_pos is not None):
            path.append((temp_pos, temp_dir))
            temp_pos, temp_dir = came_from[temp_pos]

        # the path was contructed backwards
        path.reverse()
        return cost, path
    
    def dist(pos1:Position, pos2:Position) -> float:
        deltaX = abs(pos1.x-pos2.x)
        deltaY = abs(pos1.y-pos2.y)
        return deltaX+deltaY
    
    def copy_snake(snake:Snake) -> Snake:
        '''Copies the given snake and returns the copy. 
        The underlying body list is a new list as well.'''
        
        new_body = snake.body.copy()
        
        new_snake = Snake(
            snake_id=snake.snake_id,
            snake_name=snake.snake_name,
            snake_color=snake.snake_color,
            snake_head_icon=snake.snake_head_icon,
            snake_tail_icon=snake.snake_tail_icon,
            health=snake.health,
            body = new_body,
            latency=snake.latency,
            shout=snake.shout,
            squad=snake.squad
        )
        
        # copy over elimination events
        if(snake.elimination_event):
            new_snake.elimination_event = snake.elimination_event
        
        return new_snake
        
    def current_millis_time():
        return round(time.time() * 1000)

    @staticmethod
    def eprint(*args, **kwargs):
        print(*args, file=sys.stderr, **kwargs)
        
    
    
    @staticmethod
    def to_string(board_state:BoardState):
        
        # draw empty squares on board as dots
        str_data = np.array([[" ·"] * board_state.width]*board_state.height)
            
        # draw all living enemy snakes
        for snake_id in range(len(board_state.snakes)):
            snake = board_state.snakes[snake_id]
            
            if(not snake.is_alive()): continue
            
            for body_part_id in range(len(snake.body)):
                position = snake.body[body_part_id]
                
                str_data[board_state.height - 1 - position.y][position.x] = "h" if body_part_id == 0 else str(snake_id)
        
        return np.array2string(str_data, suffix="", separator="\t").replace("'", "")
    
    
    @staticmethod
    @njit
    def normalize_heuristic(heuristic:np.ndarray) -> np.ndarray:
        
        # make all entries positive
        heuristic:np.ndarray = heuristic + abs(heuristic.min())
        
        # norm entries
        max_val = heuristic.max()
        if(max_val!=0):
            heuristic:np.ndarray = heuristic / max_val
        
        return heuristic
    
    
    @staticmethod
    def get_blocked_fields_array(board: BoardState) -> np.ndarray:
        arr = np.zeros((board.height, board.width))
        for x in range(board.height):
            for y in range(board.width):
                pos = Position(x, y)
                if (board.is_occupied_by_snake(pos)):
                    arr[x,y] = 1
                    
        return arr