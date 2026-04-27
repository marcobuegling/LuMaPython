import math
import sys
import timeit
import random
from typing import List, Dict, Optional, Tuple

import numpy as np
from agents.KILabAgentGroup7.CloseCombat import CloseCombat
from environment.Battlesnake.agents.BaseAgent import BaseAgent
from environment.Battlesnake.helper.DirectionUtil import DirectionUtil
from environment.Battlesnake.model.board_state import BoardState
from environment.Battlesnake.model.Direction import Direction
from environment.Battlesnake.model.GameInfo import GameInfo
from environment.Battlesnake.model.grid_map import GridMap
from environment.Battlesnake.model.MoveResult import MoveResult
from environment.Battlesnake.model.Occupant import Occupant
from environment.Battlesnake.model.Position import Position
from environment.Battlesnake.model.Snake import Snake
from environment.Battlesnake.util.kl_priority_queue import KLPriorityQueue

from .LocalGameState import LocalGameState

class V2Agent(BaseAgent):
    # If a connected area is not at least this large, the snake will not consider going there
    MIN_SIZE_FREE_AREA = 10

    def get_name(self):
        return "Luma V2"

    def get_head(self):
        return "lantern-fish"

    def get_tail(self):
        return "swoop"

    def start(self, game_info: GameInfo, turn: int, board: BoardState, you: Snake):
        # Initialise local game state for game
        self.local_game_states[game_info.id] = LocalGameState(board=board, player_snake=you, view_radius=game_info.ruleset_settings.viewRadius)
    
    def __init__(self):
        # Dictionary for mapping game ids to corresponding local game states
        self.local_game_states = dict()
        np.set_printoptions(precision=0, linewidth=150)

    def move(
        self, game_info: GameInfo, turn: int, board: BoardState, you: Snake
    ) -> MoveResult:
        grid_map = board.generate_grid_map()
        
        #print(self.local_game_states)
        # Update local game state
        self.local_game_states[game_info.id].update(board, you)

        current_local_game_state = self.local_game_states[game_info.id]

        # in case there are enemies in attack range close combat begins
        fighting_player_ids = current_local_game_state.get_fighting_player_ids(attack_range=CloseCombat.CLOSE_COMBAT_RANGE)
        if(len(fighting_player_ids) > 1):
            action = CloseCombat.calculate_actions_iterative_deeping(local_game_state=current_local_game_state,
                                                                     fighting_player_ids=fighting_player_ids,
                                                                     time_in_millis=400)[0]
            
            return MoveResult(direction=action)

        # Calculate heuristic
        heuristic = current_local_game_state.calc_overall_heuristic()

        #print("Overall heuristic", heuristic)

        # Get top 10 fields (sorted)
        flat_indices = np.argpartition(heuristic.flatten(), -10)[-10:]
        top_10_indices = np.unravel_index(flat_indices, heuristic.shape)
        indices_list = list(zip(*top_10_indices))
        sorted_indices = sorted(indices_list, key=lambda x: -heuristic[x])

        # Get size of free areas on board
        occupation_array = self.get_blocked_fields_array(board)
        connected_components = self.measure_connected_components(occupation_array).T
        #print(connected_components)

        # Try to get path to best field possible
        for index in sorted_indices:
            if connected_components[index[1], index[0]] < self.MIN_SIZE_FREE_AREA:
                #print("Won't go there, too little space.")
                continue
            path = self.a_star_search(you.get_head(), Position(index[1], index[0]), board, grid_map)
            if path and len(path[1]) > 0:
                #print("Heading for field", Position(index[1], index[0]))
                #print("Direction:", path[1][0][1])
                return MoveResult(direction=path[1][0][1])
            else:
                #print("Path to", Position(index[1], index[0]), "not found")
                pass
            
        #print("Do random action")
        # Do random action of nothing else works
        return MoveResult(direction=self.random_action(you, board))

    def end(self, game_info: GameInfo, turn: int, board: BoardState, you: Snake):
        pass

    def get_filtered_actions(self, snake: Snake, board: BoardState) -> List[Direction]:
        possible_actions = snake.possible_actions()
        head = snake.get_head()

        possible_actions_filtered = []
        for action in possible_actions:
            neighbor_position = head.advanced(action)

            if board.is_out_of_bounds(neighbor_position) or board.is_occupied_by_snake(neighbor_position):
                continue

            possible_actions_filtered.append(action)

        return possible_actions_filtered
    
    def random_action(self, snake: Snake, board: BoardState) -> Direction:
        possible_actions_filtered = self.get_filtered_actions(snake, board)

        if len(possible_actions_filtered) == 0:
            return None
        return np.random.choice(possible_actions_filtered)
        
    @staticmethod
    def get_blocked_fields_array(board) -> np.ndarray:
        arr = np.zeros((board.height, board.width))
        for x in range(board.height):
            for y in range(board.width):
                pos = Position(x, y)
                if (board.is_occupied_by_snake(pos)):
                    arr[x,y] = 1
        return arr
    
    @staticmethod
    def measure_connected_components(arr):
        result = np.zeros(arr.shape)
        while 0 in arr[:,:]:
            first_free = np.unravel_index(np.argmax(arr==0), arr.shape)
            to_expand = [first_free]
            index_list = []
            while len(to_expand) > 0:
                x = to_expand[0][0]
                y = to_expand[0][1]
                arr[x, y] = 1
                if (x - 1) >= 0 and arr[x - 1, y] == 0 and not (x - 1, y) in to_expand:
                    to_expand.append((x - 1, y))
                if (y - 1) >= 0 and arr[x, y - 1] == 0 and not (x, y - 1) in to_expand:
                    to_expand.append((x, y - 1))
                if (x + 1) < arr.shape[0] and arr[x + 1, y] == 0 and not (x + 1, y) in to_expand:
                    to_expand.append((x + 1, y))
                if (y + 1) < arr.shape[1] and arr[x, y + 1] == 0 and not (x, y + 1) in to_expand:
                    to_expand.append((x, y + 1))
                if not to_expand[0] in index_list:
                    index_list.append(to_expand[0])
                to_expand.pop(0)
            size = len(index_list)
            for index in index_list:
                result[index[0], index[1]] = size
        return result

    @staticmethod
    def a_star_search(
        start_field: Position,
        search_field: Position,
        board: BoardState,
        grid_map: GridMap,
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
                if (board.is_out_of_bounds(neighbour_pos) or board.is_occupied_by_snake(neighbour_pos)): continue
                
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