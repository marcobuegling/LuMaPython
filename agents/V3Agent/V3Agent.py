import math
import sys
import timeit
import random
from typing import List, Dict, Optional, Tuple

import numpy as np
from agents.V3Agent.FloodFill import FloodFill
from agents.V3Agent.CloseCombat import CloseCombat
from agents.V3Agent.AStar import astar_search, astar_search_longest_path
from agents.V3Agent.Util import Util
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

class V3Agent(BaseAgent):
    # If a connected area is not at least this large, the snake will not consider going there
    MIN_SIZE_FREE_AREA = 40
    RELATIVE_MIN_SIZE_FREE_AREA = 0.6

    def get_name(self):
        return "Luma Python"

    def get_head(self):
        return "lantern-fish"

    def get_tail(self):
        return "swoop"
    
    def get_color(self) -> Tuple | None:
        return (128, 0, 128)

    def start(self, game_info: GameInfo, turn: int, board: BoardState, you: Snake):
        # Initialise local game state for game
        self.local_game_states[game_info.id] = LocalGameState(board=board, player_snake=you, view_radius=game_info.ruleset_settings.viewRadius)
        
        # compile functions
        CloseCombat.compile(self.local_game_states[game_info.id].width, self.local_game_states[game_info.id].height)
        FloodFill.calc_extended_worst_case(np.array([[1],[0]]))
    
    def __init__(self):
        # Dictionary for mapping game ids to corresponding local game states
        self.local_game_states = dict()
        np.set_printoptions(precision=2, linewidth=150)

    def move(
        self, game_info: GameInfo, turn: int, board: BoardState, you: Snake
    ) -> MoveResult:

        #print("Turn", board.turn)
        
        # Update local game state
        self.local_game_states[game_info.id].update(board, you)

        current_local_game_state: LocalGameState = self.local_game_states[game_info.id]
        
        #print(Util.to_string(current_local_game_state.create_estimated_board_state())+"\n")

        # in case there are enemies in attack range close combat begins
        fighting_player_ids = current_local_game_state.get_fighting_player_ids(attack_range=CloseCombat.CLOSE_COMBAT_RANGE)
        if(len(fighting_player_ids) > 1):
            action = CloseCombat.calculate_actions_iterative_deeping(local_game_state=current_local_game_state,
                                                                     fighting_player_ids=fighting_player_ids,
                                                                     time_in_millis=400)[0]
            
            return MoveResult(direction=action)
        
        # Get free/blocked fields encoded in a binary array
        occupation_array = self.get_blocked_fields_array(board, you)
        #print(occupation_array)

        # Calculate heuristic
        heuristic = current_local_game_state.calc_overall_heuristic().T

        #print("Overall heuristic\n", heuristic)

        # Get top 20 fields (sorted)
        flat_indices = np.argpartition(heuristic.flatten(), -20)[-20:]
        top_indices = np.unravel_index(flat_indices, heuristic.shape)
        indices_list = list(zip(*top_indices))
        sorted_indices = sorted(indices_list, key=lambda x: -heuristic[x])
        #print("Best fields:", sorted_indices)

        # Get size of free areas on board
        connected_components = FloodFill.calc(occupation_array)
        #print(connected_components)

        # Check if we are in serious danger (don't consider heuristic in this case, but try to survive)
        if self.get_own_area_size(connected_components, you) >= min(you.get_length() * self.RELATIVE_MIN_SIZE_FREE_AREA, self.MIN_SIZE_FREE_AREA):
            # Try to get path to best field possible
            for index in sorted_indices:
                #print("Space at", index, ":", connected_components[index[0], index[1]])
                if connected_components[index[0], index[1]] < min(you.get_length() * self.RELATIVE_MIN_SIZE_FREE_AREA, self.MIN_SIZE_FREE_AREA):
                    #print("Won't go there, too little space.")
                    continue
                path = astar_search(occupation_array, you.get_head().to_tuple(), (index[0], index[1]))
                #print("Path:", path)
                if path and len(path) > 1:
                    #print("Heading for field", Position(index[0], index[1]))
                    direction = self.direction_from_coordinates(path[0][0], path[0][1], path[1][0], path[1][1])
                    #print("Direction:", direction)
                    if not board.is_occupied_by_snake(you.get_head().advanced(direction)) or (turn > 2 and you.get_head().advanced(direction) == you.get_tail()):
                        print("Direction:", direction)
                        return MoveResult(direction=direction)
                else:
                    #print("Path to", Position(index[1], index[0]), "not available")
                    pass

        # No path to good field was found -> We're trapped!
        escape_path = self.get_best_escape(occupation_array, you)
        if escape_path:
            #print("escape path:", escape_path)
            direction = self.direction_from_coordinates(escape_path[0][0], escape_path[0][1], escape_path[1][0], escape_path[1][1])
            print("Direction:", direction)
            return MoveResult(direction=direction)
        
        escape_path = self.get_best_enemy_escape(occupation_array, board, you)
        if escape_path:
            #print("escape path:", escape_path)
            direction = self.direction_from_coordinates(escape_path[0][0], escape_path[0][1], escape_path[1][0], escape_path[1][1])
            print("Direction:", direction)
            return MoveResult(direction=direction)
        
        path = self.stayin_alive(occupation_array, you)
        if path:
            direction = self.direction_from_coordinates(path[0][0], path[0][1], path[1][0], path[1][1])
            print("Direction:", direction)
            return MoveResult(direction=direction)
            
        # Do random action of nothing else works
        print("Do random action")
        print("Direction:", direction)
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
    def get_blocked_fields_array(board: BoardState, snake: Snake) -> np.ndarray:
        arr = Util.get_blocked_fields_array(board=board)
        for p in snake.body:
            arr[p.x, p.y] = 0
        # Ignore all parts that won't be reached in time anyway
        #print("Snake body:", snake.body)
        for i in range(len(snake.body)):
            path = astar_search(arr, snake.get_head().to_tuple(), snake.body[i].to_tuple())
            if path and (len(snake.body) - i) < len(path):
                #print("Ignore body part at", snake.body[i], "because it's the", i, "last body part. Path:", path)
                arr[snake.body[i].x, snake.body[i].y] = 0
            else:
                arr[snake.body[i].x, snake.body[i].y] = 1
        return arr
    
    @staticmethod
    def measure_connected_components(arr): # a. k. a. floodfill
        result = np.zeros(arr.shape)
        a = arr.copy()
        while 0 in a[:,:]:
            first_free = np.unravel_index(np.argmax(a==0), a.shape)
            to_expand = [first_free]
            index_list = []
            while len(to_expand) > 0:
                x = to_expand[0][0]
                y = to_expand[0][1]
                a[x, y] = 1
                if (x - 1) >= 0 and a[x - 1, y] == 0 and not (x - 1, y) in to_expand:
                    to_expand.append((x - 1, y))
                if (y - 1) >= 0 and a[x, y - 1] == 0 and not (x, y - 1) in to_expand:
                    to_expand.append((x, y - 1))
                if (x + 1) < a.shape[0] and a[x + 1, y] == 0 and not (x + 1, y) in to_expand:
                    to_expand.append((x + 1, y))
                if (y + 1) < a.shape[1] and a[x, y + 1] == 0 and not (x, y + 1) in to_expand:
                    to_expand.append((x, y + 1))
                if not to_expand[0] in index_list:
                    index_list.append(to_expand[0])
                to_expand.pop(0)
            size = len(index_list)
            for index in index_list:
                result[index[0], index[1]] = size
        return result
    
    # Returns best escape option in the form of a path
    @staticmethod
    def get_best_escape(occupation_array: np.array, snake: Snake) -> List[Tuple[int, int]]|None:
        head = snake.get_head()
        occupation_array[head.x, head.y] = 0
        possible_escapes = []
        for i in range(len(snake.body)):
            # check if there is a connection to the head
            if astar_search(occupation_array, (head.x, head.y), snake.body[len(snake.body) - 1 - i].to_tuple()):
                # get longest path possible to the field
                path = astar_search_longest_path(occupation_array, (head.x, head.y), snake.body[len(snake.body) - 1 - i].to_tuple())
                length = len(path) - 1
                # check if it would sufficient to stay alive
                if length > i:
                    possible_escapes.append((path, length))
        # return the option with the shortest path - maybe the opposite is better?
        if len(possible_escapes) > 0:
            #print("Possible escapes:", possible_escapes)
            possible_escapes.sort(key=lambda x: x[1])
            return possible_escapes[0][0]
        else:
            return None
        
    @staticmethod
    def get_best_enemy_escape(occupation_array: np.array, board: BoardState, you: Snake) -> List[Tuple[int, int]]|None:
        head = you.get_head()
        occupation_array[head.x, head.y] = 0
        possible_escapes = []
        for snake in board.snakes:
            if (you is None or snake.snake_id != you.snake_id) and snake.get_tail() and snake.get_tail().x != -1 and snake.get_tail().y != -1:
                snake_reversed = list(reversed(snake.body))
                for i in range(len(snake_reversed)):
                    if snake_reversed[len(snake_reversed) - 1 - i].x != -1 and snake_reversed[len(snake_reversed) - 1 - i].y != -1:
                        if astar_search(occupation_array, (head.x, head.y), snake_reversed[len(snake_reversed) - 1 - i].to_tuple()):
                            # get longest path possible to the field
                            path = astar_search_longest_path(occupation_array, (head.x, head.y), snake_reversed[len(snake_reversed) - 1 - i].to_tuple())
                            length = len(path) - 1
                            # check if it would sufficient to stay alive
                            if length > i:
                                possible_escapes.append((path, length))
                    # return the option with the shortest path - maybe the opposite is better?
                    if len(possible_escapes) > 0:
                        #print("Possible escapes:", possible_escapes)
                        possible_escapes.sort(key=lambda x: x[1])
                        return possible_escapes[0][0]
                    else:
                        return None
        
    @staticmethod
    # Ah, ha, ha, ha,...
    def stayin_alive(occupation_array: np.array, you: Snake):
        blocked_fields = np.argwhere(occupation_array==1)
        print(blocked_fields)
        paths = []
        head_pos = you.get_head().to_tuple()
        for index in blocked_fields:
            index_as_tuple = (index[0], index[1])
            path = astar_search_longest_path(occupation_array, head_pos, index_as_tuple)
            if path:
                print("path to", index_as_tuple, "found. Length:", len(path))
                paths.append((len(path), path))
            else:
                print("path to", index_as_tuple, "not found.")
        paths.sort(key=lambda x: x[0], reverse=True)
        path = paths[0][1]
        return path
    
    @staticmethod
    def get_own_area_size(connected_components_size_array: np.array, you: Snake) -> int:
        head_position = you.get_head().to_tuple()
        area_sizes = []
        if head_position[0] + 1 < connected_components_size_array.shape[0]:
            value = connected_components_size_array[head_position[0] + 1, head_position[1]]
            if value != 0:
                area_sizes.append(value)
        if head_position[0] - 1 >= 0:
            value = connected_components_size_array[head_position[0] - 1, head_position[1]]
            if value != 0:
                area_sizes.append(value)
        if head_position[1] + 1 < connected_components_size_array.shape[1]:
            value = connected_components_size_array[head_position[0], head_position[1] + 1]
            if value != 0:
                area_sizes.append(value)
        if head_position[1] - 1 >= 0:
            value = connected_components_size_array[head_position[0], head_position[1] - 1]
            if value != 0:
                area_sizes.append(value)
        if len(area_sizes) == 0: return 0
        return max(area_sizes)
    
    @staticmethod
    def direction_from_coordinates(x1: int, y1: int, x2: int, y2: int) -> Direction|None:
        delta_x = x2 - x1
        delta_y = y2 - y1
        if delta_x == -1 and delta_y == 0:
            return Direction.LEFT
        if delta_x == 0 and delta_y == -1:
            return Direction.DOWN
        if delta_x == 1 and delta_y == 0:
            return Direction.RIGHT
        if delta_x == 0 and delta_y == 1:
            return Direction.UP
        return None

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