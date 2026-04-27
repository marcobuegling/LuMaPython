import math
import sys
import timeit
import random
from typing import List, Optional, Tuple

import numpy as np
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


class V1Agent(BaseAgent):
    def get_name(self):
        return "V1 Luma"

    def get_head(self):
        return "sneaky"

    def get_tail(self):
        return "swoop"

    def start(self, game_info: GameInfo, turn: int, board: BoardState, you: Snake):
        pass
    
    def __init__(self):
        self.ESCAPE_DIST = 3
        self.CHASE_DIST = 4
        self.DANGER_SIZE_FACTOR = 3
        self.KILL_SIZE_FACTOR = 3
        self.food_spots:set = set()

    def move(
        self, game_info: GameInfo, turn: int, board: BoardState, you: Snake
    ) -> MoveResult:
        start_time = timeit.default_timer()
        grid_map: GridMap[Occupant] = board.generate_grid_map()
        
        # remove food spots if the snake already went on it
        for position in you.body:
            if(position in self.food_spots):
                self.food_spots.remove(position)

        # 1. order of business: run away if necessary (longer or equally long snake closer than three moves) (and try to kill snake if possible)
        enemy_snakes = self.get_known_enemy_snakes(you, board)
        #if len(enemy_snakes) > 0:
            #print(you.snake_id, ": Spottet enemy!")
        for enemy in enemy_snakes:
            if enemy.get_length() * self.DANGER_SIZE_FACTOR >= you.get_length():
                path_to_enemy = self.get_path_to_enemy(you, enemy, board, grid_map)
                if path_to_enemy and path_to_enemy[0] <= self.ESCAPE_DIST:
                    #print(you.snake_id, ": Trying to run away because dangerous snake too close.")
                    
                    # try to get as far away from enemy head as possible
                    head = you.get_head()
                    head_enemy = enemy.get_head()
                    best_dist = -(sys.maxsize)
                    best_action = None
                    filtered_actions = self.get_filtered_actions(you, board)

                    for action in filtered_actions:
                        new_dist = math.dist(head.advanced(action).to_tuple(), head_enemy.to_tuple())
                        if not self.get_path_to_center(head.advanced(action), board, grid_map):
                            new_dist -= 3
                        if new_dist > best_dist:
                            best_dist = new_dist
                            best_action = action
                    
                    if best_action:
                        #stop_time = timeit.default_timer()
                        #print("Time: ", stop_time - start_time)
                        return MoveResult(direction=best_action)
        
                        
        # 2. order of business: try to kill snake if possible
        for enemy in enemy_snakes:
            if enemy.get_length() * self.KILL_SIZE_FACTOR < you.get_length():
                path_to_enemy = self.get_path_to_enemy(you, enemy, board, grid_map)
                if path_to_enemy and path_to_enemy[0] <= self.CHASE_DIST:
                    #print(you.snake_id, ": Going for the kill.")
                    #stop_time = timeit.default_timer()
                    #print("Time: ", stop_time - start_time)
                    return MoveResult(direction=path_to_enemy[1][0][1])


        # 3. order of business: find food
        food_action = self.follow_food(you, board, grid_map)
        if food_action is not None:
            #print(you.snake_id, ": Gonna eat something. *yum*")
            #stop_time = timeit.default_timer()
            #print("Time: ", stop_time - start_time)
            return MoveResult(direction=food_action)
        
        # 4. order of business: scout
        scout_action = self.scout(you, board, grid_map)
        if scout_action is not None:
            #print(you.snake_id, ": Nothing to eat. Let's go back to the center.")
            #stop_time = timeit.default_timer()
            #print("Time: ", stop_time - start_time)
            return MoveResult(direction=scout_action)

        # if all of these plans fail for some reason: random moves
        #print(you.snake_id, ": Snake confused. Snake do random stuff.")
        random_action = self.random_action(you, board)
        
        #stop_time = timeit.default_timer()
        #print("Time: ", stop_time - start_time)
        return MoveResult(direction=random_action)

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
    
    def get_known_enemy_snakes(self, snake: Snake, board: BoardState) -> List[Snake]:
        # Get all enemy snakes that are alive and of which the head position is known
        enemy_snakes = []
        for s in board.snakes:
            if s.snake_id != snake.snake_id and s.is_alive() and s.get_head().to_tuple() != (-1, -1):
                enemy_snakes.append(s)
        return enemy_snakes
    
    def get_path_to_enemy(self, snake: Snake, enemy: Snake, board: BoardState, grid_map: GridMap) -> Tuple[int, List[Tuple[Position, Direction]]]:
        head = snake.get_head()
        head_enemy = enemy.get_head()
        dir_enemy = enemy.get_current_direction()
        if dir_enemy:
            path_to_next_enemy_pos = V1Agent.a_star_search(head, head_enemy.advanced(dir_enemy), board, grid_map)
            if path_to_next_enemy_pos:
                return path_to_next_enemy_pos
            
        possible_paths = []
        for dir in Direction:
            potential_path = V1Agent.a_star_search(head, head_enemy.advanced(dir), board, grid_map)
            if potential_path:
                possible_paths.append(potential_path)

        min_distance = sys.maxsize
        best_path = None

        for p in possible_paths:
            if p[0] < min_distance:
                min_distance = p[0]
                best_path = p

        if best_path:
            return best_path
        
    def get_path_to_center(self, position: Position, board: BoardState, grid_map: GridMap) -> Tuple[int, List[Tuple[Position, Direction]]]:
        center = Position(board.width//2,board.height//2)

        path_result = V1Agent.a_star_search(position, center, board, grid_map)
        
        if not path_result or len(path_result[1]) == 0:
            # try 15 random positions in the center
            random_tuples = set()
            for _ in range(15):
                random_tuples.add((random.randint(5, 10), random.randint(5, 10)))
            for t in random_tuples:
                path_result = V1Agent.a_star_search(position, Position(t[0], t[1]), board, grid_map)
                if path_result and len(path_result[1]) != 0:
                    return path_result
        else:
            return path_result
        
        return None


    def follow_food(self, snake: Snake, board: BoardState, grid_map: GridMap) -> Direction:
        head = snake.get_head()

        # store all the possible food spots for later
        self.food_spots = self.food_spots.union(set(board.food))
        # NOTICE: this does not give the position of each food, even if the food just spawned
        # a lot of food spots remain quite unknown
        
        # find closest food spot and move to it
        min_distance = sys.maxsize
        closest_food_action = None

        for food in self.food_spots:
            path_calc_result = V1Agent.a_star_search(head, food, board, grid_map)
            if path_calc_result:
                distance, path = path_calc_result
            else:
                continue

            # check if there is no path back to the center -> don't follow this food in this case
            if not self.get_path_to_center(food, board, grid_map):
                #print(snake.snake_id, ": Not considering food because there would be no way back.")
                continue
            
            # check if a longer enemy snake is closer to the food -> don't follow this food in this case
            enemy_snake_closer = False
            enemy_snakes = self.get_known_enemy_snakes(snake, board)
            for enemy in enemy_snakes:
                path_to_enemy = V1Agent.a_star_search(enemy.get_head(), food, board, grid_map)
                if path_to_enemy and enemy.get_length() >= snake.get_length() and path_to_enemy[0] <= distance:
                    enemy_snake_closer = True
                    #print(snake.snake_id, ": Backing away from food because dangerous snake too close.")
                    break

            if(distance < min_distance) and not enemy_snake_closer:
                min_distance = distance
                closest_food_action = path[0][1]
                
        return closest_food_action

    def scout(self, snake: Snake, board: BoardState, grid_map: GridMap) -> Direction:
        result = self.get_path_to_center(snake.get_head(), board, grid_map)
        if result:
            return result[1][0][1]
        return None

    def random_action(self, snake: Snake, board: BoardState) -> Direction:
        # select a random action from the possible actions without hitting a wall or a snake
        
        possible_actions_filtered = self.get_filtered_actions(snake, board)

        if len(possible_actions_filtered) == 0:
            return None
        return np.random.choice(possible_actions_filtered)
        

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