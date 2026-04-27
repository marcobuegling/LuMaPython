import math
from queue import Queue
import numpy as np
import random
from typing import Tuple, List, Dict, Set
from agents.KILabAgentGroup7.AStar import astar_search, astar_search_longest_path
from agents.KILabAgentGroup7.FloodFill import FloodFill
from agents.KILabAgentGroup7.EnemyHeuristic import EnemyHeuristic
from agents.KILabAgentGroup7.Util import Util
from environment.Battlesnake.model.board_state import BoardState
from environment.Battlesnake.model.Food import Food
from environment.Battlesnake.model.Hazard import Hazard
from environment.Battlesnake.model.Snake import Snake
from environment.Battlesnake.model.Position import Position
from environment.Battlesnake.model.Direction import Direction
from environment.Battlesnake.model.grid_map import GridMap
from environment.Battlesnake.model.Occupant import Occupant
from environment.Battlesnake.modes.Standard import StandardGame

from .EnemySnake import EnemySnake

class LocalGameState:

    # Constants used to balance the heuristics
    GENERAL_HEURISTIC_FACTOR = 1.0
    FOOD_BASE_WEIGHT = 300.0
    FOOD_DISTANCE_DISCOUNT = 0.5
    FOOD_MAX_DISTANCE = 2
    FOOD_MAX_TURNS = 15
    FOOD_TURN_DISCOUNT = 1 / FOOD_MAX_TURNS
    ENEMY_HEAD_WEIGHT = 1.0
    ENEMY_HEAD_DIST_DISCOUNT = ENEMY_BODY_DIST_DISCOUNT = 0.5
    ENEMY_HEAD_MAX_DIST = ENEMY_BODY_MAX_DIST = 2
    ENEMY_HEAD_TURN_DISCOUNT = ENEMY_BODY_TURN_DISCOUNT = 0.2
    ENEMY_HEAD_MAX_TURNS = ENEMY_BODY_MAX_TURNS = 5
    ENEMY_BODY_WEIGHT = 0.5
    DISTANCE_HEURISTIC_PENALTY = 0
    DISTANCE_HEURISTIC_FACTOR = 0.97
    OPTIMAL_SIZE = 40
    TAIL_MAX_VALUE = 200.0
    ENEMY_H_BODY_DIST_DISCOUNT = 0.1
    ENEMY_H_HEAD_VALUE = -30
    ENEMY_H_BODY_VALUE = -70
    ENEMY_H_INFLUENCE_THRESHOLD = 1
    WORST_CASE_FLOOD_FILL_FACTOR = 0.5
    WORST_CASE_FLOOD_FILL_OFFSET = 5
    TRAP_FACTOR = 0.01
    TRAP_OFFSET = 0
    ENEMY_FOOD_PROXIMITY_THRESHOLD = 10
    EDGE_PENALTY = -20
    MAX_HEALTH = 100

    def __init__(self, board: BoardState, player_snake: Snake, view_radius: int, copy_enemy_bodies:bool=False):
        self.height = board.height
        self.width = board.width
        self.player_snake = player_snake
        self.view_radius = view_radius
        self.turn = 0

        # Base heuristic (assumption for now: doesn't change over time)
        self.base_heuristic = self.init_base_heuristic()

        # Dictionary of foods (keys: positions, values: integers indicating how many turns ago the foods have last been seen)
        self.foods: Dict[Position, int] = {}
        
        # set of food positions of foods that reached the 'FOOD_MAX_TURNS_VALUE'. We assume they have been eaten by another snake
        self.expired_foods: set = set()

        # Dictionary of enemies (keys: ids of snakes, values: EnemySnake objects)
        self.enemies: Dict[str, EnemySnake] = {}
        for snake in board.snakes:
            if player_snake is None or snake.snake_id != self.player_snake.snake_id:
                
                if(copy_enemy_bodies):
                    self.enemies[snake.snake_id] = EnemySnake(id=snake.snake_id, positions=[(position, 0) for position in snake.body])
                else:
                    self.enemies[snake.snake_id] = EnemySnake(length=len(player_snake.body), id=snake.snake_id)

    def update(self, board: BoardState, snake: Snake):
        #print("Updating local game state")

        self.turn += 1

        #print("Own position:", snake.get_head())
        # update player snake
        self.player_snake = snake

        currently_fully_visible_snakes:Set[str] = set()

        # update enemy positions and delete dead snakes
        for snake_id, snake in list(self.enemies.items()):
            #print("Updating snake", snake_id)
            snake_board_state = board.get_snake_by_id(snake_id=snake_id)
            if not snake_board_state:
                #print("Snake", snake_id, "dead")
                del self.enemies[snake_id]
                continue

            positions = snake_board_state.body

            # account for snakes eating in view radius (should be done before updating the snakes)
            if(len(positions) > 0):
                # the head is now on a position a food was earlier
                # it is ok to do this before update, since food cannot be on the same square as the enemy head anyway
                head_position = positions[0] 
                if(head_position in self.foods):
                    del self.foods[head_position]
                    snake.increment_length(1)
                    

            snake.entire_body_visible = True
            filtered_positions = []
            for pos in positions:
                if pos.x != -1 and pos.y != -1:
                    #print("Regarding position", pos)
                    if board.get_snake_by_id(snake_id).get_head() == pos:
                        filtered_positions.append((pos, 0))
                    elif board.get_snake_by_id(snake_id).get_tail() == pos:
                        filtered_positions.append((pos, snake.estimated_length))
                    else:
                        filtered_positions.append((pos, 1))
                
                else: snake.entire_body_visible = False
            #print("Filtered positions", filtered_positions)
            
            previously_estimated_length = snake.estimated_length
            
            snake.update(filtered_positions, self.player_snake.get_head(), self.view_radius)
            
            # the difference in estimated length and actual length is added/substracted to the estimate of the other snakes
            if(snake.entire_body_visible):
                currently_fully_visible_snakes.add(snake_id)
                
                count_non_visible_snakes = len(self.enemies.keys()) - len(currently_fully_visible_snakes)
                size_estimation_difference = len(snake.positions) - previously_estimated_length
                size_correction_value = size_estimation_difference / (max(count_non_visible_snakes,1))
                
                for snake_id, snake in list(self.enemies.items()):
                    if(snake_id not in currently_fully_visible_snakes):
                        snake.decrement_length(size_correction_value)
                

        # estimate length of other snakes based on foods on the map stochastically
        enemy_count = len(self.enemies.keys())
        food_value = 1.0/enemy_count

        # option for game mode with perfect information
        if self.view_radius == -1:
            self.foods.clear()
        else:
            # update turns of foods
            for pos in list(self.foods.keys()):
                #print("Updating food at", pos)
                self.foods[pos] += 1

                # if food in view radius, delete it (it will be added again if it still exists)
                dist_to_player = self.city_block_distance(self.player_snake.get_head().x, self.player_snake.get_head().y, pos.x, pos.y)
                if dist_to_player <= self.view_radius:
                    del self.foods[pos]
                    continue

                # delete obsolete information about food
                if self.foods[pos] > self.FOOD_MAX_TURNS:
                    #print("Deleting because food is too old")

                    # distribute food consumption  uniformly amongst enemies
                    for snake_id in self.enemies.keys():
                        self.enemies[snake_id].increment_length(food_value)
                        
                    del self.foods[pos]
                    self.expired_foods.add(pos)

        # add new foods and reset turns of foods in view
        for food in board.food:
            #print("Adding food", food.x, food.y)
            food_pos = Position(food.x, food.y)
            
            self.foods[food_pos] = 0
            
            if(food_pos in self.expired_foods):
                self.expired_foods.remove(food_pos)
                
                # undo food consumption asumption
                for snake_id in self.enemies.keys():
                    self.enemies[snake_id].decrement_length(food_value)
                    
        # take out all foods in view radius from expired foods
        # case 1: Food present -> food is not expired anymore, because it is present
        # case 2: Food not present -> food was eaten and does not count as expired anymore
        # conclusion: always remove expiredbody foods from view radius
        view_radius_positions = self.calculate_view_radius_positions()
        temp_expired_foods = set()
        for expired_food in self.expired_foods:
            if(expired_food not in view_radius_positions):
                temp_expired_foods.add(expired_food)
        self.expired_foods = temp_expired_foods

        #print("Foods after update:", self.foods)

    def init_base_heuristic(self) -> np.ndarray:
        heuristic = np.zeros((self.height, self.width), dtype=float)

        x, y = np.meshgrid(np.arange(self.width), np.arange(self.height))
        center_x, center_y = self.width / 2, self.height / 2
        distance_array = np.abs(x + 0.5 - center_x) + np.abs(y + 0.5 - center_y)

        value_array = 20 - 0.25 * distance_array ** 2
        heuristic[:,:] = value_array

        return self.GENERAL_HEURISTIC_FACTOR * heuristic
    
    def calc_food_value(self):
        return self.FOOD_BASE_WEIGHT * ((1 - self.player_snake.get_length() / self.OPTIMAL_SIZE) + (1 - self.player_snake.health / self.MAX_HEALTH))
    
    def calc_food_heuristic(self) -> np.ndarray:
        heuristic = np.zeros((self.height, self.width), dtype=float)

        for pos, turn in self.foods.items():
            base_value = self.calc_food_value() * (1 - self.FOOD_TURN_DISCOUNT * turn)

            # modify the base food value based on availability and competition
            # food that another snake will snatch away is less valuable in comparison to food,
            # ... that we are the closest to

            own_distance_to_food = max(Util.dist(self.player_snake.get_head(), pos),1)

            smallest_enemy_distance = self.ENEMY_FOOD_PROXIMITY_THRESHOLD
            for enemy_snake in self.enemies.values():
                if(len(enemy_snake.positions) <= 0): continue

                current_enemy_distance = Util.dist(enemy_snake.positions[0][0], pos)

                if(current_enemy_distance < smallest_enemy_distance):
                    smallest_enemy_distance = current_enemy_distance
            
            base_value *= (smallest_enemy_distance/own_distance_to_food)**2

            x, y = np.meshgrid(np.arange(self.width), np.arange(self.height))
            distance_array = np.abs(x - pos.x) + np.abs(y - pos.y)
            value_array = base_value * (1 - self.FOOD_DISTANCE_DISCOUNT * distance_array)
            value_array[value_array < 0] = 0
            heuristic += value_array
        #print("Food heuristic\n", heuristic)
        return heuristic

    def calc_enemy_heuristic(self) -> np.ndarray:
        heuristic = EnemyHeuristic.calc(player_id=self.player_snake.snake_id, 
                                        board_state=self.create_estimated_board_state(), 
                                        body_dist_discount=self.ENEMY_H_BODY_DIST_DISCOUNT, 
                                        enemy_head_value=self.ENEMY_H_HEAD_VALUE, 
                                        body_value=self.ENEMY_H_BODY_VALUE, 
                                        influence_threshold=self.ENEMY_H_INFLUENCE_THRESHOLD)
        #print("Enemy heuristic\n", print("Overall heuristic\n", np.flip(heuristic, axis=0).astype(np.int32)))
        return heuristic
    
    def calc_worst_case_floodfill_heuristic(self) -> np.ndarray:
        heuristic = FloodFill.calc_extended_worst_case(Util.get_blocked_fields_array(self.create_estimated_board_state()))
        heuristic = self.WORST_CASE_FLOOD_FILL_FACTOR * (heuristic - self.WORST_CASE_FLOOD_FILL_OFFSET)
        #print("FloodFill heuristic:\n", heuristic)
        return heuristic
    
    def calc_trap_heuristic(self) -> np.ndarray:
        heuristic = FloodFill.calc_trap_heuristic(Util.get_blocked_fields_array(self.create_estimated_board_state()))
        heuristic = self.TRAP_FACTOR * (heuristic - self.TRAP_OFFSET)
        return heuristic
    
    def calc_tail_value(self) -> float:
        if len(self.player_snake.body) > self.OPTIMAL_SIZE:
            return self.TAIL_MAX_VALUE
        else:
            return self.TAIL_MAX_VALUE * len(self.player_snake.body) / self.OPTIMAL_SIZE

    def calc_border_penalty(self, heuristic:np.ndarray) -> np.ndarray:
        heuristic = heuristic.copy()
        height, width = heuristic.shape

        for y in range(height):
            heuristic[y,0] += self.EDGE_PENALTY
            heuristic[y, width-1] += self.EDGE_PENALTY

        for x in range(width):
            heuristic[0,x] += self.EDGE_PENALTY
            heuristic[height-1, x] += self.EDGE_PENALTY
        
        return heuristic

    def calc_distance_heuristic(self, heuristic:np.ndarray):

        x, y = np.meshgrid(np.arange(self.width), np.arange(self.height))
        distance_array = np.abs(x - self.player_snake.get_head().x) + np.abs(y - self.player_snake.get_head().y)
        value_array = distance_array * self.DISTANCE_HEURISTIC_PENALTY

        distance_factor_array = np.full(shape=heuristic.shape, fill_value=self.DISTANCE_HEURISTIC_FACTOR)

        #print("Distance heuristic\n", heuristic)
        return heuristic * np.power(distance_factor_array, distance_array) + value_array

    def calc_overall_heuristic(self) -> np.ndarray:
        # add up all heuristics
        overall_heuristic = self.base_heuristic + self.calc_food_heuristic() + self.calc_enemy_heuristic() + self.calc_worst_case_floodfill_heuristic()
        tail_position = self.player_snake.get_tail()
        #print("Tail position:", tail_position)
        #print("Tail value:", self.calc_tail_value())
        overall_heuristic[tail_position.x, tail_position.y] += self.calc_tail_value()
        overall_heuristic = self.calc_border_penalty(overall_heuristic)
        overall_heuristic = self.calc_distance_heuristic(overall_heuristic)

        #print("Overall heuristic\n", np.flip(overall_heuristic, axis=0).astype(np.int32))
        #print("Trap heuristic\n", np.flip(self.calc_trap_heuristic(), axis=0).astype(np.int32))
        return overall_heuristic
    
    # Just for testing purposes, returns static heuristic
    def calc_test_overall_heuristic(self) -> np.ndarray:
        arr = np.zeros((self.height, self.width))
        arr[14, 0] = 100
        arr[0, 14] = 150
        arr[0, 0] = 20
        arr[0, 1] = 10
        arr[1, 0] = 15
        return arr
    
    @staticmethod
    def city_block_distance(x1: int, y1: int, x2: int, y2: int) -> int:
        return abs(x1 - x2) + abs(y1 - y2)
        
    
#######################################################################################################################################################
    ## close Combat Stuff
    
    def calculate_view_radius_positions(self) -> List[Position]:
    
        player_head_position = self.player_snake.body[0]
        
        view_radius_positions = [player_head_position]
        for y in range(self.view_radius*2 + 1):
            for x in range(self.view_radius*2 + 1):
                
                # shift the range to [-radius, +radius]
                delta_y = y-self.view_radius
                delta_x = x-self.view_radius
                
                # do not add head again
                if(delta_y == 0 and delta_x == 0): continue
                
                # bigger then radius in combination
                if(abs(delta_y) + abs(delta_x) > self.view_radius): continue
                
                potentialPosition = Position(player_head_position.x + delta_x, player_head_position.y + delta_y)
                
                # check out of bounds
                if(potentialPosition.x < 0 or potentialPosition.y < 0 or 
                    potentialPosition.x >= self.width or potentialPosition.y >= self.height): continue
                view_radius_positions.append(potentialPosition)
                
        return view_radius_positions
                

        
    def get_blocked_positions(self, target_snake_id:str) -> List[Position]:
        '''Returns all positions that are confirmed to be blocked by another snake. 
        This does not include any body estimation. It also does not include the current view radius.'''
        
        # collect all the snake bodies mapped by id
        snake_bodies = {self.player_snake.snake_id: self.player_snake.body}
        for enemy_snake_id in self.enemies:
            enemy_snake = self.enemies[enemy_snake_id]
            
            snake_bodies[enemy_snake.id] = [enemy_position[0] for enemy_position in enemy_snake.positions]
        
        blocked_by_enemy:List[Position] = []
        
        for current_snake_id in snake_bodies:
            # do not include own body
            if(current_snake_id == target_snake_id): continue
            
            # mark all other snake body positions as blocked
            blocked_by_enemy.extend(snake_bodies[current_snake_id])
            
            
        return blocked_by_enemy

    
    def create_estimated_board_state(self) -> BoardState:
        
        # cannot estimate body in view radius, since it is evident that the body is not there
        view_radius_positions = self.calculate_view_radius_positions()
        
        # build all the snake bodies one by one
        # copy player snake, since the simulation is inplace
        player_copy = Snake(snake_id=self.player_snake.snake_id,
                            snake_name=self.player_snake.snake_name,
                            body=[position for position in self.player_snake.body])
        snakes = [player_copy]
        
        # the snake is now allowed to build snake bodies, that overlap with other snake bodies
        
        for enemy_snake_id in self.enemies:
            enemy_snake = self.enemies[enemy_snake_id]
            blocked_positions = view_radius_positions + self.get_blocked_positions(enemy_snake.id)
            
            estimated_snake = enemy_snake.estimate_snake(board_height=self.height, board_width=self.width, blocked_positions=blocked_positions)
            estimated_snake.snake_id = enemy_snake.id
            
            # ignore empty snakes we know nothing about
            if(len(estimated_snake.body) > 0):  
                snakes.append(estimated_snake)
        
        return BoardState(turn = 0,
                          width = self.width,
                          height = self.height,
                          snakes=snakes)
        
    def get_fighting_player_ids(self, attack_range:int) -> List[str]:
        fighters = []
        
        # no fight without player
        if(self.player_snake is None or not self.player_snake.is_alive()): return fighters
        
        player_head_position = self.player_snake.get_head()
         
        for enemy_snake_id in self.enemies:
            enemy_snake = self.enemies[enemy_snake_id]
            
            # no body part is stored -> therefore the enemy is definitely not in range
            if(len(enemy_snake.positions) == 0): continue
            
            enemy_head = enemy_snake.positions[0]
            
            # first check if the first bodypart is actually the head and then check if that head is in range
            if(enemy_head[1] == 0 and Util.dist(player_head_position, enemy_head[0]) <= attack_range):
                fighters.append(enemy_snake.id)
        
        # only add player if he fights someone
        # Notice: the player is always the first index
        if(len(fighters) != 0):
            fighters = [self.player_snake.snake_id] + fighters 
        
        return fighters
        
    def __str__(self) -> str:
        # draw empty squares on board as dots
        str_data = np.array([[" ·"] * self.width]*self.height)
        
        # draw player snake as p if not dead
        if(self.player_snake is not None):
            for body_part_id in range(len(self.player_snake.body)):
                position = self.player_snake.body[body_part_id]
                str_data[position.y][position.x] = "H" if body_part_id == 0 else "P"
            
        # draw all living enemy snakes
        for enemy_snake_id in self.enemies:
            
            enemy_snake = self.enemies[enemy_snake_id]
            
            for body_part_id in range(len(enemy_snake.positions)):
                position = enemy_snake.positions[body_part_id][0]
                
                # check for out of bounds positions (-1 blocks)
                if(position.x < 0 or position.y < 0 or position.x >= self.width or position.y >= self.height): 
                    continue
                
                str_data[position.y][position.x] = "h" if body_part_id == 0 else str(enemy_snake_id[0])
        
        return np.array2string(str_data, suffix="", separator="\t").replace("'", "")
    
    def __repr__(self):
        return str(self)
    
    def move(self, player_ids_to_directions_dict:Dict[str,Direction]) -> 'LocalGameState':
        player_snake_id = self.player_snake.snake_id
        
        dummy_board_state = self.create_estimated_board_state()
        
        standard_game = StandardGame(ruleset_settings=Util.DUMMY_RULE_SETTINGS)
        standard_game.create_next_board_state(board=dummy_board_state, moves=player_ids_to_directions_dict, only_deterministic=True)
        
        return LocalGameState(dummy_board_state, player_snake=dummy_board_state.get_snake_by_id(player_snake_id),
                              view_radius=self.view_radius, copy_enemy_bodies=True)
        
#######################################################################################################################################################
    # test implementation
    
    
    @staticmethod
    def test_view_radius_positions():
        
        # radius 0
        player_snake = Snake(snake_id="Player", body=[Position(7, 7)])
        dummy_board_state = BoardState(turn=0, width=16, height=16, snakes=[player_snake])
        state1 = LocalGameState(board=dummy_board_state, view_radius=0, player_snake=player_snake)
        seen_poses = [Position(7,7)]
        assert(set(state1.calculate_view_radius_positions()) == set(seen_poses))
        
        # radius 1
        state2 = LocalGameState(board=dummy_board_state, view_radius=1, player_snake=player_snake)
        seen_poses = [Position(7,7), Position(6,7), Position(8,7), Position(7,8), Position(7,6)]
        assert(set(state2.calculate_view_radius_positions()) == set(seen_poses))
        
        # radius 2
        state3 = LocalGameState(board=dummy_board_state, view_radius=2, player_snake=player_snake)
        seen_poses = [Position(7,7), Position(6,7), Position(8,7), Position(7,8), Position(7,6), 
                      Position(6,6), Position(8,8), Position(6,8), Position(8,6),
                      Position(5,7), Position(9,7), Position(7,9), Position(7,5)]
        assert(set(state3.calculate_view_radius_positions()) == set(seen_poses))
        
        # edge on 0
        player_snake = Snake(snake_id="Player", body=[Position(1, 7)])
        dummy_board_state = BoardState(turn=0, width=16, height=16, snakes=[player_snake])
        state3 = LocalGameState(board=dummy_board_state, view_radius=2, player_snake=player_snake)
        seen_poses = [Position(1,7), Position(0,7), Position(2,7), Position(1,8), Position(1,6), 
                      Position(0,6), Position(0,8), Position(2,8), Position(2,6),
                      Position(3,7), Position(1,9), Position(1,5)] # Position(-1,7) is not allowed
        assert(set(state3.calculate_view_radius_positions()) == set(seen_poses))
        
        # corner on max x and y
        player_snake = Snake(snake_id="Player", body=[Position(3, 3)])
        dummy_board_state = BoardState(turn=0, width=4, height=4, snakes=[player_snake])
        state4 = LocalGameState(board=dummy_board_state, view_radius=3, player_snake=player_snake)
        seen_poses = [Position(3,3), Position(2,3), Position(3,2), Position(2,2), Position(1,3), Position(3,1),
                      Position(0,3), Position(3,0), Position(1,2), Position(2,1)]
        assert(set(state4.calculate_view_radius_positions()) == set(seen_poses))