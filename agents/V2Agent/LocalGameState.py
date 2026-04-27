import math
import numpy as np
import random
from typing import List, Dict
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
    FOOD_BASE_WEIGHT = 200.0
    FOOD_DISTANCE_DISCOUNT = 0.5
    FOOD_MAX_DISTANCE = 2
    FOOD_TURN_DISCOUNT = 0.1
    FOOD_MAX_TURNS = 10
    ENEMY_HEAD_WEIGHT = 1.0
    ENEMY_HEAD_DIST_DISCOUNT = ENEMY_BODY_DIST_DISCOUNT = 0.5
    ENEMY_HEAD_MAX_DIST = ENEMY_BODY_MAX_DIST = 2
    ENEMY_HEAD_TURN_DISCOUNT = ENEMY_BODY_TURN_DISCOUNT = 0.2
    ENEMY_HEAD_MAX_TURNS = ENEMY_BODY_MAX_TURNS = 5
    ENEMY_BODY_WEIGHT = 0.5
    DISTANCE_HEURISTIC_FACTOR = 1.0

    def __init__(self, board: BoardState, player_snake: Snake, view_radius: int, copy_enemy_bodies:bool=False):
        self.height = board.height
        self.width = board.width
        self.player_snake = player_snake
        self.view_radius = view_radius

        # Base heuristic (assumption for now: doesn't change over time)
        self.base_heuristic = self.init_base_heuristic()

        # Dictionary of foods (keys: positions, values: integers indicating how many turns ago the foods have last been seen)
        self.foods: Dict[Position, int] = {}

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
        #print("Own position:", snake.get_head())
        # update player snake
        self.player_snake = snake

        # update enemy positions and delete dead snakes
        for snake_id, snake in list(self.enemies.items()):
            #print("Updating snake", snake_id)
            snake_board_state = board.get_snake_by_id(snake_id=snake_id)
            if not snake_board_state:
                #print("Snake", snake_id, "dead")
                del self.enemies[snake_id]
                continue

            positions = snake_board_state.body

            filtered_positions = []
            for pos in positions:
                if pos.x != -1 and pos.y != -1:
                    #print("Regarding position", pos)
                    if board.get_snake_by_id(snake_id).get_head() == pos:
                        filtered_positions.append((pos, 0))
                    elif board.get_snake_by_id(snake_id).get_tail() == pos:
                        filtered_positions.append((pos, self.ENEMY_HEAD_MAX_TURNS))
                    else:
                        filtered_positions.append((pos, 1))
            #print("Filtered positions", filtered_positions)
            snake.update(filtered_positions, self.player_snake.get_head(), self.ENEMY_HEAD_MAX_TURNS, self.view_radius)

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
                    closest_snake = None
                    min_dist = self.width + self.height

                    # give food to the closest enemy snake
                    for snake_id in self.enemies.keys():
                        if len(self.enemies[snake_id].positions) == 0:
                            continue
                        dist = self.city_block_distance(pos.x, pos.y, self.enemies[snake_id].positions[0][0].x, self.enemies[snake_id].positions[0][0].y)
                        if dist < min_dist:
                            min_dist = dist
                            closest_snake = snake_id
                    # choose random snake if closest can't be determined
                    if not closest_snake:
                        closest_snake = random.choice(list(self.enemies.keys()))
                    #print("Food added to", closest_snake)
                    self.enemies[closest_snake].increment_length(1)
                    del self.foods[pos]

        # add new foods and reset turns of foods in view
        for food in board.food:
            #print("Adding food", food.x, food.y)
            self.foods[Position(food.x, food.y)] = 0

        #print("Foods after update:", self.foods)

    def get_blocked_fields(self, snake_id: str) -> List[Position]:
        # TODO the blocked positions have to depend on the snake, since bodyparts of other snakes block the snake, 
        # but the own body of the snake itself does not block it
        pass

    def init_base_heuristic(self) -> np.ndarray:
        heuristic = np.zeros((self.height, self.width), dtype=float)

        x, y = np.meshgrid(np.arange(self.width), np.arange(self.height))
        center_x, center_y = self.width / 2, self.height / 2
        distance_array = np.abs(x + 0.5 - center_x) + np.abs(y + 0.5 - center_y)

        value_array = 20 - 0.25 * distance_array ** 2
        heuristic[:,:] = value_array

        return self.GENERAL_HEURISTIC_FACTOR * heuristic
    
    def calc_food_heuristic(self) -> np.ndarray:
        heuristic = np.zeros((self.height, self.width), dtype=float)

        for pos, turn in self.foods.items():
            base_value = self.FOOD_BASE_WEIGHT * (1 - self.FOOD_TURN_DISCOUNT * turn)
            x, y = np.meshgrid(np.arange(self.width), np.arange(self.height))
            distance_array = np.abs(x - pos.x) + np.abs(y - pos.y)
            value_array = base_value * (1 - self.FOOD_DISTANCE_DISCOUNT * distance_array)
            value_array[value_array < 0] = 0
            heuristic += value_array
        #print("Food heuristic", heuristic)
        return heuristic

    def calc_enemy_heuristic(self) -> np.ndarray:
        heuristic = np.zeros((self.height, self.width), dtype=float)

        #left out for now, effect too small

        return heuristic

    def calc_distance_heuristic(self) -> np.ndarray:
        heuristic = np.zeros((self.height, self.width), dtype=float)

        x, y = np.meshgrid(np.arange(self.width), np.arange(self.height))
        distance_array = np.abs(x - self.player_snake.get_head().x) + np.abs(y - self.player_snake.get_head().y)
        value_array = distance_array * (-10)

        heuristic[:,:] = value_array
        #print("Distance heuristic", heuristic)
        return self.DISTANCE_HEURISTIC_FACTOR * heuristic

    def calc_overall_heuristic(self) -> np.ndarray:
        # add up all heuristics
        return self.base_heuristic + self.calc_food_heuristic() + self.calc_enemy_heuristic() + self.calc_distance_heuristic()
    
    
    @staticmethod
    def city_block_distance(x1: int, y1: int, x2: int, y2: int) -> int:
        return abs(x1 - x2) + abs(y1 - y2)
    
    
    #######################################################################################################################################################
    ## close combat stuff
    
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
        
        # cannot build snake bodies, that overlap with other snake bodies
        already_used_positions = []
        
        for enemy_snake_id in self.enemies:
            enemy_snake = self.enemies[enemy_snake_id]
            blocked_positions = view_radius_positions + self.get_blocked_positions(enemy_snake.id) + already_used_positions
            
            estimated_snake = enemy_snake.estimate_snake(board_height=self.height, board_width=self.width, blocked_positions=blocked_positions)
            estimated_snake.snake_id = enemy_snake.id
            
            # ignore empty snakes we know nothing about
            if(len(estimated_snake.body) > 0):  
                snakes.append(estimated_snake)
                already_used_positions.extend(estimated_snake.body)
        
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
                                
    