import numpy as np
from agents.KILabAgentGroup7.Util import Util
from environment.Battlesnake.model.Direction import Direction
from environment.Battlesnake.model.Position import Position
from environment.Battlesnake.model.Snake import Snake

from environment.Battlesnake.model.board_state import BoardState

from numba import njit

@njit
def _out_of_bounds(position:tuple, width, height) -> bool:
    return position[0] < 0 or position[1] < 0 or position[0] >= width or position[1] >= height

@njit
def _calc(blocked_positions_array:np.ndarray, body_dist_discount:float, influence_threshold:float) -> np.ndarray:
    result = np.zeros(blocked_positions_array.shape)
    
    height, width = result.shape
    
    y = 0
    while y < height:
        
        x = 0
        while x < width:
            
            current_block_value = blocked_positions_array[y,x]
            
            if(current_block_value == 0): 
                x += 1
                continue
            
            # set the body value on the body position itself
            result[y, x] += current_block_value
            
            
            current_block_value *= body_dist_discount
                                        
            # spread out the body value
            distance = 1
            while(abs(current_block_value) > influence_threshold):
                
                # iterate all position with given distance to the body position
                delta_x = 0
                while delta_x < distance+1:
                    delta_y = distance-delta_x
                    
                    # 4 different cases: each coordinate either with or without minus sign
                    target_position = (x+delta_x, y+delta_y)
                    if(not _out_of_bounds(position=target_position, width=width, height=height)):
                        result[target_position[1], target_position[0]] += current_block_value
                        
                    target_position = (x-delta_x, y+delta_y)
                    if(not _out_of_bounds(position=target_position, width=width, height=height)):
                        result[target_position[1], target_position[0]] += current_block_value
                        
                    target_position = (x+delta_x, y-delta_y)
                    if(not _out_of_bounds(position=target_position, width=width, height=height)):
                        result[target_position[1], target_position[0]] += current_block_value
                        
                    target_position = (x-delta_x, y-delta_y)
                    if(not _out_of_bounds(position=target_position, width=width, height=height)):
                        result[target_position[1], target_position[0]] += current_block_value
                    
                    delta_x += 1
                
                current_block_value *= body_dist_discount
                distance += 1
            
            x += 1
        
        y += 1
    
    return result


class EnemyHeuristic:
    
    BODY_DIST_DISCOUNT = 0.2
    ENEMY_HEAD_VALUE = -50
    BODY_VALUE = -100
    
    INFLUENCE_TRESHOLD = 1.0
    
    @staticmethod
    def blocked_positions_with_body_values(player_id: str, 
             board_state:BoardState, 
             enemy_head_value:int = ENEMY_HEAD_VALUE,
             body_value:int = BODY_VALUE) -> np.ndarray:
        
        blocked_positions = np.zeros((board_state.height, board_state.width), dtype=float)

        player_snake = board_state.get_snake_by_id(player_id)
        
        # in case the player is dead: just return zeros
        if(player_snake is None): return blocked_positions

        for snake in board_state.snakes:
            if(not snake.is_alive()): continue
            
            # put a penalty on the square and its surroundings
            for body_position in snake.body:
                
                temp_body_value = body_value
                
                # head has a special value
                if(body_position == snake.get_head()):
                    
                    # ignore the own head of the player
                    if(snake.snake_id == player_id): continue
                    
                    temp_body_value = enemy_head_value
                    
                    # head to head are great if the length of the player snake is longer
                    if(len(snake.body) < len(player_snake.body)):
                        temp_body_value *= -1
                        
                # set the body value on the body position itself
                blocked_positions[body_position.y, body_position.x] += temp_body_value
        
        return blocked_positions
    
    @staticmethod
    def calc(player_id: str, 
             board_state:BoardState, 
             body_dist_discount:float = BODY_DIST_DISCOUNT,
             enemy_head_value:int = ENEMY_HEAD_VALUE,
             body_value:int = BODY_VALUE,
             influence_threshold:float = INFLUENCE_TRESHOLD,
             blocked_positions:np.ndarray = None) -> np.ndarray:
        
        if(blocked_positions is None):
            blocked_positions = EnemyHeuristic.blocked_positions_with_body_values(player_id=player_id, 
                                                                                  board_state=board_state,
                                                                                  enemy_head_value=enemy_head_value,
                                                                                  body_value=body_value)
        
                
        return _calc(blocked_positions_array=blocked_positions, body_dist_discount=body_dist_discount, influence_threshold=influence_threshold)
    
#######################################################################################################################################################
    # test implementation
    
    @staticmethod
    def test_calc():
        
        np.set_printoptions(precision=2, suppress=True, linewidth=150)
        
        board_states:list[BoardState] = []
        
        snake_1 = Snake(snake_id="Player1", body=[Position(6, 6)])
        dummy_board_state = BoardState(turn=0, width=14, height=14, snakes=[snake_1])
        board_states.append(dummy_board_state)
        
        snake_1 = Snake(snake_id="Player1", body=[Position(6, 6), Position(6,5)])
        dummy_board_state = BoardState(turn=0, width=14, height=14, snakes=[snake_1])
        board_states.append(dummy_board_state)
        
        snake_1 = Snake(snake_id="Player1", body=[Position(6, 6), Position(6,5), Position(6,4)])
        dummy_board_state = BoardState(turn=0, width=14, height=14, snakes=[snake_1])
        board_states.append(dummy_board_state)
        
        snake_1 = Snake(snake_id="Player1", body=[Position(6, 6), Position(6,5), Position(6,4), Position(6,3), Position(7,3)])
        snake_2 = Snake(snake_id="Player2", body=[Position(7, 7), Position(8,7), Position(9,7), Position(10,7)])
        dummy_board_state = BoardState(turn=0, width=14, height=14, snakes=[snake_1, snake_2])
        board_states.append(dummy_board_state)
        
        for current_board_state in board_states:
            
            print("====================================================================================================")
            print("====================================================================================================")
            print(f"Calculating the enemy heuristic for the following board state:\n{Util.to_string(current_board_state)}")
            
            for snake in current_board_state.snakes:
                current_id = snake.snake_id
            
                current_calculated_heuristic = EnemyHeuristic.calc(current_id, current_board_state)
                print(f"\nCalculated heuristic for player '{current_id}':\n{current_calculated_heuristic}")
                
                
    @staticmethod
    def test_calc_norm():
        
        np.set_printoptions(precision=2, suppress=True, linewidth=150)
        
        board_states:list[BoardState] = []
        
        snake_1 = Snake(snake_id="Player1", body=[Position(6, 6)])
        dummy_board_state = BoardState(turn=0, width=14, height=14, snakes=[snake_1])
        board_states.append(dummy_board_state)
        
        snake_1 = Snake(snake_id="Player1", body=[Position(6, 6), Position(6,5)])
        dummy_board_state = BoardState(turn=0, width=14, height=14, snakes=[snake_1])
        board_states.append(dummy_board_state)
        
        snake_1 = Snake(snake_id="Player1", body=[Position(6, 6), Position(6,5), Position(6,4)])
        dummy_board_state = BoardState(turn=0, width=14, height=14, snakes=[snake_1])
        board_states.append(dummy_board_state)
        
        snake_1 = Snake(snake_id="Player1", body=[Position(6, 6), Position(6,5), Position(6,4), Position(6,3), Position(7,3)])
        snake_2 = Snake(snake_id="Player2", body=[Position(7, 7), Position(8,7), Position(9,7), Position(10,7)])
        dummy_board_state = BoardState(turn=0, width=14, height=14, snakes=[snake_1, snake_2])
        board_states.append(dummy_board_state)
        
        for current_board_state in board_states:
            
            print("====================================================================================================")
            print("====================================================================================================")
            print(f"Calculating the enemy heuristic for the following board state:\n{Util.to_string(current_board_state)}")
            
            for snake in current_board_state.snakes:
                current_id = snake.snake_id
            
                current_calculated_heuristic = Util.normalize_heuristic(EnemyHeuristic.calc(current_id, current_board_state))
                print(f"\nCalculated heuristic for player '{current_id}':\n{current_calculated_heuristic}")