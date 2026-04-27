import copy
from enum import Enum
import math
from queue import Queue
from typing import Dict, List, Tuple, Union
from numpy import ndarray
from agents.KILabAgentGroup7.CenterHeuristic import CenterHeuristic
from agents.KILabAgentGroup7.EnemyHeuristic import EnemyHeuristic
from agents.KILabAgentGroup7.FloodFill import FloodFill
from agents.KILabAgentGroup7.LocalGameState import LocalGameState
from agents.KILabAgentGroup7.Util import Util
from environment.Battlesnake.model.Direction import Direction
from environment.Battlesnake.model.EliminationEvent import EliminatedCause
from environment.Battlesnake.model.Position import Position
from environment.Battlesnake.model.RulesetSettings import RulesetSettings
from environment.Battlesnake.model.Snake import Snake

from environment.Battlesnake.model.board_state import BoardState
from environment.Battlesnake.modes.Standard import StandardGame
from .NashGrid import NashGrid

import numpy as np
from numba import njit

PENALTY_FOR_DEATH = -10
REWARD_FOR_KILL = 1
PENALTY_FOR_GARANTUEED_SELF_KILL = -20
REWARD_STEP_DISCOUNT = 0.9

CENTER_HEURISTIC_BASE_VALUE = 20

'''Going into the 4 directions (UP, DOWN, LEFT, RIGHT) are the only valid actions, but that includes killing oneself.'''
ORIGINAL_ACTION_SPACE:ndarray[Direction] = np.array([dir for dir in Direction])

class DirectionRelativeToPlayer(Enum):
    FORWARD = 'Forward'
    RIGHT = 'Right'
    LEFT = 'Left'
    
    def __str__(self) -> str:
        return self.value
    
    def __repr__(self) -> str:
        return str(self)
    
    def to_direction(self, player_id:str, board_state:BoardState) -> Direction:
        
        # retrieve snake and make sure is has a proper body
        snake = board_state.get_alive_or_dead_snake_by_id(player_id)
        
        if(snake is None): raise AttributeError("The player with the given id does not exist!")
        if(len(snake.body) < 2): raise ValueError("The snake is not long enough for relative directions!")
        
        # retrieve relevant body parts
        head = snake.body[0]
        next_body_part = snake.body[1]
        
        # calculate the direction the body is facing accordingly
        
        if(head.x > next_body_part.x and head.y == next_body_part.y):
            if(self == DirectionRelativeToPlayer.FORWARD): return Direction.RIGHT
            if(self == DirectionRelativeToPlayer.RIGHT): return Direction.UP
            if(self == DirectionRelativeToPlayer.LEFT): return Direction.DOWN
        elif(head.x < next_body_part.x and head.y == next_body_part.y):
            if(self == DirectionRelativeToPlayer.FORWARD): return Direction.LEFT
            if(self == DirectionRelativeToPlayer.RIGHT): return Direction.DOWN
            if(self == DirectionRelativeToPlayer.LEFT): return Direction.UP
        elif(head.x == next_body_part.x and head.y > next_body_part.y):
            if(self == DirectionRelativeToPlayer.FORWARD): return Direction.UP
            if(self == DirectionRelativeToPlayer.RIGHT): return Direction.LEFT
            if(self == DirectionRelativeToPlayer.LEFT): return Direction.RIGHT
        elif(head.x == next_body_part.x and head.y < next_body_part.y):
            if(self == DirectionRelativeToPlayer.FORWARD): return Direction.DOWN
            if(self == DirectionRelativeToPlayer.RIGHT): return Direction.RIGHT
            if(self == DirectionRelativeToPlayer.LEFT): return Direction.LEFT
            
        # we assume, that all body parts are connected...
        # ...and therefore cannot have both different x and different y coordinates at the same time
        
        
        raise ValueError("Something is broken, this statement should never be reached!")
    
'''3 possible direction to move in (FORWARD, RIGHT, LEFT) relative to the player. This does not include going backwards and killing oneself.'''
RELATIVE_ACTION_SPACE:ndarray[DirectionRelativeToPlayer] = np.array([dir for dir in DirectionRelativeToPlayer])


@njit
def _calc_head_safety(position:tuple, 
                      enemyHeuristic:np.ndarray, 
                      center_heuristic:np.ndarray, 
                      blocked_positions_array:np.ndarray,
                      normalization_func,
                      floodfill_func) -> np.ndarray:
    
    if(blocked_positions_array.shape != enemyHeuristic.shape or enemyHeuristic.shape != center_heuristic.shape):
        raise ValueError("The two array do not have the same dimensions!")

    # add proximity to the center as safety
    head_safety_array = enemyHeuristic + center_heuristic
    
    # floodfill is important and spaces with less then 5 squares are even counted negatively
    
    # normalize and square in order to make less safe places more impactfull
    head_safety_array = normalization_func(head_safety_array)
    head_safety_array = head_safety_array * head_safety_array
    
    # extract relevant index
    result = head_safety_array[position[1], position[0]]
    
    # safety should be a number [-0.5, 0.5]
    result -= 0.5
    
    return result


class CloseCombat:
    CLOSE_COMBAT_RANGE = 3
    
    _current_center_heuristic:Union[np.ndarray,None] = None

    @staticmethod
    def _evaluate(board_state: BoardState, player_ids:List[str]) -> Tuple[bool, np.ndarray[int]]:
        '''Evaluates the given board state from the perspective of the given snakes and return a tuple of information on state.
        The first entry of the returned tuple declares whether the fight between the snakes is decidede 
        and the second entry is a list containing the evaluation for each of the snakes in original order of the "player_ids" list.'''
        
        dead_snakes_counter = 0
        
        eval_results:list = []
        
        # players loose points for being dead
        for player_id in player_ids:
            snake = board_state.get_snake_by_id(player_id)
            if(snake is not None and snake.is_alive()): eval_results.append(0)
            else: 
                punishment = PENALTY_FOR_DEATH
                
                dead_snakes_counter += 1
                
                # head-to-heads are always considered less bad then dying otherwise
                dead_snake = board_state.get_dead_snake_by_id(player_id)
                if(dead_snake is not None and dead_snake.elimination_event is not None # for some reason the cause can be None despite dying
                   and dead_snake.elimination_event.cause == EliminatedCause.EliminatedByHeadToHeadCollision):
                   punishment *= 0.5
                   
                eval_results.append(punishment) 
                    
            
        # players gain points for other players being dead
        for player_id_int in range(len(player_ids)):
            snake = board_state.get_snake_by_id(player_ids[player_id_int])
            
            # snake is dead@staticmethod
            if(snake is None or not snake.is_alive()): 
                
                # all snakes but the dead snake gain points
                for j in range(len(player_ids)):
                    if(player_id_int == j): continue
                    
                    # even dead snakes gain this bonus
                    # I hope this will cause the snakes to suicide in a way, that kills others in the process
                    eval_results[j] += REWARD_FOR_KILL
        
        
        # add a safety value for each player as a further metric to determine the value of each game state more precisely
        for player_id_int in range(len(player_ids)):
            player_id = player_ids[player_id_int]
            
            # safety only relevant for alive snakes
            snake = board_state.get_snake_by_id(player_ids[player_id_int])
            if(snake is not None and snake.is_alive()):
                head_position = snake.get_head()
                
                blocked_positions = EnemyHeuristic.blocked_positions_with_body_values(player_id=player_id, board_state=board_state)
                
                head_safety = _calc_head_safety(position=(head_position.x, head_position.y),
                                                enemyHeuristic = EnemyHeuristic.calc(player_id=player_id, board_state=board_state, blocked_positions=blocked_positions),
                                                center_heuristic=CloseCombat._current_center_heuristic, 
                                                blocked_positions_array=blocked_positions,
                                                normalization_func=Util.normalize_heuristic,
                                                floodfill_func=FloodFill.calc)
                
                eval_results[player_id_int] += head_safety
                
                # substract the safety from the reward for other players -> the opponents want this player to be placed badly
                for enemy_id_int in range(len(player_ids)):
                    enemy_snake = board_state.get_snake_by_id(player_ids[enemy_id_int])
                    if(enemy_snake is not None and enemy_snake.is_alive() and enemy_id_int != player_id_int): 
                        
                        # substract enemy safety
                        eval_results[enemy_id_int] -= head_safety
                    
        
        # the fight is over if only one snake or no snake survived
        is_fight_over = dead_snakes_counter >= len(player_ids)-1
        
        return (is_fight_over, np.array(eval_results))
    
    @staticmethod
    def _simulate(board_state: BoardState, player_ids_to_directions_dict:Dict[str, Direction]):
        '''Simulates executing the given actions for the associated players on the given board state. 
        IMPORTANT: the given board state is manipulated directly, so make a copy of the board state beforehand if necessary!
        This function does not return a new copy of the board state without affecting the original board state, 
        because copying board states has huge performance implications.'''
        
        # a game is necessary to simulation the actions
        # we use a very basic game with a ruleset where basically nothing is happening except moving the snakes
        # this should make the simulation more focussed to the movement of the snakes without distractions
        standard_game = StandardGame(ruleset_settings=Util.DUMMY_RULE_SETTINGS)
        
        # we recreate the next board state creation to avoid having to move all snakes
        # ...and also to remove unnecessary calculations
        
        # start of recreation of "standard_game.move_snakes(board=board, moves=moves)"
        ##############################################################################
        
        
        for i, snake in enumerate(board_state.snakes):

            if not snake.is_alive():
                continue

            if len(snake.body) == 0:
                raise ValueError('found snake with zero size body')

            # the important part: ignore all snake movements for snakes with no direction to move specified
            if(snake.snake_id not in player_ids_to_directions_dict):
                continue

            move = player_ids_to_directions_dict[snake.snake_id]
            head = snake.get_head()

            try:
                new_head = head.advanced(move)
            except ValueError:

                current_direction = snake.get_current_direction()

                if current_direction is not None:
                    move = current_direction
                else:
                    move = Direction.UP

                new_head = head.advanced(move)


            # this will modify "board_state"

            # Append new head, pop old tail
            new_head = standard_game.maybe_wrap_position(new_head, board_state)
            snake.body.insert(0, new_head)
            snake.body.pop()
        
        ##############################################################################
        # end of recreation of "standard_game.move_snakes(board=board, moves=moves)"
        
        # now the rest of "standard_game.create_next_board_state" is reconstructed,
        # ... but only the necessary parts
        
        # in other words: snake health and everything related to food is ignored
        standard_game.maybe_eliminate_snakes(board=board_state)

        now_dead_snakes = []

        for s in board_state.snakes:
            if not s.is_alive():
                now_dead_snakes.append(s)

        for s in now_dead_snakes:
            board_state.snakes.remove(s)
            board_state.dead_snakes.append(s)
        
    @staticmethod
    def _solve_board_state_recursively(board_state:BoardState, fighting_player_ids:List[str], max_depth:int, time_in_millis:int=1000000) -> Tuple[ndarray, dict, dict]:
        '''Calculates appropriate probabilites for each action of each player using a Nash Solver.
        The return value is a tuple of 1) an array of expected overall utilies for each player 
        2) a dictionary mapping arrays of action probabilites to player ids and 
        3) and dictionary mapping an array of expected utilities of each action to player ids.
        The time_in_millis attributes can be set to garantuee this function stops in the specified time. 
        This function will return 'None' if the time is up, or the parameter is negative.'''
        
        # this can happen when the previous function call is out of time: no result can be calculated anymore
        if(time_in_millis < 0): return None
        
        start_time = Util.current_millis_time()
        
        player_count = len(fighting_player_ids)
        
        is_fight_over, eval_results = CloseCombat._evaluate(board_state=board_state, player_ids=fighting_player_ids)
        
        # leaf cells just return the board evaluation, no need for the other values
        if(is_fight_over or max_depth == 0): return (eval_results, None, None)
        
        # other cells need to go deeper
        # in order to do that each cell is created by estimating the board states of deeper cells
        
        # the shape is constructed according to the player count: (player_count, action_space,...,action_space), 
        # where action_space appears 'player_count' amount of times in the shape
        nash_data = np.zeros((tuple([player_count]) + tuple([len(RELATIVE_ACTION_SPACE)]*player_count)))
        
        # simulate 'player_count' nested for loops of iteration size '#action_space' 
        # by looping '#action_space' ^ 'player_count' times and splitting the iterations in packages of size '#action_space'
        for iteration in range(len(RELATIVE_ACTION_SPACE)**player_count):
            
            # time is up: no result
            if(Util.current_millis_time()-start_time >= time_in_millis): 
                return None
            
            # each itereation number corresponds to a single combination of directions for each player
            # overall there are '#action_space' ^ 'player_count' of such possible combinations
            # you can think of the iteration i as a counter which counts through all of these combinations
            # the direction for each player is therefore dictated by the iteration i 
            
            # each combination is stored in here
            player_direction_dict = {}
            player_action_index_dict = {}
            
            
            cell_values_per_player = None
            
            for player_id_int in range(player_count):
                
                # think of i as a binary counter, but with base '#action_space' instead of base 2
                denominator = max(1, (len(RELATIVE_ACTION_SPACE) ** player_id_int))
                player_action_index = math.floor(iteration/denominator) % len(RELATIVE_ACTION_SPACE)
                player_action_index_dict[player_id_int] = player_action_index
                
                # 'player_direction_dict' is incomplete, but that does not matter since it is now not going to be used anyway,
                # no simulation is necessary anymore, after setting the values to bad ones
                if(cell_values_per_player is not None):
                    continue
                
                # the actual id of the player (not just the indepx number) is needed for the simulation
                player_action = RELATIVE_ACTION_SPACE[player_action_index].to_direction(player_id=fighting_player_ids[player_id_int], board_state=board_state)
                
                # Before we move on to the simulation and recursion:
                # let us check if the action is non-sense anyway
                # if any player garantuees to kill himself with this action...
                # ...independet of any other snake, then the action is bad to begin with.
                # In such cases, there really is no point for calculating the effect of such actions
                # there are three types of non-sensical actions: 
                # 1) running into ones own body FIXME not necessary anymore, because of relative action space
                # 2) running out of bounds
                # 3) trying to move while being dead already
                player_snake = board_state.get_snake_by_id(snake_id=fighting_player_ids[player_id_int])
                
                # the first two apply here
                if(player_snake is not None and player_snake.is_alive()):
                    player_head_position = player_snake.get_head()
                    potential_head_position = player_head_position.advanced(player_action)
                
                    # let us check for condition 1 or 2
                    if(len(player_snake.body) >= 2 and potential_head_position == player_snake.body[1]
                       or board_state.is_out_of_bounds(potential_head_position)): 
                        
                        # set ALL the cell values to a very bad value
                        # this is of course not quite the true evalution for each player not killing themselves,
                        # ...but the effects of such bad actions can be just ignored, 
                        # ...since hopefully the players will not chose such actions in the end anyway
                        # the big penalty just makes every player avoid this branch
                        cell_values_per_player = np.full(shape=(player_count), fill_value=PENALTY_FOR_GARANTUEED_SELF_KILL)
                        # we cannot break here, even though 'player_direction_dict' is irrelevant at this point, since 
                        # 'player_action_index_dict' still needs to be filled out for indexing later
                        continue
                    
                    # condition 3 needs to wait a bit, since there is another useful check on living snakes:
                    # it is not very helpful to calculate a lot of variation where the snakes run away from the fight
                    # in such cases the fight ends without a winner anyway, so I just set the value of such actions to 0 (draw)
                    # loosing snakes will still prefer this over dying, but there is also no positive gain
                    
                    # check for all alive fighting players whether they will be out of attack range
                    for internal_player_id_int in range(player_count):
                        # ignore distance to oneself
                        if(internal_player_id_int == player_id_int): continue
                        
                        opponent_snake = board_state.get_snake_by_id(snake_id=fighting_player_ids[player_id_int])
                        if(opponent_snake is not None and opponent_snake.is_alive()):
                            opponent_head_position = opponent_snake.get_head()
                            
                            # if player is out of range to ANY other player the result is 0
                            # this MAY change the outcome, but hopefully only slightly
                            if(Util.dist(potential_head_position, opponent_head_position) > CloseCombat.CLOSE_COMBAT_RANGE):
                                cell_values_per_player = np.zeros(shape=(player_count))
                                continue
                    
                # now for dead players:
                else:
                    # ordinarily it would not make sense to include dead snakes in the calculation anyway,
                    # ...but the algorithm cannot reduce dimensionality of the actions space midway, because a snake died along the way
                    # that would break the upper recursion levels expecting the high dimensional result
                    # so to compensate for this problem only one action of dead players is calculated seriously, the rest is set to bad values
                    # that is an indirect dimensionality reduction while keeping the necessary structure intact
                    if(player_action != ORIGINAL_ACTION_SPACE[0]):
                        cell_values_per_player = np.full(shape=(player_count), fill_value=PENALTY_FOR_GARANTUEED_SELF_KILL)
                        continue
                
                player_direction_dict[fighting_player_ids[player_id_int]] = player_action
                
            
            # only simulate if sensible
            if(cell_values_per_player is None):
            
                # copying board states for each step of the simulation is extreme time inefficient
                # therefore only the important data is stored beforehand and reset after each simulation
                # in this case, the only important data is the positions (and states) of the snakes
                snakes_copy = [Util.copy_snake(snake) for snake in board_state.snakes]
                dead_snakes_copy = [Util.copy_snake(snake) for snake in board_state.dead_snakes]
                # it is important to note, that the copies have to be made every iteration so that the next iteration...
                # ...does not write to the same list object
                
                CloseCombat._simulate(board_state=board_state, player_ids_to_directions_dict=player_direction_dict)
                
                # overall time - time passed
                remaining_time = time_in_millis - (Util.current_millis_time()-start_time)
                
                # time is up: no result
                if(remaining_time <= 0): return None
                
                # recursion
                result_value = CloseCombat._solve_board_state_recursively(board_state=board_state, 
                                                    fighting_player_ids=fighting_player_ids,
                                                    max_depth=max_depth-1, time_in_millis=remaining_time) 
                
                # time is up: No result value
                if(result_value is None):
                    return None
                
                # only the expected utilities are important here
                cell_values_per_player = result_value[0] 
                
                # make results in the farther future less impactful
                cell_values_per_player = cell_values_per_player * REWARD_STEP_DISCOUNT
                
                # IMPORTANT: reset the board state to the state before the simulation as to not disturb the other calculation paths
                board_state.snakes = snakes_copy
                board_state.dead_snakes = dead_snakes_copy
                board_state.turn = board_state.turn -  1 # do not make the snakes starve after a few simulation steps
                
                
                
            # enter the resulting values of this specific set of actions for each player
            # the cell is defined by the actions taken by each player
            # each player has his own utility tensor, which is selected by the first data dimension
            
            # first select the correct tensor corresponding to the player in question
            for player_id_int in range(player_count):
                player_tensor = nash_data[player_id_int]
                
                # now find the correct cell
                for inner_player_id_int in range(player_count):
                    
                    # as long as the final dimension is not reached yet resolve more dimensions (go deeper)
                    if(inner_player_id_int < player_count - 1):
                        player_tensor = player_tensor[player_action_index_dict[inner_player_id_int]]
                        
                    # now that we arrived at the final dimension of the tensor: 
                    # input the utility associated with this player for the correct cell
                    else:
                        player_utility = cell_values_per_player[player_id_int]
                        player_tensor[player_action_index_dict[inner_player_id_int]] = player_utility 
            
            
        # nashdata is now filled accordingly
        return NashGrid(numbers=nash_data).solveWithLogitEquilibrium()
        
        
        
        
        
    @staticmethod
    def _choose_action_for_each_player(player_probability_dict:Dict[int,np.ndarray], board_state: BoardState) -> List[Direction]:
        actions = []
        
        for player_id in player_probability_dict:
            action_probabilities = player_probability_dict[player_id]
            action_probabilities = action_probabilities/np.sum(action_probabilities)
                
            random_relative_action = np.random.choice(a=RELATIVE_ACTION_SPACE, size=1, p=action_probabilities)[0]
            actions.append(random_relative_action.to_direction(player_id=player_id, board_state=board_state))
        
        return actions
        
    
    @staticmethod
    def _calculate_actions_and_statistics(local_game_state:LocalGameState, fighting_player_ids:List[str], max_depth:int, time_in_millis:int=1000000) -> Tuple[np.ndarray[Direction], np.ndarray[int], dict, dict]:
        
        start_time = Util.current_millis_time()
        
        CloseCombat._current_center_heuristic = CenterHeuristic.calc(local_game_state.width, 
                                                                     local_game_state.height, 
                                                                     CENTER_HEURISTIC_BASE_VALUE)
        
        if(local_game_state is None or fighting_player_ids is None): raise ValueError("Arguments cannot be 'None'!")
        
        if(len(fighting_player_ids)==0): raise ValueError("The list of fighting players is empty!")
        if(len(fighting_player_ids)==1): raise ValueError("One player cannot fight alone!")
        
        if(max_depth < 0): raise ValueError("'max_depth' cannot be negative!")
        
        # the state is estimated since it is not clear where the enemy snakes are exactly
        current_board_state = local_game_state.create_estimated_board_state()
        
        # filter out players we know nothing about
        fighting_player_ids = [id for id in fighting_player_ids if current_board_state.get_alive_or_dead_snake_by_id(id) is not None]
        if(len(fighting_player_ids)==0): return None # TODO find better way to deal with this
        
        
        result = CloseCombat._solve_board_state_recursively(
            current_board_state, 
            fighting_player_ids=fighting_player_ids, 
            max_depth=max_depth,
            time_in_millis=time_in_millis-(Util.current_millis_time()-start_time))
        
        # time is up: no result
        if(result is None): return None
        
        utilities, probsDict, expValsDict = result
        
        # create a new dict that maps the string version of player ids instead of their simple numnber
        # this is used to convert the relative directions into absolute directions
        newProbsDict = dict()
        for player_id_int in probsDict.keys():
            newProbsDict[fighting_player_ids[player_id_int]] = probsDict[player_id_int]
    
        #print("TIME:" + str(Util.current_millis_time()- start_time)
        
        return (CloseCombat._choose_action_for_each_player(newProbsDict, current_board_state), utilities, newProbsDict, expValsDict)

    @staticmethod
    def calculate_actions(local_game_state:LocalGameState, fighting_player_ids:List[str], max_depth:int, time_in_millis:int=1000000) -> np.ndarray[Direction]:
        result_value = CloseCombat._calculate_actions_and_statistics(local_game_state, fighting_player_ids, max_depth, time_in_millis=time_in_millis)
        if(result_value is None): return None
        return result_value[0]
    
    @staticmethod
    def calculate_actions_iterative_deeping(local_game_state:LocalGameState, fighting_player_ids:List[str], time_in_millis:int):
        resulting_actions = None
        
        start_time = Util.current_millis_time()
        
        current_depth = 1
        
        remaining_time = time_in_millis - (Util.current_millis_time()-start_time)
        while( remaining_time > 0):
            
            result = CloseCombat.calculate_actions(local_game_state=local_game_state,
                                                   fighting_player_ids=fighting_player_ids,
                                                   max_depth=current_depth,
                                                   time_in_millis=remaining_time)
            
            # out of time
            if(result is None): break
            else:
                resulting_actions = result
            
            current_depth += 1
            remaining_time = time_in_millis - (Util.current_millis_time()-start_time)
            
        print("Depth " + str(current_depth-1))
        
        return resulting_actions


    @staticmethod
    def compile(width=15, height=15):
        test_player_snake = Snake(snake_id="Player", body=[Position(1, 1), Position(1,2), Position(1,3)])
        test_victim_snake = Snake(snake_id="Victim", body=[Position(0, 2), Position(0,3), Position(0,4), Position(0,5)])
        dummy_board_state = BoardState(turn=0, width=width, height=height, snakes=[test_player_snake, test_victim_snake])
        test_state1 = LocalGameState(board=dummy_board_state, player_snake=test_player_snake, view_radius=5, copy_enemy_bodies=True)
        
        # precompile
        CloseCombat._calculate_actions_and_statistics(local_game_state=test_state1, 
                                            fighting_player_ids=["Victim", "Player"],
                                            max_depth=1)
        

#######################################################################################################################################################
    # test implementation
    
    @staticmethod
    def test_relative_directions():
        test_player_snake = Snake(snake_id="Player", body=[Position(1, 1), Position(1,2), Position(1,3)])
        test_snake_1 = Snake(snake_id="Snake1", body=[Position(0, 5), Position(0,4), Position(0,3), Position(0,2)])
        test_snake_2 = Snake(snake_id="Snake2", body=[Position(2, 0), Position(3,0), Position(3,1), Position(3,2)])
        test_snake_3 = Snake(snake_id="Snake3", body=[Position(3, 5), Position(2,5)])
        
        dummy_board_state = BoardState(turn=0, width=4, height=6, snakes=[test_player_snake, test_snake_1, test_snake_2, test_snake_3])
        test_state = LocalGameState(board=dummy_board_state, player_snake=test_player_snake, view_radius=5, copy_enemy_bodies=True)
        
        print("====================================================================================================")
        print("====================================================================================================")
        print(f"Testing directions for the players of the following board state:\n\n%s"%(str(test_state)))
        print("====================================================================================================")
        
        print(f"Player forward direction = {DirectionRelativeToPlayer.FORWARD.to_direction(player_id='Player', board_state=dummy_board_state)}")
        assert(DirectionRelativeToPlayer.FORWARD.to_direction(player_id='Player', board_state=dummy_board_state) == Direction.DOWN)
        print(f"Player right direction = {DirectionRelativeToPlayer.RIGHT.to_direction('Player', dummy_board_state)}")
        assert(DirectionRelativeToPlayer.RIGHT.to_direction('Player', dummy_board_state) == Direction.RIGHT)
        print(f"Player left direction = {DirectionRelativeToPlayer.LEFT.to_direction('Player', dummy_board_state)}")
        assert(DirectionRelativeToPlayer.LEFT.to_direction('Player', dummy_board_state) == Direction.LEFT)
        print("")
        print(f"Left snake forward direction = {DirectionRelativeToPlayer.FORWARD.to_direction('Snake1', dummy_board_state)}")
        assert(DirectionRelativeToPlayer.FORWARD.to_direction('Snake1', dummy_board_state) == Direction.UP)
        print(f"Left snake right direction = {DirectionRelativeToPlayer.RIGHT.to_direction('Snake1', dummy_board_state)}")
        assert(DirectionRelativeToPlayer.RIGHT.to_direction('Snake1', dummy_board_state) == Direction.LEFT)
        print(f"Left snake left direction = {DirectionRelativeToPlayer.LEFT.to_direction('Snake1', dummy_board_state)}")
        assert(DirectionRelativeToPlayer.LEFT.to_direction('Snake1', dummy_board_state) == Direction.RIGHT)
        print("")
        print(f"Upper-right snake  forward direction = {DirectionRelativeToPlayer.FORWARD.to_direction('Snake2', dummy_board_state)}")
        assert(DirectionRelativeToPlayer.FORWARD.to_direction('Snake2', dummy_board_state) == Direction.LEFT)
        print(f"Upper-right right direction = {DirectionRelativeToPlayer.RIGHT.to_direction('Snake2', dummy_board_state)}")
        assert(DirectionRelativeToPlayer.RIGHT.to_direction('Snake2', dummy_board_state) == Direction.DOWN)
        print(f"Upper-right left direction = {DirectionRelativeToPlayer.LEFT.to_direction('Snake2', dummy_board_state)}")
        assert(DirectionRelativeToPlayer.LEFT.to_direction('Snake2', dummy_board_state) == Direction.UP)
        print("")
        print(f"Lower-right forward direction = {DirectionRelativeToPlayer.FORWARD.to_direction('Snake3', dummy_board_state)}")
        assert(DirectionRelativeToPlayer.FORWARD.to_direction('Snake3', dummy_board_state) == Direction.RIGHT)
        print(f"Lower-right right direction = {DirectionRelativeToPlayer.RIGHT.to_direction('Snake3', dummy_board_state)}")
        assert(DirectionRelativeToPlayer.RIGHT.to_direction('Snake3', dummy_board_state) == Direction.UP)
        print(f"Lower-right left direction = {DirectionRelativeToPlayer.LEFT.to_direction('Snake3', dummy_board_state)}")
        assert(DirectionRelativeToPlayer.LEFT.to_direction('Snake3', dummy_board_state) == Direction.DOWN)
        

    @staticmethod
    def test_creation():
        
        test_player_snake = Snake(snake_id="Player", body=[Position(1, 1), Position(1,2), Position(1,3)])
        test_victim_snake = Snake(snake_id="Victim", body=[Position(0, 2), Position(0,3), Position(0,4), Position(0,5)])
        test_advesary_snake = Snake(snake_id="Advesary", body=[Position(2, 0), Position(3,0), Position(3,1), Position(3,2)])
        
        dummy_board_state = BoardState(turn=0, width=4, height=6, snakes=[test_player_snake, test_victim_snake])
        test_state1 = LocalGameState(board=dummy_board_state, player_snake=test_player_snake, view_radius=5, copy_enemy_bodies=True)
        
        dummy_board_state = BoardState(turn=0, width=4, height=6, snakes=[test_player_snake, test_victim_snake, test_advesary_snake])
        test_state2 = LocalGameState(board=dummy_board_state, player_snake=test_player_snake, view_radius=5, copy_enemy_bodies=True)
        
        test_states = [test_state1, test_state2]
        
        # precompile
        CloseCombat.compile()
        
        #for temp_test_state in [test_state1]:
        for temp_test_state in test_states:
        
            print("====================================================================================================")
            print("====================================================================================================")
            print(f"Simulating the following board state:\n\n%s"%(str(temp_test_state)))
            
            for steps in range(3):
                
                # look how many players are left fighting
                fighting_player_ids = temp_test_state.get_fighting_player_ids(attack_range=5)
                if(len(fighting_player_ids) <= 1): break
                
                
                result = CloseCombat._calculate_actions_and_statistics(local_game_state=temp_test_state, 
                                            fighting_player_ids=fighting_player_ids,
                                            max_depth=3, time_in_millis=1000)
                
                if(result is None):
                    print("Out of time! No result!")
                    break
                    
                actions, utilities, probs_dict, exp_vals_dict = result
                
                # round values for visual effect
                NashGrid.round_results((utilities, probs_dict, exp_vals_dict), 3)
                
                players_actions_dict = {}
                for player_id in range(len(fighting_player_ids)):
                    players_actions_dict[fighting_player_ids[player_id]] = actions[player_id]
                    
                temp_test_state = temp_test_state.move(player_ids_to_directions_dict=players_actions_dict)
                
                utilities_with_player_ids = dict()
                probs_with_player_ids = dict()
                exp_vals_with_player_ids = dict()
                for player_id in range(len(fighting_player_ids)):
                    utilities_with_player_ids[fighting_player_ids[player_id]] = utilities[player_id]
                    probs_with_player_ids[fighting_player_ids[player_id]] = probs_dict[fighting_player_ids[player_id]]
                    exp_vals_with_player_ids[fighting_player_ids[player_id]] = exp_vals_dict[player_id]
                
                print("====================================================================================================")
                print(f"Action Space: {str(RELATIVE_ACTION_SPACE)}")
                print(f"Utilities per action: {str(exp_vals_with_player_ids)}")
                print(f"Probabilities for each action: {str(probs_with_player_ids)}")
                print(f"Overall Expected Utilities for each Player: {str(utilities_with_player_ids)}")
                print(f"Chosen action by each player: {str(players_actions_dict)}")
                print(f"Results in:\n\n{str(temp_test_state)}\n")
                
    @staticmethod
    def test_iterative_deepening():
                    
            test_player_snake = Snake(snake_id="Player", body=[Position(1, 1), Position(1,2), Position(1,3)])
            test_victim_snake = Snake(snake_id="Victim", body=[Position(0, 2), Position(0,3), Position(0,4), Position(0,5)])
            test_advesary_snake = Snake(snake_id="Advesary", body=[Position(2, 0), Position(3,0), Position(3,1), Position(3,2)])
            
            dummy_board_state = BoardState(turn=0, width=4, height=6, snakes=[test_player_snake, test_victim_snake])
            test_state1 = LocalGameState(board=dummy_board_state, player_snake=test_player_snake, view_radius=5, copy_enemy_bodies=True)
            
            dummy_board_state = BoardState(turn=0, width=4, height=6, snakes=[test_player_snake, test_victim_snake, test_advesary_snake])
            test_state2 = LocalGameState(board=dummy_board_state, player_snake=test_player_snake, view_radius=5, copy_enemy_bodies=True)
            
            dummy_board_state = BoardState(turn=0, width=11, height=11, snakes=[test_player_snake, test_advesary_snake])
            test_state3 = LocalGameState(dummy_board_state, player_snake=test_player_snake, view_radius=5, copy_enemy_bodies=True)
            
            test_states = [test_state1, test_state2, test_state3]
            
            # precompile
            CloseCombat.compile()
            
            #for temp_test_state in [test_state3]:
            for temp_test_state in test_states:
                
                print("====================================================================================================")
                print("====================================================================================================")
                print(f"Simulating the following board state:\n\n%s"%(str(temp_test_state)))
                
                for step in range(100):
                
                    # look how many players are left fighting
                    fighting_player_ids = temp_test_state.get_fighting_player_ids(attack_range=5)
                    if(len(fighting_player_ids) <= 1): 
                        print("Close combat over: players dead or out of range!")
                        break
                        
                    actions = CloseCombat.calculate_actions_iterative_deeping(local_game_state=temp_test_state, 
                                                fighting_player_ids=fighting_player_ids,
                                                time_in_millis=480)
                    
                    players_actions_dict = {}
                    for player_id in range(len(fighting_player_ids)):
                        players_actions_dict[fighting_player_ids[player_id]] = actions[player_id]
                    
                    temp_test_state = temp_test_state.move(player_ids_to_directions_dict=players_actions_dict)
                        
                    print("====================================================================================================")
                    print(f"Step: {str(step)}")
                    print(f"Action Space: {str(RELATIVE_ACTION_SPACE)}")
                    print(f"Chosen action by each player: {str(players_actions_dict)}")
                    print(f"Results in:\n\n{str(temp_test_state)}\n")
                    
                    
    
    @staticmethod
    def test_crossing_bodies():       
        
        test_snake_1 = Snake(snake_id="Snake 1", body=[Position(2, 5), Position(2,4), Position(2,3), Position(2,2), Position(2,1)])
        test_snake_2 = Snake(snake_id="Snake 2", body=[Position(5, 2), Position(4,2), Position(3,2), Position(2,2), Position(1,2)])
        dummy_board_state = BoardState(turn=0, width=7, height=7, snakes=[test_snake_1, test_snake_2])
    
        print("====================================================================================================")
        print("====================================================================================================")
        print(f"Simulating the following board state:\n\n%s"%(Util.to_string(dummy_board_state)))
        
        CloseCombat._simulate(dummy_board_state, {"Snake 1": Direction.UP, "Snake 2": Direction.RIGHT})
        
        print(f"Both snakes move forward. This results in the following board state:\n{Util.to_string(dummy_board_state)}")
        print(f"Dead snakes: {dummy_board_state.dead_snakes}")