import copy
import math
from queue import Queue
from typing import Dict, List, Tuple
from numpy import ndarray
from agents.KILabAgentGroup7.LocalGameState import LocalGameState
from agents.KILabAgentGroup7.Util import Util
from environment.Battlesnake.model.Direction import Direction
from environment.Battlesnake.model.Position import Position
from environment.Battlesnake.model.RulesetSettings import RulesetSettings
from environment.Battlesnake.model.Snake import Snake

from environment.Battlesnake.model.board_state import BoardState
from environment.Battlesnake.modes.Standard import StandardGame
from .NashGrid import NashGrid

import numpy as np

PENALTY_FOR_DEATH = -10
REWARD_FOR_KILL = 1
PENALTY_FOR_GARANTUEED_SELF_KILL = -20

# going into the 4 directions (UP, DOWN, LEFT, RIGHT) are the only valid actions
ACTION_SPACE:ndarray[Direction] = np.array([dir for dir in Direction])


class CloseCombat:
    CLOSE_COMBAT_RANGE = 3

    @staticmethod
    def _evaluate(board_state: BoardState, player_ids:List[str]) -> Tuple[bool, List[int]]:
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
                eval_results.append(PENALTY_FOR_DEATH)
                dead_snakes_counter += 1
            
        # players gain points for other players being dead
        for i in range(len(player_ids)):
            snake = board_state.get_snake_by_id(player_ids[i])
            
            # snake is dead@staticmethod
            if(snake is None or not snake.is_alive()): 
                
                # all snakes but the dead snake gain points
                for j in range(len(player_ids)):
                    if(i == j): continue
                    
                    # even dead snakes gain this bonus
                    # I hope this will cause the snakes to suicide in a way, that kills others in the process
                    eval_results[j] += REWARD_FOR_KILL
        
        # the fight is over if only one snake or no snake survived
        is_fight_over = dead_snakes_counter >= len(player_ids)-1
        
        return (is_fight_over, eval_results)
    
    @staticmethod
    def _simulate(board_state: BoardState, player_ids_to_directions_dict:Dict[str, Direction]):
        '''Simulates executing the given actions for the associated players on the given board state. 
        IMPORTANT: the given board state is manipulated directly, so make a copy of the board state beforehand if necessary!
        This function does not return a new copy of the board state without affecting the original board state, 
        because copying board states has huge performance implications.'''
        
        # a game is necessary to simulation the actions
        # I use a very basic game with a ruleset where basically nothing is happening except moving the snakes
        # this should make the simulation more focussed to the movement of the snakes without distractions
        
        # this should write directly into the board state reference
        standard_game = StandardGame(ruleset_settings=Util.DUMMY_RULE_SETTINGS)
        standard_game.create_next_board_state(board=board_state, moves=player_ids_to_directions_dict, only_deterministic=True)
        
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
        nash_data = np.zeros((tuple([player_count]) + tuple([len(ACTION_SPACE)]*player_count)))
        
        # simulate 'player_count' nested for loops of iteration size '#action_space' 
        # by looping '#action_space' ^ 'player_count' times and splitting the iterations in packages of size '#action_space'
        for iteration in range(len(ACTION_SPACE)**player_count):
            
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
                denominator = max(1, (len(ACTION_SPACE) ** player_id_int))
                player_action_index = math.floor(iteration/denominator) % len(ACTION_SPACE) # TODO check if this counts correctly for each player
                player_action_index_dict[player_id_int] = player_action_index
                
                # 'player_direction_dict' is incomplete, but that does not matter since it is now not going to be used anyway,
                # no simulation is necessary anymore, after setting the values to bad ones
                if(cell_values_per_player is not None):
                    continue
                
                # the actual id of the player (not just the indepx number) is needed for the simulation
                player_action = ACTION_SPACE[player_action_index]
                
                # before we move on to the simulation and recursion:
                # let us check if the action is non-sense anyway
                # if any player garantuees to kill himself with this action...
                # ...independet of any other snake, then the action is bad to begin with
                # in such cases, there really is no point for calculating the effect of such actions
                # there are three types of non-sensical actions: 
                # 1) running into ones own body 
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
                    if(player_action != ACTION_SPACE[0]):
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
                    
                # remove all snakes not taking action
                board_state.snakes = [snake for snake in board_state.snakes if snake.snake_id in fighting_player_ids]
                
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
    def _choose_action_for_each_player(player_probability_dict:Dict[int,np.ndarray]) -> List[Direction]:
        actions = []
        
        for player_id in player_probability_dict:
            action_probabilities = player_probability_dict[player_id]
            action_probabilities = action_probabilities/np.sum(action_probabilities)
                
            actions.append(np.random.choice(a=ACTION_SPACE, size=1, p=action_probabilities)[0])
        
        return actions
        
    
    @staticmethod
    def _calculate_actions_and_statistics(local_game_state:LocalGameState, fighting_player_ids:List[str], max_depth:int, time_in_millis:int=1000000) -> Tuple[np.ndarray[Direction], np.ndarray[int], dict, dict]:
        
        start_time = Util.current_millis_time()
        
        
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
    
        #print("TIME:" + str(Util.current_millis_time()- start_time)
        
        return (CloseCombat._choose_action_for_each_player(probsDict), utilities, probsDict, expValsDict)

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
            
        #print("Depth " + str(current_depth-1))
        
        return resulting_actions

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
                    probs_with_player_ids[fighting_player_ids[player_id]] = probs_dict[player_id]
                    exp_vals_with_player_ids[fighting_player_ids[player_id]] = exp_vals_dict[player_id]
                
                print("====================================================================================================")
                print(f"Action Space: {str(ACTION_SPACE)}")
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
                    print(f"Action Space: {str(ACTION_SPACE)}")
                    print(f"Chosen action by each player: {str(players_actions_dict)}")
                    print(f"Results in:\n\n{str(temp_test_state)}\n")
            