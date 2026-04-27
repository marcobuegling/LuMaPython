import math
from typing import List, Dict, Tuple
import numpy as np
from environment.Battlesnake.model.Snake import Snake
from environment.Battlesnake.model.Position import Position
from environment.Battlesnake.model.board_state import BoardState

from .Util import Util

#from .KILabAgent import KILabAgent


class EnemySnake():
    def __init__(self, length:int=-1, id:str = "", positions:List[Tuple[Position, int]] = [], exact_body_visible:bool=False):
        self.id=id
        self.positions: List[Tuple[Position, int]] = positions
        self.estimated_length = length if length != -1 else len(positions)

        # sort by actuality of information
        if (len(self.positions) > 0): 
            self.positions:List[Tuple[Position, int]] = sorted(positions, key=lambda x: x[1])
            
        # used to determine whether there is uncertainty about the provided by
        self.entire_body_visible = exact_body_visible

    def update(self, positions: List[Tuple[Position, int]], own_position: Position, view_radius: int):
        
        if(self.entire_body_visible):
            # completely replace stored body with actual body if the entire enemy snake body is currently visible
            self.positions.clear()
            self.estimated_length = len(positions)
        
        else: 
            # update actuality of information, delete positions that are too old or lie in the view radius
            self.positions = [(pos, turns + 1) for pos, turns in self.positions if turns + 1 <= self.estimated_length/2 and self.city_block_distance(own_position.x, own_position.y, pos.x, pos. y) > view_radius]
            
            # also make sure the length estimate is at least as high as what is seen (+1, because at least one body part is missing from view)
            if(self.estimated_length < len(positions) + 1):
                self.estimated_length = len(positions) + 1

        # add new positions
        for new_pos in positions:
            found = False
            for i, (pos, turns) in enumerate(self.positions):
                if new_pos[0] == pos:
                    found = True
                    self.positions.pop(i)
                    self.positions.append(new_pos)
                    break
            if not found:
                self.positions.append(new_pos)

        # sort by actuality of information
        self.positions = sorted(self.positions, key=lambda x: x[1])
        #print("Snake update of positions:", self.positions)

    def increment_length(self, increment: int):
        #print("Length incremented")
        self.estimated_length += increment
        
    def decrement_length(self, deincrement: int):
        #print("Length incremented")
        self.estimated_length -= deincrement

    def __str__(self):
        return f'{self.id}: {str(self.positions)}'

    def __repr__(self):
        return str(self)

    def estimate_snake(self, board_height:int, board_width:int, blocked_positions: List[Position]) -> Snake:
        if(len(self.positions) == 0): return Snake(snake_id=self.id, body=[])
        
        # remove duplicates (important for removing part temporarily)
        blocked_positions = list(set(blocked_positions))
        
        dummy_board_state = BoardState(turn = 0, width=board_width, height=board_height)
        
        completed_body = []
    
        for body_part_id in range(len(self.positions)-1):
            current_body_pos = self.positions[body_part_id][0]
            next_body_pos = self.positions[body_part_id+1][0]
            
            # the new body can now cross his own body parts and even body parts of other snakes, as long as the simulated parts are not visible
            
            # no need to connect body parts that are already next to each other, just add it as is
            if(Util.dist(current_body_pos, next_body_pos) <= 1):
                completed_body.append(current_body_pos)
                continue
            
            # the start and destination field cannot be blocked, otherwise no path will be found
            blocked_for_this_search = list(set(blocked_positions) - {current_body_pos, next_body_pos})
            connection_result = Util.a_star_search(start_field=current_body_pos, search_field=next_body_pos, 
                                                   board=dummy_board_state, blocked_positions=blocked_for_this_search)
            
            # cannot connect this part of the snake
            if(connection_result == None): continue
            
            _, connection = connection_result
            connection = map(lambda tuple: tuple[0], connection)
            
            completed_body.extend(connection)

        completed_body.append(self.positions[-1][0])
        
        # up until now the body estimation only considered the evidence of snake positions and connected them
        # now we adjust the size of the created snake body by our own size estimation
        
        target_length = int(math.ceil(self.estimated_length))
        
        # be a bit more cautious, if the enemy body is not entirely visible
        if(not self.entire_body_visible):
            target_length += 1
        
        length_difference = len(completed_body) - target_length
        
        # in we need to shorten the snake we simply remove parts from the rear
        if(length_difference > 0):
            completed_body = completed_body[:-length_difference]
        
        # we increase the length in a rather crude way: we just duplicate the last body part
        elif(length_difference < 0):
            completed_body.extend([completed_body[-1]]*-length_difference)

        # length_difference == 0 good already
        
        '''
        # check if all parts are really connected
        for body_part_id in range(len(completed_body)-1):
            first_part = completed_body[body_part_id]
            next_part = completed_body[body_part_id+1]
            

            if(Util.dist(first_part, next_part) > 1): 
                Util.eprint("Body connected incorrectly. There are holes!")
        '''
                
        return Snake(snake_id=self.id, body=completed_body)
    
    @staticmethod
    def city_block_distance(x1: int, y1: int, x2: int, y2: int) -> int:
        return abs(x1 - x2) + abs(y1 - y2)
    
    
    
    @staticmethod
    def test():
        #print(Position(2,4) == Food(2,4,10))

        for snake_length in [6, 14]:
            body = [(Position(0,4), 5), (Position(0,2), 0), (Position(5,5), 7)]
            blocked = [Position(0,3)]
            
            snake = EnemySnake(id="Test", positions=body)
            snake.estimated_length = snake_length
            
            completed_snake = snake.estimate_snake(6,6, blocked)
            
            print(f"Completed Snake Body with length {str(snake_length)}: {str(completed_snake.body)}")
        
            # allows crossing own body
            assert(Position(0,2) in completed_snake.body)
            assert(Position(0,4) in completed_snake.body)
            if(snake_length >= 10):
                assert(Position(5,5) in completed_snake.body)
            
            assert(completed_snake.body.index(Position(0,2)) < completed_snake.body.index(Position(0,4)))
            if(snake_length >= 10):
                assert(completed_snake.body.index(Position(0,4)) < completed_snake.body.index(Position(5,5)))
            assert(len(completed_snake.body) == snake_length)
            
            
        # without crossing it would be this
        #assert(completed_snake.body == [Position (0, 2), Position (1, 2), Position (1, 3), Position (1, 4), Position (0, 4), Position (0, 5), Position (1, 5), Position (2, 5), Position (3, 5), Position (4, 5), Position (5, 5)])
            
            
            
        # a more complicated test reproducing a scenario from an actual game
            
        snake1 = EnemySnake(id="Test1", positions=[(Position (3, 6), 1), (Position (2, 6), 2), (Position (3, 7), 5)])
        snake1.estimated_length = 8
        snake2 = EnemySnake(id="Test2", positions=[(Position (5, 13), 0), (Position (6, 13), 1), (Position (8, 11), 1)])
        snake2.estimated_length = 9
        snake3 = EnemySnake(id="Test3", positions=[(Position (7, 11), 0), (Position (6, 11), 1), (Position (5, 11), 1), (Position (4, 11), 1), (Position (4, 12), 5)])
        snake3.estimated_length = 4
        
        # blocked positions based on the view radius of the player
        blocked_positions = [Position (5, 4), 
                             Position (4, 5), Position (5, 5), Position (6, 5), 
                             Position (3, 6), Position (4, 6), Position (5, 6), Position (6, 6), Position (7, 6), 
                             Position (2, 7), Position (3, 7), Position (4, 7), Position (5, 7), Position (6, 7), Position (7, 7), Position (8, 7), 
                             Position (1, 8), Position (2, 8), Position (3, 8), Position (4, 8), Position (5, 8), Position (6, 8), Position (7, 8), Position (8, 8), Position (9, 8), 
                             Position (0, 9), Position (1, 9), Position (2, 9), Position (3, 9), Position (4, 9), Position (5, 9), Position (6, 9), Position (7, 9), Position (8, 9), Position (10, 9), 
                             Position (1, 10), Position (2, 10), Position (3, 10), Position (4, 10), Position (5, 10), Position (6, 10), Position (7, 10), Position (8, 10), Position (9, 10), 
                             Position (2, 11), Position (3, 11), Position (4, 11), Position (5, 11), Position (6, 11), Position (7, 11), Position (8, 11), 
                             Position (3, 12), Position (4, 12), Position (5, 12), Position (6, 12), Position (7, 12), 
                             Position (4, 13), Position (5, 13), Position (6, 13), 
                             Position (5, 14),
                            # invisible body part of snake 1
                             Position (2, 6)]
        
        print("====================================================================================================")
        print("====================================================================================================")
        print("Testing snake completion for the complicated example:")
        print(f"Snake 1 body: {snake1.positions} with estimated length: {snake1.estimated_length}")
        print(f"Snake 2 body: {snake2.positions} with estimated length: {snake2.estimated_length}")
        print(f"Snake 3 body: {snake3.positions} with estimated length: {snake3.estimated_length}")
        print("Bocked Postions (player sight + snake bodies): \n" + str(blocked_positions))
        
        snake1_complete = snake1.estimate_snake(14,14,blocked_positions)
        print("\nSnake 1 resulting body: " + str(snake1_complete))
        blocked_positions += snake1_complete.body
        
        snake2_complete = snake2.estimate_snake(14,14,blocked_positions)
        print("Snake 2 resulting body: " + str(snake2_complete))
        blocked_positions += snake2_complete.body
        
        snake3_complete = snake3.estimate_snake(14,14,blocked_positions)
        print("Snake 3 resulting body: " + str(snake3_complete))
        