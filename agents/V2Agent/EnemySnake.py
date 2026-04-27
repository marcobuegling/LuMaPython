import math
from typing import List, Dict, Tuple
import numpy as np
from environment.Battlesnake.model.Snake import Snake
from environment.Battlesnake.model.Position import Position
from environment.Battlesnake.model.board_state import BoardState

from .Util import Util

#from .KILabAgent import KILabAgent


class EnemySnake():
    def __init__(self, length:int=-1, id:str = "", positions:List[Tuple[Position, int]] = []):
        self.id=id
        self.positions: List[Tuple[Position, int]] = positions
        self.estimated_length = length if length != -1 else len(positions)

        # sort by actuality of information
        if (len(self.positions) > 0): 
            self.positions:List[Tuple[Position, int]] = sorted(positions, key=lambda x: x[1])

    def update(self, positions: List[Tuple[Position, int]], own_position: Position, max_turns: int, view_radius: int):
        
        # update actuality of information, delete positions that are too old or lie in the view radius
        self.positions = [(pos, turns + 1) for pos, turns in self.positions if turns + 1 <= max_turns and self.city_block_distance(own_position.x, own_position.y, pos.x, pos. y) > view_radius]

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

    def __str__(self):
        return f'{self.id}: {str(self.positions)}'

    def __repr__(self):
        return str(self)

    def estimate_snake(self, board_height:int, board_width:int, blocked_positions: List[Position]) -> Snake:
        if(len(self.positions) == 0): return Snake(snake_id=self.id, body=[])
        
        dummy_board_state = BoardState(turn = 0, width=board_width, height=board_height)
        
        completed_body = []
    
        for body_part_id in range(len(self.positions)-1):
            current_body_pos = self.positions[body_part_id][0]
            next_body_pos = self.positions[body_part_id+1][0]
            
            # the new body cannot cross the previously blocked position, but also not its own body parts
            #blocked_positions.extend(completed_body)
            # at least that was how it was previously, now we allow crossing its own body until we think of a better solution
            
            # no need to connect body parts that are already next to each other, just add it as is
            if(Util.dist(current_body_pos, next_body_pos) <= 1):
                completed_body.append(current_body_pos)
                continue
            
            connection_result = Util.a_star_search(current_body_pos, next_body_pos, dummy_board_state, blocked_positions)
            
            # cannot connect this part of the snake
            if(connection_result == None): continue
            
            _, connection = connection_result
            connection = map(lambda tuple: tuple[0], connection)
            
            completed_body.extend(connection)

        completed_body.append(self.positions[-1][0])
        
        # up until now the body estimation only considered the evidence of snake positions and connected them
        # now we adjust the size of the created snake body by our own size estimation
        
        length_difference = len(completed_body) - self.estimated_length
        
        # in we need to shorten the snake we simply remove parts from the rear
        if(length_difference > 0):
            completed_body = completed_body[:-length_difference]
        
        # we increase the length in a rather crude way: we just duplicate the last body part
        elif(length_difference < 0):
            completed_body.extend([completed_body[-1]]*-length_difference)

        # length_difference == 0 good already

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