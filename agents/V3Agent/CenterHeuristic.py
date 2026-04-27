import numpy as np
from agents.KILabAgentGroup7.Util import Util

from environment.Battlesnake.model.Position import Position


class CenterHeuristic:
    
    @staticmethod
    def calc(width:int, height:int, base_value:int) -> np.ndarray:
        '''Calculates a heuristic with the given dimensions, that rewards the proximity to the center.
        The given base value will be the max value in the center position(s) and the minimum value will be 
        the negative base value. The decrease is evenly spaced out over distance to the center in proportion 
        to the given dimensions. '''
        
        heuristic = np.zeros((height, width), dtype=float)
        
        # all cells will have values ranging from -BASE_VALUE to +BASE_VALUE depending on the distance to the center
        width_denominator = width-1 if width%2==1 else width - 2
        width_step_size = base_value*2 / max(( width_denominator ),1)
        height_denominator = height-1 if height%2==1 else height - 2
        height_step_size = base_value*2 / max(( height_denominator ),1)
        
        # find ALL center positions
        center_positions = [Position(int(width/2), int(height/2))]
        if(width%2==0): center_positions.append(Position(int(width/2) - 1, int(height/2)))
        if(height%2==0): center_positions.append(Position(int(width/2), int(height/2) - 1))
        if(width%2==0 and height%2==0): center_positions.append(Position(int(width/2) - 1, int(height/2) - 1))
        
        for y in range(height):
            for x in range(width):
                
                current_position = Position(x, y)
                
                # find the closest center position
                closest_center_position = center_positions[0]
                smallest_distance = float("inf")
                for current_center_position in center_positions:
                    dist = Util.dist(current_position, current_center_position)
                    
                    if(dist < smallest_distance):
                        smallest_distance = dist
                        closest_center_position = current_center_position
                        
                # calculate reward falloff based on x and y distance to the center
                x_falloff = width_step_size * abs(current_position.x - closest_center_position.x)
                y_falloff = height_step_size * abs(current_position.y - closest_center_position.y)
                reward = base_value - x_falloff - y_falloff
                    
                
                heuristic[y,x] = reward
        
        return heuristic
    
    
#######################################################################################################################################################
    # test implementation
    
    @staticmethod
    def test_calc():
        
        np.set_printoptions(precision=2, suppress=True, linewidth=150)
        
        dimension_pairs = [(1,1), (1,2), (3,3), (3,5), (5,3), (4,4), (11,11), (15,15)]
        #dimension_pairs = [(4,4)]
        
        for dimension_pair in dimension_pairs:
            
            print("====================================================================================================")
            print("====================================================================================================")
            print(f"Calculating the center heuristic for the following dimensions:\nx={dimension_pair[0]}, y={dimension_pair[1]}")
            current_calculated_heuristic = CenterHeuristic.calc(dimension_pair[0], dimension_pair[1], 5)
            print(f"\nCalculated heuristic:\n{current_calculated_heuristic}")
                