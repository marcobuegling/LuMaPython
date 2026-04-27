import copy
import math
import numpy as np
from typing import Dict, Tuple

ITERATIONS = 30
K = 10
LEARNING_RATE = 0.99

class NashGrid:
    '''Defines a Nashgrid for N players. 
    Stores n dimensional data of the structure [playerId][player1_dimension][player2_dimension]...[playerN_dimension].'''
    
    def __init__(self, numbers:np.ndarray):
        '''Creates a Nd Nashgrid with the specified numbers in it.
        Also rounds the cell entries to 2 decimal places.'''
        self.setContents(numbers)
        
        
        
    def setContents(self, numbers:np.ndarray):
        self._contents = numbers
        self.dtype = self._contents.dtype
        
    def getContents(self):
        return copy.deepcopy(self._contents)
        
        
    def __eq__(self, other) -> bool:
        if(not issubclass(type(other), NashGrid)): return False
        
        other:NashGrid = other
        
        return np.array_equal(self._contents, other._contents)
    
    def __str__(self):
     return str(self._contents)

    def copy(self) -> 'NashGrid':
        return NashGrid(self.getContents())
    
    def solveWithLogitEquilibrium(self) -> Tuple[np.ndarray, dict, dict]:
        '''Solves this Nashgrid and returns the solution in the form of an tuple.
        The first index contains an array of the expected utilities for each player.
        The second index contains the probabilities used as the policy to achieve these overall utilities.
        The third index contains the expected utilities for each action for each player.'''
        
        playerCount = len(self._contents)
        
        # initialise the volatile probabilities and expected values for each action for each player
        playerProbabilitiesDict:Dict[int, np.ndarray] = dict()
        playersExpectedValuesDict:Dict[int, np.ndarray] = dict()
        
        for playerId in range(playerCount):
            playerProbabilitiesDict[playerId] = np.full(shape=(self._contents.shape[playerId+1]), fill_value=1.0/self._contents.shape[playerId+1])
            playersExpectedValuesDict[playerId] = np.zeros(shape=(self._contents.shape[playerId+1]))
        
        # this algorithm optimizes iteratively
        for iteration in range(ITERATIONS):
            
            # for each player the new expected utilities are calculated
            for playerId in range(playerCount):
            
                # create tensors of dimensionality (action space) ^ (player count)
                eu_tensor = self._contents[playerId]
                p_tensor = np.ones(shape=(self._contents.shape[playerId+1])) if playerId == 0 else playerProbabilitiesDict[0]
                for intern_player_id in range(1, playerCount):
                    other_p_tensor = np.ones(shape=(self._contents.shape[playerId+1])) if intern_player_id == playerId else playerProbabilitiesDict[intern_player_id]
                    p_tensor = np.multiply.outer(p_tensor, other_p_tensor)
                    
                
                # recalculate the expected value each iteration using the new probabilities
                new_eu_tensor = p_tensor*eu_tensor
                
                # the expected values per action are extracted from the tensor
                all_axis_shape = tuple( i for i in range(playerCount) )
                
                specific_axis_shape = tuple(i for i in all_axis_shape if i != playerId)
                playersExpectedValuesDict[playerId] = new_eu_tensor.sum(axis=specific_axis_shape)
                    
            # recalculate the probabilities again,now based on the new expected values
            
            # for each player each action now has a new probability
            for playerId in range(playerCount):
                for actionId in range(len(playersExpectedValuesDict[playerId])):
                    
                    # this is basically just normalization of the expected values
                    actionValue = math.exp(K*playersExpectedValuesDict[playerId][actionId])
                    allActionsSumValue = np.sum(np.exp(K*playersExpectedValuesDict[playerId]))
                    newProbability = actionValue / allActionsSumValue
                    
                    # the probabilities are not updated fully, but rather in smaller steps decreasing in size over time
                    # this should make the probabilities converge (hopefully)
                    learningFactor = LEARNING_RATE * (1 - iteration/ITERATIONS)
                    playerProbabilitiesDict[playerId][actionId] = learningFactor * newProbability + (1-learningFactor) * playerProbabilitiesDict[playerId][actionId]
                
            
        # calculate the overall expected utility for each player
        playerUtilities = []
                    
        for playerId in range(playerCount):
            utility = np.sum(playerProbabilitiesDict[playerId] * playersExpectedValuesDict[playerId])
            playerUtilities.append(utility)
            
        # make the result a numpy array to make it consitent with the other return values
        playerUtilities = np.array(playerUtilities)
            
        return (playerUtilities, playerProbabilitiesDict, playersExpectedValuesDict)
    
    @staticmethod
    def _round_dict_of_ndarrays(array_dict: Dict[int, np.ndarray], floating_point_precision:int = 3):
        for key in array_dict:
            array_dict[key] = np.round(array_dict[key], floating_point_precision)
    
    @staticmethod
    def round_results(results_tuple: Tuple[np.ndarray, dict, dict], floating_point_precision:int = 3):
        if(results_tuple is None or floating_point_precision is None): raise ValueError("The input argument cannot be 'None'!")
        
        playerUtilities, playerProbabilitiesDict, playersExpectedValuesDict = results_tuple
        
        np.round(a=playerUtilities, decimals=floating_point_precision, out=playerUtilities)
        NashGrid._round_dict_of_ndarrays(playerProbabilitiesDict, floating_point_precision)
        NashGrid._round_dict_of_ndarrays(playersExpectedValuesDict, floating_point_precision)
        

# Test out the implementation
if __name__ == "__main__":

    ngrid1 = NashGrid(np.zeros((2,2,3)))
    ngrid2 = NashGrid(np.array( [[[1, 2, 1], [0, 0, 10]], [[-1, 0, 3], [3, 4, 5]]] ))
    ngrid3 = NashGrid(np.array( [[[1, -2, -10], [-4, 0, 10]], [[-1, 2, 10], [4, 0, -10]]] ))
    ngrid4 = NashGrid(np.array( [[[-1, 1, -1], [1, -1, 1], [-1, 1, -1]], [[1, -1, 1], [-1, 1, -1], [1, -1, 1]]] ))
    ngrid5 = NashGrid(np.array( [[[1, -1, 0, 0], [0, 1, -1, 0], [0, 0, 1, -1], [-1, 0, 0, 1]], [[-1, 1, 0, 0], [0, -1, 1, 0], [0, 0, -1, 1], [1, 0, 0, -1]]] ))
    ngrid6 = NashGrid(np.array([[[2, 0], [1, 1]],[[2, 1], [0, 1]]]))
    ngrid7 = NashGrid(np.array([[[9, 12, 12], [8, 12, 16], [10, 10, 15]],[[9, 8, 10], [12, 12, 10], [12, 16, 15]]]))
    
    player1_3d_grid = np.array([ [[2,1,1], [2,2,2], [0,-2,2]], [[2,-1,-1], [2,2,2], [2,0,0]], [[0,0,0], [0,0,0], [0,0,0]] ])
    player2_3d_grid = np.array([ [[2,-1,-1], [0,-1,-1], [1,-1,-1]], [[2,1,1], [1,0,0], [2,1,1]], [[0,0,0], [0,0,0], [0,0,0]] ])
    player3_3d_grid = np.array([ [[1,0,0],[1,0,0],[1,0,0]], [[-2,0,0], [-2,0,0], [-2,0,0]], [[1,1,1], [1,1,1], [-1,-1,-1]] ])
    ngrid3d = NashGrid(np.array( [player1_3d_grid, player2_3d_grid, player3_3d_grid]  ))

    #for grid in [ngrid3d]:
    for grid in [ngrid1, ngrid2, ngrid3, ngrid4, ngrid5, ngrid6, ngrid7, ngrid3d]:
        print("====================================================================================================")
        print("Solving the following grid:")
        print(grid)

        results = grid.solveWithLogitEquilibrium()
        NashGrid.round_results(results)
        utilities, probsDict, expValsDict = results

        print("Resulting Probabilities: \n%s\nResulting Expected Values: \n%s\nOverall Expected Utilities for each Player:\n%s"%(probsDict, expValsDict, utilities))