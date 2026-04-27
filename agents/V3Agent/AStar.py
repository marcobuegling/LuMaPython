import numpy as np
import heapq

from typing import Tuple, List

class Node:
    def __init__(self, position, parent=None, cost=0, heuristic=0):
        self.position = position
        self.parent = parent
        self.cost = cost
        self.heuristic = heuristic

    def __lt__(self, other):
        # Override less than for heapq
        return (self.cost + self.heuristic) < (other.cost + other.heuristic)

def heuristic(current, goal):
    # Euclidean distance heuristic
    return np.linalg.norm(np.array(current) - np.array(goal))

def get_neighbors(position, rows, cols):
    row, col = position
    neighbors = []

    if row > 0:
        neighbors.append((row - 1, col))
    if row < rows - 1:
        neighbors.append((row + 1, col))
    if col > 0:
        neighbors.append((row, col - 1))
    if col < cols - 1:
        neighbors.append((row, col + 1))

    return neighbors

# Execute A* search on binary numpy array (1 = occupied, 0 = free)
def astar_search(grid: np.ndarray, start: Tuple[int, int], target: Tuple[int, int]) -> List[Tuple[int, int]]:
    rows, cols = grid.shape
    visited = set()
    priority_queue = []

    if start == target: return [start]

    start_node = Node(start)
    target_node = Node(target)

    grid_temp = grid.copy()
    grid_temp[target[0], target[1]] = 0

    heapq.heappush(priority_queue, start_node)

    while priority_queue:
        current_node = heapq.heappop(priority_queue)

        if current_node.position == target_node.position:
            path = []
            while current_node:
                path.append(current_node.position)
                current_node = current_node.parent
            return path[::-1]

        if current_node.position in visited:
            continue

        visited.add(current_node.position)

        for neighbor in get_neighbors(current_node.position, rows, cols):
            if grid_temp[neighbor[0], neighbor[1]] == 1 or neighbor in visited:
                continue

            cost = current_node.cost + 1
            neighbor_node = Node(neighbor, current_node, cost, heuristic(neighbor, target))
            heapq.heappush(priority_queue, neighbor_node)

    #print("No path found to", target)
    return None  # No path found

class NodeLongestPath:
    def __init__(self, position, parent=None, cost=0, heuristic=0):
        self.position = position
        self.parent = parent
        self.cost = cost
        self.heuristic = heuristic

    def __lt__(self, other):
        # Override less than for heapq
        return (self.cost + self.heuristic) > (other.cost + other.heuristic)
    
# Execute A* search for longest path on binary numpy array (1 = occupied, 0 = free)
def astar_search_longest_path(grid: np.ndarray, start: Tuple[int, int], target: Tuple[int, int]) -> List[Tuple[int, int]]:
    rows, cols = grid.shape
    visited = set()
    priority_queue = []

    if start == target: return [start]

    start_node = NodeLongestPath(start)
    target_node = NodeLongestPath(target)

    grid_temp = grid.copy()
    grid_temp[target[0], target[1]] = 0

    heapq.heappush(priority_queue, start_node)

    while priority_queue:
        current_node = heapq.heappop(priority_queue)

        if current_node.position == target_node.position:
            path = []
            while current_node:
                path.append(current_node.position)
                current_node = current_node.parent
            return path[::-1]

        if current_node.position in visited:
            continue

        visited.add(current_node.position)

        for neighbor in get_neighbors(current_node.position, rows, cols):
            if grid_temp[neighbor[0], neighbor[1]] == 1 or neighbor in visited:
                continue

            cost = current_node.cost + 1
            neighbor_node = NodeLongestPath(neighbor, current_node, cost, heuristic(neighbor, target))
            heapq.heappush(priority_queue, neighbor_node)

    #print("No path found to", target)
    return None  # No path found