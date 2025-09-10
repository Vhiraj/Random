```python
"""
Procedural 2D Dungeon Generation with Cellular Automata and A* Pathfinding

This script generates a 2D dungeon using a Cellular Automata algorithm.
The process involves:
1.  **Initialization**: The grid is randomly filled with a percentage of wall tiles.
2.  **Cellular Automata Simulation**: The grid undergoes several simulation steps.
    In each step, a cell's state (wall or floor) changes based on the number
    of its neighboring wall cells. This process smooths out the random noise
    into cave-like structures.
3.  **Connectivity Enforcement**: After the CA simulation, the script identifies
    the largest connected component of floor tiles using a Breadth-First Search (BFS).
    All other smaller, isolated floor regions are converted into walls,
    ensuring that there is one main traversable area in the dungeon.
4.  **Pathfinding**: Once the dungeon is generated and connected, the A* search
    algorithm is used to find the shortest path between a randomly chosen
    start and end point within the dungeon's floor tiles.
5.  **Visualization**: The generated dungeon, along with the found path,
    is printed to the console.

The goal is to provide a clear, well-structured example of these two fundamental
algorithms applied together for game development or procedural content generation.
"""

import random
import heapq
from collections import deque

# --- Configuration Constants ---
GRID_WIDTH = 80
GRID_HEIGHT = 40
INITIAL_FILL_PERCENT = 0.45  # Percentage of cells initially filled with walls (0.0 to 1.0)
NUM_SIMULATION_STEPS = 5     # Number of Cellular Automata simulation steps

# Cellular Automata Rules (common for cave generation):
# R1_THRESHOLD: If a wall cell has fewer than this many wall neighbors, it becomes floor.
# R2_THRESHOLD: If a floor cell has more than this many wall neighbors, it becomes wall.
# (A 3x3 neighborhood is considered, excluding the cell itself, so 0-8 neighbors)
R1_THRESHOLD = 4  # e.g., if wall has < 4 neighbors, it becomes floor (dies)
R2_THRESHOLD = 5  # e.g., if floor has > 5 neighbors, it becomes wall (born)

SMOOTHING_ITERATIONS = 2     # Number of post-processing smoothing passes
                             # (e.g., removing isolated 1x1 walls/floors)

# Display characters for the dungeon grid
FLOOR = '.'
WALL = '#'
PATH = '*'
START = 'S'
END = 'E'

# A* Pathfinding movement directions:
# Using 4-directional movement (cardinal directions only) for simplicity.
# For 8-directional movement, use DIRECTIONS_8 and adjust heuristic if needed.
DIRECTIONS = [
    (0, 1),   # Up
    (0, -1),  # Down
    (1, 0),   # Right
    (-1, 0)   # Left
]

# 8-directional movement (used for connected component check)
DIRECTIONS_8 = DIRECTIONS + [
    (1, 1),   # Up-Right
    (1, -1),  # Down-Right
    (-1, 1),  # Up-Left
    (-1, -1)  # Down-Left
]

class DungeonGenerator:
    """
    Generates a 2D dungeon using Cellular Automata and ensures a single large
    connected component of floor tiles suitable for pathfinding.
    """
    def __init__(self, width, height,
                 initial_fill_percent=INITIAL_FILL_PERCENT,
                 num_simulation_steps=NUM_SIMULATION_STEPS,
                 r1_threshold=R1_THRESHOLD,
                 r2_threshold=R2_THRESHOLD,
                 smoothing_iterations=SMOOTHING_ITERATIONS):
        """
        Initializes the dungeon generator with specified parameters.

        Args:
            width (int): The width of the dungeon grid.
            height (int): The height of the dungeon grid.
            initial_fill_percent (float): Probability a cell starts as a wall.
            num_simulation_steps (int): How many CA steps to run.
            r1_threshold (int): CA rule: wall becomes floor if < this many neighbors.
            r2_threshold (int): CA rule: floor becomes wall if > this many neighbors.
            smoothing_iterations (int): How many times to run the post-CA smoothing.
        """
        self.width = width
        self.height = height
        self.initial_fill_percent = initial_fill_percent
        self.num_simulation_steps = num_simulation_steps
        self.r1_threshold = r1_threshold
        self.r2_threshold = r2_threshold
        self.smoothing_iterations = smoothing_iterations
        self.grid = [] # The 2D list representing the dungeon

    def _initialize_grid(self):
        """
        Initializes the grid with a random distribution of walls and floors.
        Cells are more likely to be walls based on `initial_fill_percent`.
        """
        self.grid = [[WALL if random.random() < self.initial_fill_percent else FLOOR
                      for _ in range(self.width)]
                     for _ in range(self.height)]

    def _count_wall_neighbors(self, x, y, current_grid):
        """
        Counts the number of wall neighbors for a given cell (x, y)
        in a 3x3 area (including diagonals).

        Args:
            x (int): X-coordinate of the cell.
            y (int): Y-coordinate of the cell.
            current_grid (list[list[str]]): The grid to check neighbors in.

        Returns:
            int: The count of wall neighbors.
        """
        count = 0
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue  # Don't count self
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    if current_grid[ny][nx] == WALL:
                        count += 1
        return count

    def _do_simulation_step(self):
        """
        Applies one step of the Cellular Automata rules to the dungeon grid.
        A new grid is created to apply changes simultaneously, preventing
        changes from affecting neighbor counts within the same step.
        """
        new_grid = [[FLOOR for _ in range(self.width)] for _ in range(self.height)]

        for y in range(self.height):
            for x in range(self.width):
                wall_neighbors = self._count_wall_neighbors(x, y, self.grid)

                if self.grid[y][x] == WALL:
                    # Rule R1: A wall cell with fewer than R1_THRESHOLD neighbors becomes floor.
                    if wall_neighbors < self.r1_threshold:
                        new_grid[y][x] = FLOOR
                    else:
                        new_grid[y][x] = WALL
                else: # Cell is a FLOOR
                    # Rule R2: A floor cell with more than R2_THRESHOLD neighbors is born (becomes wall).
                    if wall_neighbors > self.r2_threshold:
                        new_grid[y][x] = WALL
                    else:
                        new_grid[y][x] = FLOOR
        self.grid = new_grid

    def _smooth_dungeon(self):
        """
        An optional post-processing step to further smooth out rough edges
        or remove isolated single walls/floors.
        Rule: An isolated floor cell (surrounded by 8 walls) becomes a wall.
              An isolated wall cell (surrounded by 0 walls) becomes a floor.
        """
        for _ in range(self.smoothing_iterations):
            new_grid = [row[:] for row in self.grid] # Make a copy to apply changes simultaneously
            for y in range(self.height):
                for x in range(self.width):
                    wall_neighbors = self._count_wall_neighbors(x, y, self.grid)

                    if self.grid[y][x] == FLOOR and wall_neighbors == 8:
                        new_grid[y][x] = WALL # Isolated floor becomes wall
                    elif self.grid[y][x] == WALL and wall_neighbors == 0:
                        new_grid[y][x] = FLOOR # Isolated wall becomes floor
            self.grid = new_grid

    def _get_largest_connected_component(self):
        """
        Finds the largest connected component of floor tiles using BFS.
        All other floor tiles (smaller components) are converted to walls
        to ensure pathfinding is possible within a single large area.
        """
        visited = [[False for _ in range(self.width)] for _ in range(self.height)]
        largest_component = set()

        for y in range(self.height):
            for x in range(self.width):
                if self.grid[y][x] == FLOOR and not visited[y][x]:
                    current_component = set()
                    q = deque([(x, y)])
                    visited[y][x] = True
                    current_component.add((x, y))

                    while q:
                        cx, cy = q.popleft()
                        # Use 8 directions for component check to allow diagonal connectivity
                        for dx, dy in DIRECTIONS_8:
                            nx, ny = cx + dx, cy + dy
                            if 0 <= nx < self.width and 0 <= ny < self.height and \
                               self.grid[ny][nx] == FLOOR and not visited[ny][nx]:
                                visited[ny][nx] = True
                                current_component.add((nx, ny))
                                q.append((nx, ny))

                    if len(current_component) > len(largest_component):
                        largest_component = current_component
        
        # Convert all floor tiles not in the largest component to walls
        for y in range(self.height):
            for x in range(self.width):
                if self.grid[y][x] == FLOOR and (x, y) not in largest_component:
                    self.grid[y][x] = WALL

    def generate(self):
        """
        Generates the dungeon by running the Cellular Automata process,
        applying smoothing, and then ensuring a single large connected component.
        """
        self._initialize_grid()
        for _ in range(self.num_simulation_steps):
            self._do_simulation_step()
        
        if self.smoothing_iterations > 0:
            self._smooth_dungeon()

        self._get_largest_connected_component()

    def get_grid(self):
        """
        Returns the generated dungeon grid.

        Returns:
            list[list[str]]: The 2D dungeon grid.
        """
        return self.grid

    def display(self, grid_to_display=None):
        """
        Prints the dungeon grid to the console with a border.

        Args:
            grid_to_display (list[list[str]], optional): The grid to display.
                                                         If None, displays the internal grid.
        """
        if grid_to_display is None:
            grid_to_display = self.grid
        print("\n" + "=" * (self.width + 2))
        for row in grid_to_display:
            print('|' + ''.join(row) + '|')
        print("=" * (self.width + 2) + "\n")

class AStarPathfinder:
    """
    Finds the shortest path between two points in a 2D grid using the A* algorithm.
    """
    def __init__(self, grid):
        """
        Initializes the A* pathfinder with the dungeon grid.

        Args:
            grid (list[list[str]]): The 2D dungeon grid.
        """
        self.grid = grid
        self.width = len(grid[0])
        self.height = len(grid)

    def _is_valid(self, x, y):
        """
        Checks if a coordinate is within grid bounds and is a floor tile.

        Args:
            x (int): X-coordinate.
            y (int): Y-coordinate.

        Returns:
            bool: True if valid and traversable, False otherwise.
        """
        return 0 <= x < self.width and \
               0 <= y < self.height and \
               self.grid[y][x] != WALL

    def _heuristic(self, a, b):
        """
        Manhattan distance heuristic for A* pathfinding.
        Estimates the cost from point 'a' to point 'b'.

        Args:
            a (tuple): (x, y) coordinates of the first point.
            b (tuple): (x, y) coordinates of the second point.

        Returns:
            int: The Manhattan distance.
        """
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def find_path(self, start, end):
        """
        Finds the shortest path from start to end using the A* algorithm.

        Args:
            start (tuple): (x, y) coordinates of the starting point.
            end (tuple): (x, y) coordinates of the ending point.

        Returns:
            list: A list of (x, y) coordinates representing the path from start to end,
                  or None if no path is found (e.g., start/end invalid, or blocked).
        """
        if not self._is_valid(start[0], start[1]) or \
           not self._is_valid(end[0], end[1]):
            # Start or end is on a wall or out of bounds
            return None 

        # Priority queue for nodes to explore: (f_score, (x, y))
        open_set = [(0, start)]
        heapq.heapify(open_set)

        # Dictionary to reconstruct the path after finding the end
        came_from = {}

        # g_score is the cost from start to current node
        g_score = { (x, y): float('inf') for y in range(self.height) for x in range(self.width) }
        g_score[start] = 0

        # f_score is g_score + heuristic_score
        f_score = { (x, y): float('inf') for y in range(self.height) for x in range(self.width) }
        f_score[start] = self._heuristic(start, end)

        while open_set:
            current_f, current_coords = heapq.heappop(open_set)
            current_x, current_y = current_coords

            if current_coords == end:
                # Path found, reconstruct it
                path = []
                while current_coords in came_from:
                    path.append(current_coords)
                    current_coords = came_from[current_coords]
                path.append(start)
                return path[::-1] # Reverse to get path from start to end

            for dx, dy in DIRECTIONS: # Iterate over cardinal neighbors
                neighbor_x, neighbor_y = current_x + dx, current_y + dy
                neighbor_coords = (neighbor_x, neighbor_y)

                if self._is_valid(neighbor_x, neighbor_y):
                    # Cost to move to neighbor is 1 (for cardinal directions)
                    tentative_g_score = g_score[current_coords] + 1 

                    if tentative_g_score < g_score[neighbor_coords]:
                        came_from[neighbor_coords] = current_coords
                        g_score[neighbor_coords] = tentative_g_score
                        f_score[neighbor_coords] = tentative_g_score + self._heuristic(neighbor_coords, end)
                        heapq.heappush(open_set, (f_score[neighbor_coords], neighbor_coords))
        
        return None # No path found

if __name__ == "__main__":
    print("Initializing 2D Dungeon Generation with Cellular Automata and A* Pathfinding...")

    # --- Dungeon Generation ---
    dungeon_gen = DungeonGenerator(GRID_WIDTH, GRID_HEIGHT)
    dungeon_gen.generate()
    initial_dungeon_grid = dungeon_gen.get_grid()

    print("\n--- Initial Dungeon (after CA and connectivity enforcement) ---")
    dungeon_gen.display(initial_dungeon_grid)

    # --- Pathfinding ---
    pathfinder = AStarPathfinder(initial_dungeon_grid)

    # Find all available floor tiles
    floor_tiles = []
    for y in range(GRID_HEIGHT):
        for x in range(GRID_WIDTH):
            if initial_dungeon_grid[y][x] == FLOOR:
                floor_tiles.append((x, y))

    if len(floor_tiles) < 2:
        print("Not enough floor tiles to place start and end points. Try adjusting generation parameters.")
    else:
        # Choose random start and end points from available floor tiles
        # Ensure they are distinct
        start_node = random.choice(floor_tiles)
        end_node = random.choice(floor_tiles)
        while start_node == end_node:
            end_node = random.choice(floor_tiles)

        print(f"Attempting to find path from {START} {start_node} to {END} {end_node}...")

        # Find the path using A*
        path = pathfinder.find_path(start_node, end_node)

        if path:
            print("\n--- Dungeon with Path ---")
            # Create a copy of the grid to draw the path on without altering the original dungeon
            path_display_grid = [row[:] for row in initial_dungeon_grid]

            # Mark the path on the display grid
            for px, py in path:
                if (px, py) != start_node and (px, py) != end_node:
                    path_display_grid[py][px] = PATH
            
            # Mark start and end points
            path_display_grid[start_node[1]][start_node[0]] = START
            path_display_grid[end_node[1]][end_node[0]] = END
            
            dungeon_gen.display(path_display_grid)
            print(f"Path found! Length: {len(path) - 1} steps (excluding start).")
        else:
            print("\nNo path found between the chosen start and end points.")
            print("This is unexpected if 'largest connected component' worked correctly and there are enough floor tiles.")
            print("Consider trying different start/end points or regenerating the dungeon.")

    print("\n--- Generation and pathfinding complete ---")
```