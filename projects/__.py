This Python project provides a robust and efficient implementation of Dijkstra's Algorithm for finding the shortest paths in a weighted, directed graph. It's structured for clarity, reusability, and includes best practices for production-ready code.

```python
import heapq
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Union

# --- Explanation Block ---
"""
This Python project implements Dijkstra's Algorithm to find the shortest path
from a single source node to all other nodes in a weighted, directed graph.

Dijkstra's Algorithm works by iteratively exploring the graph, maintaining
the shortest known distance from the source to each node found so far.
It uses a priority queue (implemented with a min-heap) to efficiently select
the unvisited node with the smallest tentative distance. This ensures that
when a node is finalized, the path to it is indeed the shortest possible.
It requires all edge weights to be non-negative.

Key Components:
1.  Graph Representation: The `Graph` class uses an adjacency list
    (a dictionary of dictionaries) to store nodes and their weighted edges.
2.  Dijkstra's Algorithm: The `dijkstra` function takes a graph and a
    start node, returning dictionaries for shortest distances and predecessors.
3.  Path Reconstruction: The `reconstruct_path` function uses the
    predecessor map to build the actual shortest path from the start to a
    given target node.

The code is designed for readability, reusability, and includes type hints,
docstrings, and comprehensive example usage.
"""

class Graph:
    """
    Represents a weighted, directed graph using an adjacency list.
    """
    def __init__(self):
        """
        Initializes an empty graph.
        The adjacency list maps each node (string) to another dictionary,
        where keys are neighbor nodes (strings) and values are edge weights
        (integers or floats).

        Example structure:
        {
            'A': {'B': 1, 'C': 4},
            'B': {'D': 2},
            'C': {} # C has no outgoing edges
        }
        """
        self.adj: Dict[str, Dict[str, Union[int, float]]] = defaultdict(dict)

    def add_edge(self, source: str, destination: str, weight: Union[int, float]):
        """
        Adds a directed edge from source to destination with a given weight.

        Args:
            source (str): The starting node of the edge.
            destination (str): The ending node of the edge.
            weight (Union[int, float]): The weight (cost) of the edge.
                                        Must be non-negative for Dijkstra's.

        Raises:
            ValueError: If the edge weight is negative.
        """
        if weight < 0:
            raise ValueError("Edge weights must be non-negative for Dijkstra's Algorithm.")
        
        self.adj[source][destination] = weight
        
        # Ensure destination node also exists in the graph's node set,
        # even if it has no outgoing edges. This ensures all nodes are
        # considered when initializing distances.
        if destination not in self.adj:
            self.adj[destination] = {}

    def get_nodes(self) -> List[str]:
        """
        Returns a list of all unique nodes present in the graph.
        """
        return list(self.adj.keys())

    def get_neighbors(self, node: str) -> Dict[str, Union[int, float]]:
        """
        Returns a dictionary of neighbors and their associated edge weights
        for a given node.

        Args:
            node (str): The node for which to get neighbors.

        Returns:
            Dict[str, Union[int, float]]: A dictionary where keys are neighbor nodes
                                         and values are edge weights. Returns an
                                         empty dictionary if the node has no
                                         outgoing edges or does not exist.
        """
        return self.adj.get(node, {})

def dijkstra(
    graph: Graph, start_node: str
) -> Tuple[Dict[str, Union[int, float]], Dict[str, Optional[str]]]:
    """
    Implements Dijkstra's algorithm to find the shortest path from a given
    `start_node` to all other reachable nodes in the provided graph.

    Args:
        graph (Graph): The graph object on which to run Dijkstra's.
        start_node (str): The starting node for the shortest path calculation.

    Returns:
        Tuple[Dict[str, Union[int, float]], Dict[str, Optional[str]]]:
        A tuple containing two dictionaries:
        - distances: A dictionary mapping each node to its shortest distance
                     from the `start_node`. If a node is unreachable, its distance
                     will be `float('inf')`.
        - predecessors: A dictionary mapping each node to its predecessor
                        node in the shortest path from the `start_node`.
                        Can be used to reconstruct the path. The predecessor for
                        the `start_node` itself is `None`.

    Raises:
        ValueError: If the `start_node` is not found in the graph.
    """
    if start_node not in graph.get_nodes():
        raise ValueError(f"Start node '{start_node}' not found in the graph.")

    # Initialize distances: all nodes start at infinity, source node at 0
    # Use float('inf') to represent unreachable nodes.
    distances: Dict[str, Union[int, float]] = {node: float('inf') for node in graph.get_nodes()}
    distances[start_node] = 0

    # Initialize predecessors: to reconstruct the path after algorithm completes.
    # Maps a node to the node that immediately precedes it on the shortest path
    # from the start_node.
    predecessors: Dict[str, Optional[str]] = {node: None for node in graph.get_nodes()}

    # Priority queue: stores tuples of (distance, node).
    # `heapq` module implements a min-heap, which is perfect for a priority queue.
    # The smallest distance element is always at the top.
    priority_queue: List[Tuple[Union[int, float], str]] = [(0, start_node)]

    while priority_queue:
        # Extract the node with the smallest known distance
        current_distance, current_node = heapq.heappop(priority_queue)

        # If we've already found a shorter path to `current_node`, skip it.
        # This can happen because multiple paths to the same node might be
        # pushed to the priority queue before the shortest one is processed.
        if current_distance > distances[current_node]:
            continue

        # Explore neighbors of the `current_node`
        for neighbor, weight in graph.get_neighbors(current_node).items():
            # Calculate the distance to the neighbor through the `current_node`
            distance = current_distance + weight

            # If this path to `neighbor` is shorter than any previously found path
            if distance < distances[neighbor]:
                distances[neighbor] = distance        # Update shortest distance
                predecessors[neighbor] = current_node # Update predecessor
                heapq.heappush(priority_queue, (distance, neighbor)) # Add to queue

    return distances, predecessors

def reconstruct_path(
    predecessors: Dict[str, Optional[str]], start_node: str, target_node: str
) -> Optional[List[str]]:
    """
    Reconstructs the shortest path from the `start_node` to the `target_node`
    using the `predecessors` map generated by Dijkstra's algorithm.

    Args:
        predecessors (Dict[str, Optional[str]]): A dictionary mapping each node
                                                 to its predecessor in the shortest path.
        start_node (str): The original starting node from which Dijkstra's was run.
        target_node (str): The desired end node for the path.

    Returns:
        Optional[List[str]]: A list of nodes representing the shortest path
                             from `start_node` to `target_node`. Returns `None`
                             if the `target_node` is unreachable from `start_node`
                             or if `target_node` was not part of the graph.
    """
    if target_node not in predecessors:
        # Target node was not part of the graph used for Dijkstra's or
        # the predecessors map is incomplete.
        return None

    path: List[str] = []
    current_node: Optional[str] = target_node

    # Traverse backward from `target_node` to `start_node` using the predecessors map.
    while current_node is not None and current_node != start_node:
        path.append(current_node)
        current_node = predecessors[current_node]

    # If `current_node` reached `start_node`, a path exists.
    if current_node == start_node:
        path.append(start_node)
        return path[::-1] # Reverse the path to get it in start-to-target order
    else:
        # This implies `target_node` was unreachable from `start_node`,
        # (i.e., `current_node` became `None` before reaching `start_node`).
        return None

# --- Example Usage ---
if __name__ == "__main__":
    print("--- Dijkstra's Algorithm for Shortest Path ---")

    # 1. Create a graph instance
    graph = Graph()

    # 2. Add edges to the graph
    print("\nAdding edges to the graph...")
    # Example graph structure:
    # A --4--> B <--1-- C
    # |        |       /|\
    # 2        3        | 5
    # |        |        |
    # C -------+-------> D --2--> F
    #          |          /|\
    #          +--------- E
    # G --10--> H (disconnected component)
    graph.add_edge('A', 'B', 4)
    graph.add_edge('A', 'C', 2)
    graph.add_edge('B', 'E', 3)
    graph.add_edge('C', 'B', 1)
    graph.add_edge('C', 'D', 5)
    graph.add_edge('D', 'F', 2)
    graph.add_edge('E', 'D', 1)
    graph.add_edge('E', 'F', 4)
    graph.add_edge('G', 'H', 10) # A disconnected component

    print("Graph nodes:", graph.get_nodes())
    # Expected nodes: ['A', 'B', 'C', 'E', 'D', 'F', 'G', 'H'] (order might vary)

    # 3. Choose a starting node and run Dijkstra's algorithm
    start_node_1 = 'A'
    print(f"\nRunning Dijkstra's algorithm from start node: {start_node_1}")
    try:
        distances_1, predecessors_1 = dijkstra(graph, start_node_1)

        # 4. Print shortest distances
        print(f"\nShortest Distances from {start_node_1}:")
        for node, dist in sorted(distances_1.items()):
            if dist == float('inf'):
                print(f"  {node}: Unreachable")
            else:
                print(f"  {node}: {dist}")

        # 5. Reconstruct and print shortest paths
        print(f"\nShortest Paths from {start_node_1}:")
        target_nodes_to_test_1 = ['A', 'B', 'D', 'F', 'H', 'Z'] # 'Z' is a non-existent node

        for target_node in target_nodes_to_test_1:
            if target_node not in graph.get_nodes():
                print(f"  Path to {target_node}: Target node '{target_node}' not in graph.")
                continue

            path = reconstruct_path(predecessors_1, start_node_1, target_node)
            if path:
                path_str = " -> ".join(path)
                print(f"  Path to {target_node}: {path_str} (Distance: {distances_1[target_node]})")
            else:
                print(f"  Path to {target_node}: Unreachable from {start_node_1}.")

    except ValueError as e:
        print(f"Error: {e}")

    # --- Test with another start node (from a different component) ---
    print("\n--- Testing with another start node (G) ---")
    start_node_2 = 'G'
    try:
        distances_2, predecessors_2 = dijkstra(graph, start_node_2)
        print(f"\nShortest Distances from {start_node_2}:")
        for node, dist in sorted(distances_2.items()):
            if dist == float('inf'):
                print(f"  {node}: Unreachable")
            else:
                print(f"  {node}: {dist}")

        print(f"\nShortest Paths from {start_node_2}:")
        target_nodes_to_test_2 = ['H', 'A', 'F']
        for target_node in target_nodes_to_test_2:
            if target_node not in graph.get_nodes():
                print(f"  Path to {target_node}: Target node '{target_node}' not in graph.")
                continue
            path = reconstruct_path(predecessors_2, start_node_2, target_node)
            if path:
                path_str = " -> ".join(path)
                print(f"  Path to {target_node}: {path_str} (Distance: {distances_2[target_node]})")
            else:
                print(f"  Path to {target_node}: Unreachable from {start_node_2}.")

    except ValueError as e:
        print(f"Error: {e}")

    # --- Test with an invalid start node ---
    print("\n--- Testing with an invalid start node ('InvalidNode') ---")
    try:
        dijkstra(graph, 'InvalidNode')
    except ValueError as e:
        print(f"Caught expected error: {e}")

    # --- Test adding an edge with negative weight (should raise error) ---
    print("\n--- Testing adding an edge with negative weight ---")
    try:
        graph_negative_test = Graph()
        graph_negative_test.add_edge('X', 'Y', -5)
    except ValueError as e:
        print(f"Caught expected error: {e}")

```