```python
import heapq
import math

"""
dijkstra.py

This script implements Dijkstra's Algorithm to find the shortest paths from a single source node
to all other nodes in a weighted, directed (or undirected) graph.

Dijkstra's Algorithm works by iteratively selecting the unvisited node with the smallest known
distance from the source and updating the distances of its neighbors. It guarantees to find
the shortest paths in graphs where all edge weights are non-negative.

Key components:
1.  Graph Representation: An adjacency list, implemented as a dictionary where:
    -   Keys are nodes (e.g., 'A', 1, etc.).
    -   Values are lists of tuples, each tuple `(neighbor, weight)` representing an edge
        from the key node to the `neighbor` node with a specified `weight`.
    Example: {'A': [('B', 1), ('C', 4)], 'B': [('C', 2)]}

2.  Priority Queue: Implemented using Python's `heapq` module. This module provides a
    min-heap data structure, which efficiently allows:
    -   Adding new `(distance, node)` pairs.
    -   Extracting the node with the smallest `distance` in O(log N) time,
        where N is the number of nodes in the queue.

3.  Distance Tracking: A dictionary `distances` that stores the shortest distance found so far
    from the source node to every other node in the graph. It's initialized with `infinity`
    for all nodes except the `start_node` (which is 0).

4.  Predecessor Tracking: A dictionary `predecessors` that stores the immediate predecessor
    node for each node on its shortest path from the `start_node`. This is crucial for
    reconstructing the actual path after the algorithm completes.

Usage:
-   Define your graph as a dictionary following the specified adjacency list format.
-   Call the `dijkstra(graph, start_node)` function to compute shortest distances and predecessors.
-   Optionally, use `reconstruct_path(predecessors, start_node, target_node)` to get the
    actual sequence of nodes in the shortest path to a specific `target_node`.
"""


def dijkstra(graph, start_node):
    """
    Implements Dijkstra's algorithm to find the shortest path from a start node
    to all other nodes in a weighted graph.

    Args:
        graph (dict): A dictionary representing the graph.
                      Keys are nodes, values are lists of (neighbor, weight) tuples.
                      Example: {'A': [('B', 1), ('C', 4)], 'B': [('C', 2)]}
        start_node: The starting node for the algorithm. Must be present in the graph.

    Returns:
        tuple: A tuple containing two dictionaries:
               - distances (dict): Shortest distance from start_node to all other nodes.
                                   Keys are nodes, values are their shortest distances.
                                   `math.inf` if a node is unreachable.
               - predecessors (dict): Predecessor of each node in the shortest path.
                                      Keys are nodes, values are their predecessor nodes.
                                      Used for path reconstruction.

    Raises:
        ValueError: If the start_node is not found in the graph.
    """
    if start_node not in graph:
        raise ValueError(f"Start node '{start_node}' not found in the graph.")

    # Initialize distances: all to infinity, start_node to 0
    # Use math.inf for unreachable nodes.
    distances = {node: math.inf for node in graph}
    distances[start_node] = 0

    # Priority queue: stores tuples of (distance, node).
    # `heapq` module treats the first element of a tuple as the priority.
    # The smallest distance node is always at the top (index 0)
    priority_queue = [(0, start_node)]

    # Predecessors dictionary to reconstruct the path after the algorithm finishes.
    # It stores {node: predecessor_node}.
    predecessors = {}

    while priority_queue:
        # Extract the node with the smallest distance from the priority queue.
        current_distance, current_node = heapq.heappop(priority_queue)

        # If we've already found a shorter path to `current_node` and processed it, skip.
        # This can happen because `heapq` allows pushing duplicate entries, and we only
        # care about the first time we extract a node with its true minimum distance.
        if current_distance > distances[current_node]:
            continue

        # Explore neighbors of the `current_node`.
        # `graph.get(current_node, [])` handles cases where a node might have no outgoing edges.
        for neighbor, weight in graph.get(current_node, []):
            distance = current_distance + weight

            # If a shorter path to the `neighbor` is found, update its distance
            # and push it to the priority queue.
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                predecessors[neighbor] = current_node
                heapq.heappush(priority_queue, (distance, neighbor))

    return distances, predecessors


def reconstruct_path(predecessors, start_node, target_node):
    """
    Reconstructs the shortest path from the start node to the target node
    using the predecessors dictionary generated by Dijkstra's algorithm.

    Args:
        predecessors (dict): Dictionary mapping a node to its predecessor in the
                             shortest path from the start node.
                             Generated by `dijkstra()`.
        start_node: The starting node of the path.
        target_node: The target node of the path.

    Returns:
        list: A list of nodes representing the shortest path from `start_node` to `target_node`.
              Returns an empty list if `target_node` is unreachable from `start_node`.
              If `start_node` equals `target_node`, it returns `[start_node]`.
    """
    path = []
    current = target_node

    # Trace back from the target node to the start node using the predecessors dictionary.
    while current is not None:
        path.append(current)
        if current == start_node:
            break  # Reached the start node, path is complete.
        if current not in predecessors:
            # This means the target_node is unreachable from start_node,
            # or it's an isolated node not connected to the path.
            return []
        current = predecessors[current]

    # If the path exists but doesn't start with the start_node (meaning it's incomplete
    # or the target was unreachable), return an empty list.
    if path and path[-1] != start_node:
        return []

    # The path was built in reverse, so reverse it to get it from start_node to target_node.
    return path[::-1]


if __name__ == "__main__":
    print("--- Dijkstra's Algorithm Implementation ---")

    # --- Example Graph 1: A typical connected graph ---
    graph1 = {
        'A': [('B', 1), ('C', 4)],
        'B': [('C', 2), ('D', 5)],
        'C': [('D', 1)],
        'D': [('E', 3)],
        'E': []  # No outgoing edges from E
    }
    start_node1 = 'A'
    print(f"\n--- Example Graph 1 (Connected Graph) ---")
    print(f"Graph: {graph1}")
    print(f"Starting node: {start_node1}")

    try:
        distances1, predecessors1 = dijkstra(graph1, start_node1)
        print("\nShortest distances from A:")
        for node, dist in sorted(distances1.items()):
            print(f"  To {node}: {dist if dist != math.inf else 'Infinity'}")

        print("\nShortest paths from A:")
        for target_node in sorted(graph1.keys()):
            path = reconstruct_path(predecessors1, start_node1, target_node)
            distance = distances1.get(target_node, math.inf) # Get distance from the computed dict

            if path: # Check if a path was found
                print(f"  A to {target_node}: {' -> '.join(path)} (Distance: {distance if distance != math.inf else 'Infinity'})")
            else:
                print(f"  A to {target_node}: No path found (Distance: {distance if distance != math.inf else 'Infinity'})")

    except ValueError as e:
        print(f"Error: {e}")

    # --- Example Graph 2: Graph with a disconnected component ---
    graph2 = {
        'X': [('Y', 7), ('Z', 9)],
        'Y': [('W', 2)],
        'Z': [('W', 12)],
        'W': [],
        'P': [('Q', 1)],  # Disconnected component
        'Q': []
    }
    start_node2 = 'X'
    target_node2 = 'P'  # This node is unreachable from 'X'
    print(f"\n--- Example Graph 2 (Disconnected Component) ---")
    print(f"Graph: {graph2}")
    print(f"Starting node: {start_node2}")

    try:
        distances2, predecessors2 = dijkstra(graph2, start_node2)
        print("\nShortest distances from X:")
        for node, dist in sorted(distances2.items()):
            print(f"  To {node}: {dist if dist != math.inf else 'Infinity'}")

        print(f"\nAttempting to find path from {start_node2} to {target_node2} (unreachable):")
        path2 = reconstruct_path(predecessors2, start_node2, target_node2)
        distance2 = distances2.get(target_node2, math.inf) # Get distance from the computed dict

        if path2:
            print(f"  {start_node2} to {target_node2}: {' -> '.join(path2)} (Distance: {distance2 if distance2 != math.inf else 'Infinity'})")
        else:
            print(f"  {start_node2} to {target_node2}: No path found (Distance: {distance2 if distance2 != math.inf else 'Infinity'})")

    except ValueError as e:
        print(f"Error: {e}")

    # --- Example Graph 3: Testing with a non-existent start node ---
    graph3 = {'A': [('B', 1)]}
    start_node3 = 'Z'  # 'Z' is not in graph3
    print(f"\n--- Example Graph 3 (Non-existent Start Node) ---")
    print(f"Graph: {graph3}")
    print(f"Starting node: {start_node3}")
    try:
        dijkstra(graph3, start_node3)
    except ValueError as e:
        print(f"Caught expected error: {e}")

    # --- Example Graph 4: Target node is the same as the start node ---
    graph4 = {
        'S': [('A', 1)],
        'A': []
    }
    start_node4 = 'S'
    target_node4 = 'S'
    print(f"\n--- Example Graph 4 (Target Node is Start Node) ---")
    print(f"Graph: {graph4}")
    print(f"Starting node: {start_node4}, Target node: {target_node4}")

    try:
        distances4, predecessors4 = dijkstra(graph4, start_node4)
        path4 = reconstruct_path(predecessors4, start_node4, target_node4)
        distance4 = distances4.get(target_node4, math.inf)

        if path4:
            print(f"  {start_node4} to {target_node4}: {' -> '.join(path4)} (Distance: {distance4 if distance4 != math.inf else 'Infinity'})")
        else:
            print(f"  {start_node4} to {target_node4}: No path found (Distance: {distance4 if distance4 != math.inf else 'Infinity'})")
    except ValueError as e:
        print(f"Error: {e}")

```