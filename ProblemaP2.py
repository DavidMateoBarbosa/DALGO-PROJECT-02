import math
import sys
from collections import defaultdict, deque
from pprint import pprint

SOURCE = object()
INITIATOR = object()
CALCULATOR = object()
EXECUTOR = object()
SINK = object()
TYPES = {'1': INITIATOR, '2': CALCULATOR, '3': EXECUTOR}

RADIUS = 0


def maketuple(data):
    components = data.split()
    return int(components[0]), {'position': (int(components[1]), int(components[2])),
                                'type': TYPES[components[3]],
                                'peptides': set(components[4:])}


def connected(origin, target):
    if origin['type'] is SOURCE and target['type'] is INITIATOR:
        return True
    if origin['type'] is EXECUTOR and target['type'] is SINK:
        return True
    if (origin['type'], target['type']) in {
        (INITIATOR, CALCULATOR),
        (CALCULATOR, CALCULATOR),
        (CALCULATOR, EXECUTOR),
    }:
        x = (origin['position'][0] - target['position'][0]) ** 2
        y = (origin['position'][1] - target['position'][1]) ** 2
        return 0 < x + y <= RADIUS ** 2
    return False


def capacity(origin, target):
    if (origin['type'], target['type']) in {
        (SOURCE, INITIATOR),
        (EXECUTOR, SINK)
    }:
        return len(origin['peptides'] or target['peptides'])
    return len(origin['peptides'] & target['peptides'])


# Complexity Analysis:
# O(V²E + K ⋅ F ⋅ E)
# V: Number of vertices (nodes).
# E: Number of edges.
# F: Maximum flow value.
# K: Number of calculator nodes.
# Worst-Case Complexity: O(V⁴ + F ⋅ V³)  # Nearly impossible due to sparse connectivity in practical cases.
def solver(cells, flownet, source, sink):
    """
    Calculates the maximum flow in a network and evaluates the impact of removing each calculator node.

    This function uses the Push-Relabel algorithm to compute the maximum flow and
    evaluates the flow reduction if a specific "calculator" node is removed. For each node,
    the flow is recalculated using a modified Edmonds-Karp algorithm.

    :param cells:
    :return: ...
    """

    # Initialize data structures for Push-Relabel
    def initialize():
        """
        Sets up the initial residual graph, flows, excesses, and heights for the Push-Relabel algorithm.
        Pushes initial preflows from the source node.
        """
        nonlocal residuals, excesses, heights, nodes, flows
        heights[source] = len(flownet)  # Initial height of the source
        for target in residuals[source]:
            delta = residuals[source][target]
            # Push preflow from source to all directly connected nodes
            flows[source][target] += delta
            flows[target][source] -= delta
            residuals[source][target] -= delta
            residuals[target][source] += delta
            excesses[target] += delta
            excesses[source] -= delta
            # Add active nodes to queue (excluding source and sink)
            if target != source and target != sink:
                nodes.append(target)

    def push(origin, target):
        """
        Pushes flow from `origin` to `target` if possible, updating residual capacities and excess flows.
        """
        nonlocal residuals, excesses, heights, nodes, flows
        if residuals[origin][target] <= 0 or excesses[origin] <= 0:
            return
        delta = min(excesses[origin], residuals[origin][target])
        # Update flow and residual capacities
        flows[origin][target] += delta
        flows[target][origin] -= delta
        residuals[origin][target] -= delta
        residuals[target][origin] += delta
        # Update excess flows
        excesses[origin] -= delta
        excesses[target] += delta
        # Add target to the queue if it becomes active
        if target != source and target != sink and target not in nodes:
            nodes.append(target)

    def relabel(node):
        """
        Increases the height of a node to allow for further pushes.
        """
        minimum = float('inf')
        for neighbor in residuals[node]:
            if residuals[node][neighbor] - flows[node][neighbor] > 0:
                minimum = min(minimum, heights[neighbor])
        heights[node] = minimum + 1

    def discharge(node):
        """
        Processes a node by repeatedly pushing flow to its neighbors or relabeling it.
        """
        while excesses[node] > 0:
            for neighbor in residuals[node]:
                if residuals[node][neighbor] > 0 and heights[node] > heights[neighbor]:
                    push(node, neighbor)
                    if excesses[node] == 0:
                        break
            else:
                relabel(node)

    # Residual graph, flow, and helper structures
    residuals = {origin: {target: flownet[origin].get(target, 0) for target in flownet} for origin in flownet}
    excesses = {id: 0 for id in flownet}
    heights = {id: 0 for id in flownet}
    flows = defaultdict(lambda: defaultdict(int))
    nodes = deque()  # Active nodes queue for Push-Relabel
    initialize()

    # Execute Push-Relabel algorithm
    while nodes:
        discharge(nodes.popleft())

    def reduce(id):
        """
        Calculates the maximum flow in the network after removing node `id`.

        Parameters:
        ----------
        id : Any
            Node ID to remove from the flow network.

        Returns:
        -------
        int
            Maximum flow in the modified network.
        """
        # Create a modified flow network without the node `id`
        modified_flow_network = defaultdict(lambda: defaultdict(int))
        for source_node in flownet:
            if source_node == id:
                continue
            for target_node in flownet[source_node]:
                if target_node == id:
                    continue
                modified_flow_network[source_node][target_node] = flownet[source_node][target_node]

        total_flow = 0
        while True:
            visited_nodes = set()
            bfs_queue = deque([source])
            parent_map = {source: None}

            # BFS to find an augmenting path
            while bfs_queue:
                current_node = bfs_queue.popleft()
                for neighbor_node in modified_flow_network[current_node]:
                    if neighbor_node not in visited_nodes and modified_flow_network[current_node][neighbor_node] > 0:
                        visited_nodes.add(neighbor_node)
                        parent_map[neighbor_node] = current_node
                        bfs_queue.append(neighbor_node)
                        if neighbor_node == sink:
                            break

            # If we found an augmenting path, calculate the bottleneck capacity
            if sink in parent_map:
                bottleneck_capacity = float('inf')
                node = sink
                while node != source:
                    bottleneck_capacity = min(bottleneck_capacity, modified_flow_network[parent_map[node]][node])
                    node = parent_map[node]
                total_flow += bottleneck_capacity

                # Update residual capacities along the augmenting path
                node = sink
                while node != source:
                    previous_node = parent_map[node]
                    modified_flow_network[previous_node][node] -= bottleneck_capacity
                    modified_flow_network[node][previous_node] += bottleneck_capacity
                    node = previous_node
            else:
                break

        return total_flow

    # Calculate flow reductions for each calculator node
    reductions = {id: reduce(id) for id in cells if cells[id]['type'] is CALCULATOR}
    id = min(reductions, key=reductions.get)  # Find node with maximum impact
    maxflow = sum(flows[source][vertex] for vertex in flownet[source])
    reduction = reductions[id]
    return id, maxflow, reduction


def solution(cells):
    cells[0] = {'position': None, 'type': SOURCE, 'peptides': set()}
    cells[len(cells)] = {'position': None, 'type': SINK, 'peptides': set()}
    flownet = {origin: {target: capacity(cells[origin], cells[target]) for target in filter(lambda target: origin != target and connected(cells[origin], cells[target]), cells)} for origin in cells}
    return solver(cells, flownet, 0, len(cells) - 1)


def main():
    global RADIUS
    # sys.stdin = open('main.in')
    cases = int(sys.stdin.readline())
    for _ in range(cases):
        cellcount, RADIUS = map(int, sys.stdin.readline().split())
        cells = dict(map(maketuple, (sys.stdin.readline().strip() for _ in range(cellcount))))
        sys.stdout.write(' '.join(map(str, solution(cells))))
        sys.stdout.write('\n')
    # sys.stdin.close()


if __name__ == '__main__':
    main()
