import math
import sys
from collections import defaultdict, deque

# Cell types
SOURCE = object()  # Represents the virtual source node, used once per problem
INITIATOR = object()
CALCULATOR = object()
EXECUTOR = object()
SINK = object()  # Represents the virtual sink node, used once per problem

# Mapping cell type identifiers to type objects
CELL_TYPES = {
    '1': INITIATOR,
    '2': CALCULATOR,
    '3': EXECUTOR
}

RADIUS = 0  # Global variable controlling the maximum connection distance


def parse(data):
    """
    Parses a single line of data describing a cell into a structured dictionary.

    :param data: A string containing the cell details, formatted as:
                 'id x y type peptides...'
                 where:
                    - id: Unique identifier for the cell.
                    - x, y: Integer coordinates of the cell's position.
                    - type: A string ('1', '2', '3') representing the cell type.
                    - peptides: Optional strings representing the peptides associated with the cell.
    :return: A dictionary containing parsed cell information:
             {
                 'id': str,            # Cell ID
                 'position': (int, int),  # Cell position (x, y)
                 'type': object,       # Cell type object (e.g., INITIATOR, CALCULATOR, etc.)
                 'peptides': set       # Set of peptides associated with the cell
             }
    :raises ValueError: If the data format is invalid.
    """
    components = data.split()
    return {
        'id': components[0],
        'position': (int(components[1]), int(components[2])),
        'type': CELL_TYPES[components[3]],
        'peptides': set(components[4:])
    }


def connected(origin, target):
    """
    Determines if two cells are connected based on their types and distance.

    :param origin: Dictionary containing the details of the origin cell.
    :param target: Dictionary containing the details of the target cell.
    :return: True if the cells are connected, False otherwise.
    """
    global RADIUS

    # Direct connections based on type rules
    if origin['type'] is SOURCE and target['type'] is INITIATOR:
        return True
    if origin['type'] is EXECUTOR and target['type'] is SINK:
        return True
    if (origin['type'], target['type']) in {
        (INITIATOR, CALCULATOR),
        (CALCULATOR, EXECUTOR),
        (CALCULATOR, CALCULATOR),
    }:
        # Allow based on Euclidean distance
        x = (origin['position'][0] - target['position'][0]) ** 2
        y = (origin['position'][1] - target['position'][1]) ** 2
        return 0 < x + y <= RADIUS ** 2
    return False


def capacity(origin, target):
    """
    Determines the capacity of the connection between two cells based on their types
    and shared peptides.

    :param origin: Dictionary containing the details of the origin cell.
    :param target: Dictionary containing the details of the target cell.
    :return: Capacity as an integer (number of shared peptides) or infinity for special cases.
    """
    # Special cases with infinite capacity
    if origin['type'] is SOURCE and target['type'] is INITIATOR:
        return math.inf
    if origin['type'] is EXECUTOR and target['type'] is SINK:
        return math.inf

    # If either cell has no peptides, capacity is 0
    if not origin['peptides'] or not target['peptides']:
        return 0

    # Calculate capacity based on the number of shared peptides
    return len(origin['peptides'] & target['peptides'])


def augmenting(graph, source, sink, parents, node_flows):
    """
    Finds an augmenting path in the residual graph using BFS and tracks flow through CALCULATOR nodes.

    :param graph: Residual graph as a defaultdict of defaultdicts.
    :param source: Source node ID.
    :param sink: Sink node ID.
    :param parents: Dictionary to store the augmenting path.
    :param node_flows: Dictionary to track flow through CALCULATOR nodes.
    :return: True if an augmenting path is found, False otherwise.
    """
    parents.clear()
    parents[source] = None
    queue = deque([source])
    while queue:
        current = queue.popleft()
        for neighbor in graph[current]:
            if neighbor not in parents and graph[current][neighbor] > 0:
                parents[neighbor] = current
                if neighbor == sink:
                    # Track flow passing through CALCULATOR nodes
                    temp = sink
                    while temp != source:
                        parent = parents[temp]
                        if parent in node_flows:
                            node_flows[parent] += graph[parent][temp]
                        temp = parent
                    return True
                queue.append(neighbor)
    return False


def solver(graph, source, sink, information):
    """
    Computes the maximum flow in the flow network and tracks the flow impact of CALCULATOR nodes.

    :param graph: Residual graph as a defaultdict of defaultdicts.
    :param source: Source node ID.
    :param sink: Sink node ID.
    :param information: Dictionary containing cell information.
    :return: Tuple (best CALCULATOR node to block, max flow, reduced flow).
    """
    parents = {}
    maxflow = 0
    node_flows = {node_id: 0 for node_id, cell in information.items() if cell['type'] == CALCULATOR}

    while augmenting(graph, source, sink, parents, node_flows):
        # Find bottleneck flow
        flow = float('inf')
        temp = sink
        while temp != source:
            parent = parents[temp]
            flow = min(flow, graph[parent][temp])
            temp = parent

        # Update residual capacities
        temp = sink
        while temp != source:
            parent = parents[temp]
            graph[parent][temp] -= flow
            graph[temp][parent] += flow
            temp = parent

        maxflow += flow

    # Evaluate node impacts
    best_node = max(node_flows, key=node_flows.get, default=None)
    reduced_flow = node_flows[best_node] if best_node else 0

    return best_node, maxflow, reduced_flow


def build_graph(cells):
    """
    Constructs the residual graph from the given cells.

    :param cells: List of cell dictionaries.
    :return: Residual graph as a defaultdict of defaultdicts.
    """
    graph = defaultdict(lambda: defaultdict(int))
    for origin in cells:
        for target in cells:
            if origin['id'] == target['id'] or not connected(origin, target):
                continue
            graph[origin['id']][target['id']] = capacity(origin, target)
    return graph


def solution(cells):
    """
    Constructs the flow network and solves the problem for the given cells.

    :param cells: List of dictionaries, each representing a cell with its attributes.
    :return: Tuple of (blocked cell id, total messages, reduced messages).
    """
    # Add virtual source and sink nodes
    cells.reverse()
    cells.append({'id': 'source', 'position': None, 'type': SOURCE, 'peptides': None})
    cells.append({'id': 'sink', 'position': None, 'type': SINK, 'peptides': None})

    # Build the graph
    graph = build_graph(cells)

    # Solve the problem
    return solver(graph, 'source', 'sink', {cell['id']: cell for cell in cells})


def main():
    """
    Main function to read input, solve each test case, and output results.
    Assumes the input is always well-formed and adheres to the problem constraints.
    """
    global RADIUS
    # Read the number of test cases
    cases = int(sys.stdin.readline())

    for _ in range(cases):
        # Read the number of cells and the connection radius
        cellcount, RADIUS = map(int, sys.stdin.readline().split())

        # Parse the cell data for this case
        cells = list(map(parse, (sys.stdin.readline().strip() for _ in range(cellcount))))

        # Solve the problem and write the output
        sys.stdout.write(' '.join(map(str, solution(cells))))
        sys.stdout.write('\n')


if __name__ == '__main__':
    main()

"""
3
7 1
1 0 0 1 AETQT DFTYA PHLYT
2 0 2 1 DSQTS IYHLK LHGPS LTLLS
3 1 0 2 AETQT DFTYA HGCYS LSVGG SRFNH
4 1 1 2 DFTYA HGCYS IYHLK SRFNH
5 1 2 2 DSQTS IYHLK LSVGG LTLLS TTVTG
6 2 1 3 AETQT HGCYS IYHLK LSVGG LTLLS
7 2 2 3 HGCYS SRFNH TTVTG
7 2
1 0 0 1 AETQT DFTYA PHLYT
2 0 2 1 DSQTS IYHLK LHGPS LTLLS
3 1 0 2 AETQT DFTYA HGCYS LSVGG SRFNH
4 1 1 2 DFTYA HGCYS IYHLK SRFNH
5 1 2 2 DSQTS IYHLK LSVGG LTLLS TTVTG
6 2 1 3 AETQT HGCYS IYHLK LSVGG LTLLS
7 2 2 3 HGCYS SRFNH TTVTG
4 1
1 0 0 1 AETQT DFTYA
2 0 1 2 AETQT HGCYS
3 1 0 2 DFTYA IYHLK
4 1 1 3 HGCYS IYHLK
"""  # NOQA: Possible misspellings (remove in final version)
