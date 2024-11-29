import collections
import itertools
import sys

# Define a named tuple for neuron data
NeuronType = collections.namedtuple('NeuronType', ('position', 'peptides'))

# Global variable to store the maximum distance for connectivity
MAXLENGTH = 0


# O(1)
def parse(data):
    """
    Parse a line of input into a neuron identifier and its properties.

    This function processes a string input representing a neuron. The input string contains:

    - The neuron ID (integer).
    - The position of the neuron as two integers (x, y).
    - A list of peptides associated with the neuron.

    :param data: A string containing the neuron ID, position (x, y), and a list of peptides.
    :return: A tuple containing: An integer ID of the neuron and the NeuronType object.
    """
    components = data.split()
    return int(components[0]), NeuronType(position=(int(components[1]), int(components[2])),
                                          peptides=set(components[3:]))

# O(1)
def connected(origin, target) -> bool:
    """
    Determine if two neurons are connected based on their positions and the global MAXLENGTH.

    Two neurons are considered connected if the Euclidean distance between their positions is greater than 0 and less
    than or equal to MAXLENGTH. This excludes cases where the neurons occupy the same position (superposition).

    :param origin: A NeuronType object representing the origin neuron.
    :param target: A NeuronType object representing the target neuron.
    :return: True if the neurons are connected; False otherwise.
    """
    # Calculate the squared differences in the x and y coordinates.
    x = (origin.position[0] - target.position[0]) ** 2
    y = (origin.position[1] - target.position[1]) ** 2

    # Check if the squared Euclidean distance is within the allowable range.
    return 0 < x + y <= MAXLENGTH ** 2


# O(1)
def weight(origin, target):
    """
    Compute the weight of the connection between two neurons.

    The weight is defined as the number of common peptides between the two neurons.
    A higher weight indicates a stronger connection between the neurons.

    :param origin: A NeuronType object representing the origin neuron. It must have a 'peptides' attribute,
                   which is a set of peptides associated with the neuron.
    :param target: A NeuronType object representing the target neuron. It must also have a 'peptides' attribute,
                   which is a set of peptides.
    :return: An integer representing the size of the intersection between the 'peptides' sets of the origin and
             target neurons.
    """
    # Calculate and return the size of the intersection of peptides sets.
    return len(origin.peptides & target.peptides)


# O(V³)
def solution(cells):
    """
    Groups connected components of cells into cliques based on conditions.

    This algorithm constructs a graph from the given cells and iteratively groups connected nodes
    into cliques. Cliques are formed by checking if all nodes within a group are fully connected.
    The goal is to partition the graph into the smallest number of fully connected components (cliques).

    :param cells: A dictionary where keys are cell IDs and values are properties of the cells. Each cell ID
                  represents a unique node in the graph.
    :return: A generator producing strings of the format 'id group', where 'id' is the cell ID and 'group' is the
             clique group ID.
    """

    # O(V): Initialize a dictionary to store group assignments for each cell.
    # Keys are cell IDs, and values will be assigned clique group IDs later.
    groups = dict.fromkeys(cells)

    # O(V²): Build a graph where each node (cell) connects to its neighbors if they satisfy
    # the `connected` condition and have a positive `weight`.
    # The graph is represented as an adjacency list (dictionary of sets).
    graph = {x: {y for y in cells if connected(cells[x], cells[y]) and weight(cells[x], cells[y]) > 0} for x in cells}

    # Set of nodes that have already been assigned to a clique.
    grouped = set()

    # Counter to keep track of the current clique group ID.
    group = 0

    def gather():
        """
        Iteratively adds nodes to the current clique group.

        This function identifies ungrouped nodes and attempts to form a clique by verifying
        if each node can connect to all other nodes in the current group. The nodes are processed
        in descending order of their degree (number of connections) to maximize clique size.
        """
        nonlocal grouped, groups, graph, group

        # O(V · log₂(V)): Filter out already grouped nodes and sort the remaining nodes
        # by their degree (number of connections) in descending order.
        vertexes = sorted(itertools.filterfalse(grouped.__contains__, graph), key=lambda x: len(graph[x]), reverse=True)

        # O(V²): Iterate through the sorted nodes and attempt to add them to the current group.
        for vertex in vertexes:
            # Check if the vertex can be part of the current clique group.
            # This is true if it connects to all nodes already in the group.
            if all(neighbour in graph[vertex] for neighbour, clique in groups.items() if clique == group):
                # Assign the vertex to the current group and mark it as grouped.
                groups[vertex] = group
                grouped.add(vertex)

    # O(V) · O(V²) ≈ O(V³): Repeat the process until all nodes are grouped.
    while len(grouped) != len(groups):
        # Start a new group for the next iteration.
        group += 1
        gather()

    # O(1): Generate the result as a stream of strings in the format 'id group'.
    # Each line corresponds to a node and its assigned clique group.
    yield from (f'{id} {group}' for id, group in groups.items())


def main():
    """
    Main function to process multiple test cases of neuron connectivity.

    The function reads input data, processes each test case, and outputs the results.
    Each test case specifies a number of neurons and their properties, and the function
    determines the group (clique) assignment for each neuron based on connectivity rules.
    """
    global MAXLENGTH  # Define MAXLENGTH as global to use it in the `connected` function

    # Read the number of test cases
    cases = int(sys.stdin.readline().strip())

    # Process each test case
    for _ in range(cases):
        # Read cell count and maximum connection distance
        cellcount, MAXLENGTH = map(int, sys.stdin.readline().strip().split())

        # Parse cell data into a dictionary
        # Each key is the cell ID, and the value is a NeuronType object
        cells = dict(map(parse, (sys.stdin.readline().strip() for _ in range(cellcount))))

        # Solve the problem for the current test case and collect results
        results = '\n'.join(solution(cells))

        # Output results for this test case
        sys.stdout.write(results)
        sys.stdout.write('\n')  # Ensure results are followed by a single newline


# Entry point of the script
if __name__ == '__main__':
    # Ensure the script runs only when executed directly, not when imported as a module
    # Call the main() function to process the input and handle the logic
    main()
