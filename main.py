import collections
import sys
import time

SOURCE = object()
INITIATOR = object()
CALCULATOR = object()
EXECUTOR = object()
TARGET = object()
_types = {'1': INITIATOR,
          '2': CALCULATOR,
          '3': EXECUTOR}


def parse(string):
    """
    Parses a string representing a cell's information into a dictionary with
    details about the cell.

    :param string: A string containing the cell's ID, x and y coordinates, type, and the list of peptides.
    :return: A dictionary with keys 'id', 'position', 'type', and 'peptides'.
    """
    lst = string.split()
    return {'id': int(lst[0]),
            'position': (int(lst[1]), int(lst[2])),
            'type': _types[lst[3]],
            'peptides': set(lst[4:])}


def connected(source, target, distance):
    """
    Determines if two cells are directly connected based on their types and the
    maximum allowed distance.

    :param source: Dictionary representing the source cell.
    :param target: Dictionary representing the target cell.
    :param distance: Maximum distance allowed between cells to be considered connected.
    :return: True if the target cell is reachable from the source cell and satisfies the conditions; False otherwise.
    """
    if source['type'] is SOURCE and target['type'] is INITIATOR:
        return True
    if source['type'] is EXECUTOR and target['type'] is TARGET:
        return True
    if source['type'] is SOURCE and target['type'] is not INITIATOR:
        return False
    if source['type'] is not EXECUTOR and target['type'] is TARGET:
        return False
    if source['type'] is EXECUTOR or target['type'] is INITIATOR:
        return False
    if source['type'] is TARGET or target['type'] is SOURCE:
        return False
    x = (source['position'][0] - target['position'][0]) ** 2
    y = (source['position'][1] - target['position'][1]) ** 2
    return 0 < x + y <= distance ** 2


def capacity(source, target):
    """
    Computes the number of shared peptides between two cells.

    :param source: Dictionary representing the source cell.
    :param target: Dictionary representing the target cell.
    :return: An integer representing the number of shared peptides.
    """
    if source['type'] is SOURCE and target['type'] is INITIATOR:
        return float('inf')
    if source['type'] is EXECUTOR and target['type'] is TARGET:
        return float('inf')
    return len(source['peptides'] & target['peptides'])


def bfs(source, graph, target, parents):
    queue = collections.deque([source])
    parents.clear()
    parents[source] = None
    while queue:
        current = queue.popleft()
        for neighbor in graph[current]:
            if neighbor not in parents and graph[current][neighbor] > 0:
                parents[neighbor] = current
                if neighbor == target:
                    return True
                queue.append(neighbor)
    return False


def edmonds(graph, source, target, info):
    maxflow = 0
    parents = {}
    flows = collections.defaultdict(int)

    while bfs(source, graph, target, parents):
        flow = float('inf')
        temp = target
        while temp != source:
            current = parents[temp]
            flow = min(flow, graph[current][temp])
            temp = current

        temp = target
        while temp != source:
            current = parents[temp]
            graph[current][temp] -= flow
            graph[temp][current] += flow
            temp = current

        temp = target
        while temp != source:
            parent = parents[temp]
            if info[parent]['type'] is CALCULATOR and info[temp]['type'] is EXECUTOR:
                flows[parent] += flow
                break
            temp = parent

        maxflow += flow
    # This part is incorrect or is not perfect (teacher's words)
    cell = max(flows, key=flows.get)
    return cell, maxflow, maxflow - flows[cell]


def resolve(cells, distance):
    """
    Determines which calculator cell to block to maximize the reduction in
    message transmissions.

    :param cells: List of dictionaries, each representing a cell's details.
    :param distance: Maximum distance allowed for message transmission.
    :return: A tuple (id, total, reduced)
    """
    cells += [
        {
            'id': 0,
            'position': None,
            'type': SOURCE,
            'peptides': None,
        },
        {
            'id': len(cells) + 1,
            'position': None,
            'type': TARGET,
            'peptides': None
        },
    ]  # Add virtual cells to make a flownet (instead of n-sources (initiators) and n-sinks (executors)).
    graph = collections.defaultdict(lambda: collections.defaultdict(int))
    for origin in cells:
        for target in cells:
            if connected(origin, target, distance):
                if origin['type'] is CALCULATOR and target['type'] is CALCULATOR and graph[target['id']][origin['id']] > 0:
                    graph[origin['id']][(origin['id'] + target['id']) / 2] = graph[(origin['id'] + target['id']) / 2][target['id']] = capacity(origin, target)
                else:
                    graph[origin['id']][target['id']] = capacity(origin, target)
    return edmonds(graph, cells[-2]['id'], cells[-1]['id'], {cell['id']: cell for cell in cells})


def main():
    """
    Main function to read input, process each test case, and output the results.

    :return: None
    """
    cases = int(sys.stdin.readline())
    for _ in range(cases):
        amount, distance = map(int, sys.stdin.readline().split())
        cells = list(map(parse, (sys.stdin.readline() for _ in range(amount))))
        sys.stdout.write(' '.join(map(str, resolve(cells, distance))))
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

"""  # NOQA
