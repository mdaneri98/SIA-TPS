from State import State
from tree import Node, Tree
from collections import deque
import soko


def bfs(state, board):
    visited_states = set()
    frontier_nodes = deque()

    tree = Tree(state)
    root = tree.get_root()
    visited_states.add(state)
    frontier_nodes.append(root)

    # Vemos los, hasta cuatro, movimientos posibles
    movimientos = [(0, 1), (1, 0), (-1, 0), (0, -1)]

    while frontier_nodes:
        current_node = frontier_nodes.popleft()
        current_state = current_node.state

        if current_state.is_finished():
            return current_node.get_root_path(current_node), current_node.get_depth(), len(visited_states), len(
                frontier_nodes)

        for mov in movimientos:
            if soko.puede_moverse(board, current_state.playerPos, current_state.goalsPos, current_state.boxesPos, mov):
                new_playerPos, new_boxesPos = soko.moverse(board, current_state.playerPos, current_state.goalsPos,
                                                           current_state.boxesPos, mov)

                new_state = State(new_playerPos, new_boxesPos, current_state.goalsPos)

                if new_state not in visited_states:
                    visited_states.add(new_state)
                    next_node = current_node.add_child(new_state)
                    frontier_nodes.append(next_node)

        # Suponiendo que tienes una función print_board en State que acepta el tablero regenerado
        # print_board = soko.regenerate(board, current_state.playerPos, current_state.goalsPos, current_state.boxesPos)
        # current_state.print_board(print_board)

    # Cambio para cuando no se encuentra una solución
    return [], 0, len(visited_states), len(frontier_nodes)


def dfs(state, board):
    visited_states = set()
    frontier_nodes = deque()

    tree = Tree(state)
    root = tree.get_root()
    visited_states.add(state)
    frontier_nodes.append(root)

    # Vemos los, hasta cuatro, movimientos posibles
    movimientos = [(0, 1), (1, 0), (-1, 0), (0, -1)]

    while frontier_nodes:
        current_node = frontier_nodes.pop()
        current_state = current_node.state

        if current_state.is_finished():
            return current_node.get_root_path(current_node), current_node.get_depth(), len(visited_states), len(
                frontier_nodes)

        for mov in movimientos:
            if soko.puede_moverse(board, current_state.playerPos, current_state.goalsPos, current_state.boxesPos, mov):
                new_playerPos, new_boxesPos = soko.moverse(board, current_state.playerPos, current_state.goalsPos,
                                                           current_state.boxesPos, mov)

                new_state = State(new_playerPos, new_boxesPos, current_state.goalsPos)

                if new_state not in visited_states:
                    visited_states.add(new_state)
                    next_node = current_node.add_child(new_state)
                    frontier_nodes.append(next_node)

    # Cambio para cuando no se encuentra una solución
    return [], 0, len(visited_states), len(frontier_nodes)


def greedy(state: State, board: list[list[str]], heuristic):
    visited_states = set()
    frontier_nodes = []

    tree = Tree(state)
    root = tree.get_root()
    visited_states.add(state)
    frontier_nodes.append((root, heuristic(state, board)))

    movimientos = [(0, -1), (-1, 0), (1, 0), (0, 1)]

    while len(frontier_nodes) > 0:
        current_node, _heuristic = frontier_nodes.pop(0)
        current_state = current_node.state

        if current_state.is_finished():
            return current_node.get_root_path(current_node), current_node.get_depth(), len(visited_states), len(
                frontier_nodes)

        for mov in movimientos:
            if soko.puede_moverse(board, current_state.playerPos, current_state.goalsPos, current_state.boxesPos, mov):
                new_playerPos, new_boxesPos = soko.moverse(board, current_state.playerPos, current_state.goalsPos,
                                                           current_state.boxesPos, mov)

                new_state = State(new_playerPos, new_boxesPos, current_state.goalsPos)

                if new_state not in visited_states:
                    visited_states.add(new_state)
                    next_node = current_node.add_child(new_state)
                    frontier_nodes.append((next_node, heuristic(new_state, board)))

        frontier_nodes.sort(key=lambda x: x[1])
    return [], 0, len(visited_states), len(frontier_nodes)


def astar(initialState, boardMatrix, heuristic):
    exploredStates = set()
    frontierNodes = []

    tree = Tree(initialState)
    root = tree.get_root()
    exploredStates.add(initialState)
    frontierNodes.append((root, 0, heuristic(initialState, boardMatrix)))

    while len(frontierNodes) > 0:
        currentNode, _, _ = frontierNodes.pop(0)
        currentState = currentNode.state

        if currentState.is_finished():
            return currentNode.get_root_path(currentNode), currentNode.get_depth(), len(exploredStates), len(
                frontierNodes)

        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # Lista de direcciones posibles
        for direction in directions:
            if soko.puede_moverse(boardMatrix, currentState.playerPos, currentState.goalsPos, currentState.boxesPos, direction):
                new_playerPos, new_boxesPos = soko.moverse(boardMatrix, currentState.playerPos, currentState.goalsPos,
                                                           currentState.boxesPos, direction)
                new_state = State(new_playerPos, new_boxesPos, currentState.goalsPos)

                if new_state not in exploredStates:
                    exploredStates.add(new_state)
                    next_node = currentNode.add_child(new_state)
                    h = heuristic(new_state, boardMatrix)
                    f = next_node.depth + h
                    frontierNodes.append((next_node, f, h))

        frontierNodes.sort(key=lambda x: x[1])

    return [], 0, len(exploredStates), len(frontierNodes)


def manhattan_heuristic(state, board):
    total_distance = 0  # Acumula la distancia total para todas las cajas
    for box_pos in state.boxesPos:
        min_distance = float('inf')  # Inicializar con infinito para encontrar el mínimo
        for goal_pos in state.goalsPos:
            distance = abs(box_pos[0] - goal_pos[0]) + abs(box_pos[1] - goal_pos[1])
            min_distance = min(min_distance, distance)
        total_distance += min_distance  # Acumular la distancia mínima para esta caja

    return total_distance


def closest_target_distance(board, x, y):
    min_distance = float('inf')
    for i, row in enumerate(board.board):
        for j, cell in enumerate(row):
            if cell == '.':
                distance = abs(x - i) + abs(y - j)
                min_distance = min(min_distance, distance)
    return min_distance


def combined_heuristic(state, board):
    distance = 0
    for box in state.boxesPos:
        min_distance_with_turns = float('inf')
        for goal in state.goalsPos:
            manhattan_distance = abs(box[0] - goal[0]) + abs(box[1] - goal[1])
            turns_needed = 0 if (box[0] - goal[0]) * (box[1] - goal[1]) == 0 else 1
            distance_with_turns = manhattan_distance + turns_needed * 2
            min_distance_with_turns = min(min_distance_with_turns, distance_with_turns)
        distance += min_distance_with_turns
    return distance
