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
            return current_node.get_root_path(current_node), current_node.get_depth(), len(visited_states), len(frontier_nodes)
        
        for mov in movimientos: 
            if soko.puede_moverse(board, current_state.playerPos, current_state.goalsPos, current_state.boxesPos, mov):
                new_playerPos, new_boxesPos = soko.moverse(board, current_state.playerPos, current_state.goalsPos, current_state.boxesPos, mov)

                new_state = State(new_playerPos, new_boxesPos, current_state.goalsPos)

                if new_state not in visited_states:
                    visited_states.add(new_state)
                    next_node = current_node.add_child(new_state)
                    frontier_nodes.append(next_node)

        # Suponiendo que tienes una función print_board en State que acepta el tablero regenerado
        #print_board = soko.regenerate(board, current_state.playerPos, current_state.goalsPos, current_state.boxesPos)
        #current_state.print_board(print_board)

    # Cambio para cuando no se encuentra una solución
    return None, 0, len(visited_states), len(frontier_nodes)


def manhattan_heuristic(board):
    distance = 0
    for i, row in enumerate(board.board):  # Acceder a board.board en lugar de board
        for j, cell in enumerate(row):
            if cell == '*':
                distance += closest_target_distance(board, i, j)
    return distance


def closest_target_distance(board, x, y):
    min_distance = float('inf')
    for i, row in enumerate(board.board):
        for j, cell in enumerate(row):
            if cell == '.':
                distance = abs(x - i) + abs(y - j)
                min_distance = min(min_distance, distance)
    return min_distance

    
def combined_heuristic(board):
    goals = [(i, j) for i, row in enumerate(board.board) for j, cell in enumerate(row) if cell == '.']
    return combined(board, goals)


def combined(board, goals):
    distance = 0
    for box in board.get_boxes_positions():
        min_distance_with_turns = float('inf')
        for goal in goals:
            manhattan_distance = abs(box[0] - goal[0]) + abs(box[1] - goal[1])
            turns_needed = 0 if (box[0] - goal[0]) * (box[1] - goal[1]) == 0 else 1
            distance_with_turns = manhattan_distance + turns_needed * 2
            min_distance_with_turns = min(min_distance_with_turns, distance_with_turns)
        distance += min_distance_with_turns
    return distance
