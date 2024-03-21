class Node(object):
    """Represents a node in a tree"""
    def __init__(self, gameState, children = None, parent = None, depth = 0):
        self.parent = parent
        self.children = []
        self.depth = depth
        self.gameState = gameState
        if children is not None:
            for child in children:
                self.add_child(child)
            
    def is_root(self):
        return self.parent is None
    
    def is_leaf(self):
        return len(self.children) == 0
    
    def depth(self):
        return self.depth
    
    def get_depth(self):
        return self.depth
    
    def game_state(self):
        return self.gameState
    
    def add_child(self, gameState):
        newNode = Node(gameState, parent = self, depth = self.depth + 1)
        self.children.append(newNode)
        return newNode
    
    def remove_child(self, node):
        assert isinstance(node, Node)
        node.parent = None
        self.children.remove(node)

    def get_root_path(self, node):
        path = []
        while node.parent is not None:
            path.append(node)
            node = node.parent
        path.append(node)
        path.reverse()
        return path
    
    def print_game_state(self, boardMatrix, goals):
       return self.gameState.printGameState(boardMatrix, goals)

class Tree(object):
    """Represents a tree"""
    def __init__(self, initialGameState):
        self.root = Node(initialGameState)
    
    def get_root(self):
        return self.root
    