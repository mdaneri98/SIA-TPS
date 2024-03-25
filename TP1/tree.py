class Node(object):
    """Represents a node in a tree"""
    def __init__(self, state, children = None, parent = None, depth = 0):
        self.parent = parent
        self.children = []
        self.depth = depth
        self.state = state
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
    
    def state(self):
        return self.state
    
    def add_child(self, state):
        newNode = Node(state, parent = self, depth = self.depth + 1)
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
    
    def __str__(self):
        return f"Node: {self.state}"

class Tree(object):
    """Represents a tree"""
    def __init__(self, initialstate):
        self.root = Node(initialstate)
    
    def get_root(self):
        return self.root
    