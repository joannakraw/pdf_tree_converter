import matplotlib.pyplot as plt
import numpy as np
from Bio import Phylo
from io import StringIO

class Leaf:
    def __init__(self, coordinates, name):
        self.left = None
        self.right = None
        self.coordinates = coordinates
        self.name = name


class BinNode:
    def __init__(self, left, right, coordinates, name=None):
        self.coordinates = coordinates
        self.left = left
        self.right = right
        self.name = name

    def generate_newick_node(self, newick=''):
        """
        Generates newick string for a node
        """
        if self.left != None:
            newick += '('
            if isinstance(self.left, Leaf):
                newick += self.left.name
            else:
                newick = self.left.generate_newick_node(newick=newick)

        newick += ','

        if self.right != None:
            if not isinstance(self.right, Leaf):
                newick = self.right.generate_newick_node(newick=newick)
            else:
                newick += self.right.name
            newick += ')'

        return newick


class BinTree:
    def __init__(self, node=None):
        self.root = node

    def generate_newick(self):
        """
        Generates newick string for a tree
        """
        if self.root != None:
            newick = self.root.generate_newick_node()
        else:
            newick = ''
        return newick


def leaves_list2class(leaves: list, labels: list = None) -> list:
    """
    Converts list of leaves coordinates to list of Leaf objects
    """
    leaves = sorted(leaves)
    if not labels:
        return [Leaf(leaf, f'leaf_{str(leaf[0]).split(".")[0]}_{str(leaf[1]).split(".")[0]}') for i, leaf in
                enumerate(leaves)]
    else:
        assert len(leaves) == len(labels), "Leaves and labels must have the same length"
        return [Leaf(leaf, label) for leaf, label in zip(leaves, labels)]


def find_the_nearest_leaves(leaves: set, node: tuple) -> tuple:
    """
    Find the nearest (on the x-axis) leaves between which a given node is located.
    Args:
        leaves (set): set of leaves to search
        node (tuple): node coordinates
    Returns:
        tuple: nearest left leaf, nearest right leaf
    """
    x, _ = node
    nearest_left = None
    nearest_right = None
    for leaf in leaves:
        if leaf.coordinates[0] < x:
            if nearest_left is None:
                nearest_left = leaf
            elif nearest_left.coordinates[0] <= leaf.coordinates[0]:
                # assert nearest_left.coordinates[0] != leaf.coordinates[0], "Leaves cannot have the same x-coordinate"
                nearest_left = leaf

        if leaf.coordinates[0] > x:
            if nearest_right is None:
                nearest_right = leaf
            elif nearest_right.coordinates[0] >= leaf.coordinates[0]:
                # assert nearest_right.coordinates[0] != leaf.coordinates[0], "Leaves cannot have the same x-coordinate"
                nearest_right = leaf

    return nearest_left, nearest_right


def rotate_coordinates(coordinates: list) -> list:
    """
    Rotates coordinates.
    Args:
        coordinates (list): list of coordinates
    Returns:
        list: rotated coordinates
    """
    return [(y, x) for x, y in coordinates]


def create_tree(leaves: list, nodes: list, orientation: str, labels: list = None) -> BinTree:
    """
    Creates a binary tree from a lists of leaves and nodes.
    Args:
        leaves (list): list of leaves coordinates
        nodes (list): list of nodes coordinates
        orientation (str): orientation of the tree
        labels (list): list of labels for leaves
    Returns:
        BinTree: binary tree
    """
    # Rotate coordinates if needed
    if orientation == 'horizontal':
        leaves = rotate_coordinates(leaves)
        nodes = rotate_coordinates(nodes)
        # Replace spaces in labels with underscores
    if labels:
        labels = [label.replace(' ', '_') for label in labels]

    leaves = leaves_list2class(leaves, labels)
    nodes_y_asc = sorted(nodes, key=lambda x: x[1])
    leaves_to_search = set(leaves)
    created_nodes = []
    while len(leaves_to_search) > 2:
        node = nodes_y_asc.pop()
        nearest_left, nearest_right = find_the_nearest_leaves(leaves_to_search, node)
        new_node = BinNode(nearest_left, nearest_right, node,
                           name=f'node_{str(node[0]).split(".")[0]}_{str(node[1]).split(".")[0]}')
        created_nodes.append(new_node)
        # Remove used leaves and add newly created node as artificial leaf
        leaves_to_search.remove(nearest_left)
        leaves_to_search.remove(nearest_right)
        leaves_to_search.add(new_node)
    # Create artificial root node
    root_x = np.mean([leaf.coordinates[0] for leaf in leaves_to_search])
    root_y = np.mean([leaf.coordinates[1] for leaf in leaves_to_search]) + 10
    leaves_to_search = sorted(list(leaves_to_search), key=lambda x: x.coordinates[0])
    root_node = BinNode(leaves_to_search[0], leaves_to_search[1], (root_x, root_y), name='root')

    return BinTree(root_node)


def generate_newick_str(leaves: list, nodes: list, orientation: str = 'horizontal', labels: list = None) -> str:
    """
    Generates newick string from a lists of leaves and nodes.
    Args:
        leaves (list): list of leaves coordinates
        nodes (list): list of nodes coordinates
        orientation (str): orientation of the tree
        labels (list): list of labels for leaves
    Returns:
        str: newick string
    """
    tree = create_tree(leaves, nodes, orientation, labels)
    return tree.generate_newick() + ";"


def save_to_newick(newick: str, path: str) -> None:
    """
    Saves newick string to a file.
    Args:
        newick (str): newick string
        path (str): path to the file
    """
    with open(path, 'w') as f:
        f.write(newick)


def draw_newick(newick: str, path=None) -> None:
    """
    Draws a tree from newick string.
    Args:
        newick (str): newick string
        path (str): path to the file
    """
    newick = StringIO(newick)
    tree_newick = Phylo.read(newick, 'newick')
    fig = Phylo.draw(tree_newick, do_show=False)
    if path is not None:
        plt.savefig(path)
    return fig
