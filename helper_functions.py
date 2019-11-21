import numpy as np
import random
#construct a class of nodes that resembles that in the scipy implementation
class Node:
    def __init__(self, id = None, left = None, right = None, count = 0, rho = None):
        self.id = id
        self.left = left
        self.right = right
        self.count = count
        self.rho = rho

    def get_count(self):
        return self.count

    def get_id(self):
        return self.id

    def get_left(self):
        return self.left

    def get_right(self):
        return self.right

    def get_rho(self):
        return self.rho

    def is_leaf(self):
        if self.left == None and self.right == None:
            return True
        return False

def get_leaves(root):
    if root.is_leaf():
        return [root.get_id()]
    else:
        return get_leaves(root.get_left()) + get_leaves(root.get_right())

def get_dist_matrix(points):
    n = points.shape[0]
    dist = np.zeros((n,n))
    for i in range(n):
        for j in range(i):
            dif = np.array(points[i]) - np.array(points[j])
            d = np.linalg.norm(dif)
            dist[i][j] = d
            dist[j][i] = d
    return dist

def load_data(filename):
    delim = ','
    points = []
    for line in open(filename):
        y = line.strip().split(delim)
        n = len(y)
        for i in range(n):
            y[i] = float(y[i])
        if n >= 1:
            points.append(y[1:])
    return np.array(points)

def print_tree(node, s=""):
    print(s, node.get_id(), node.get_count(), node.get_rho()[:3])
    if not node.is_leaf():
        print_tree(node.get_left(), "\t" + s[:-3] + "|--")
        print_tree(node.get_right(), "\t" + s[:-3] + "\\--")

def subsample(points, num):
    n = points.shape[0]
    sample_index = random.sample(range(n), num)
    return points[sample_index][:]