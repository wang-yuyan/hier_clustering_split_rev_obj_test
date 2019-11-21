from sklearn.cluster import KMeans
import numpy as np
import random
from scipy.cluster.hierarchy import average as scipy_avlk
from scipy.cluster.hierarchy import single as scipy_sglk
from scipy.cluster.hierarchy import to_tree
from scipy.spatial.distance import pdist
from helper_functions import Node, get_leaves, get_dist_matrix, load_data, print_tree, subsample

#get the ckmm objective function value at one split
def cohen_addad_rev(A, B):
    rev = 0.0
    n_1 = A.shape[0]
    n_2 = B.shape[0]
    for i in range(n_1):
        for j in range(n_2):
            rev += np.linalg.norm(A[i] - B[j])
    return rev * (n_1 + n_2)


#get the Hierarchical Revenue objective value
def get_split_obj(root, points):
    if root.is_leaf():
        return 0
    left_obj = get_split_obj(root.get_left(), points)
    right_obj = get_split_obj(root.get_right(), points)
    left_indices = get_leaves(root.get_left())
    right_indices = get_leaves(root.get_right())
    A = points[left_indices][:]
    B = points[right_indices][:]
    return split_revenue(A,B) + left_obj + right_obj


#get the total ckmm objectve value
def get_cohen_addad_obj(root, points):
    if root.is_leaf():
        return 0
    left_obj = get_cohen_addad_obj(root.get_left(), points)
    right_obj = get_cohen_addad_obj(root.get_right(), points)
    left_indices = get_leaves(root.get_left())
    right_indices = get_leaves(root.get_right())
    A = points[left_indices][:]
    B = points[right_indices][:]
    return cohen_addad_rev(A, B) + left_obj + right_obj


#get a random split
def split_random(root, leaves, id_count):
    n = len(leaves)
    if n <= 1:
        return None, None, None, None, 0, id_count

    points = np.array([leaf.get_rho() for leaf in leaves])
    left_indices = []
    right_indices = []
    for i in range(n):
        if random.uniform(0,1) <= 0.5:
            left_indices.append(i)
        else:
            right_indices.append(i)

    #if one of the side is empty, put the first point into the other side
    if len(left_indices) == 0:
        left_indices.append(right_indices.pop(0))
    if len(right_indices) == 0:
        right_indices.append(left_indices.pop(0))

    left_points = points[left_indices][:]
    right_points = points[right_indices][:]

    left_leaves = [leaves[x] for x in left_indices]
    right_leaves = [leaves[y] for y in right_indices]

    rho_1 = np.average(left_points, axis=0)
    rho_2 = np.average(right_points, axis=0)
    L = len(left_leaves)
    R = len(right_leaves)
    if L == 1:
        left_node = left_leaves[0]
    else:
        id_count -= 1
        left_node = Node(id=id_count, left=None, right=None, count=L, rho=rho_1)

    if R == 1:
        right_node = right_leaves[0]
    else:
        id_count -= 1
        right_node = Node(id=id_count, left=None, right=None, count=R, rho=rho_2)

    root.left = left_node
    root.right = right_node

    rev = split_revenue(left_points, right_points)
    return left_node, right_node, left_leaves, right_leaves, rev, id_count

#get the Hierarchical Revenue objective at one split
def split_revenue(A, B):
    #print(A)
    #print(B)
    n_1 = A.shape[0]
    n_2 = B.shape[0]
    rho_a = np.average(A, axis = 0)
    rho_b = np.average(B, axis = 0)
    #print(rho_a, rho_b)
    rev = 0
    for i in range(n_1):
        for j in range(n_2):
            p_1 = A[i]
            p_2 = B[j]
            d = np.linalg.norm(p_1 - p_2)
            d_1 = np.linalg.norm(p_1 - rho_a)
            d_2 = np.linalg.norm(p_2 - rho_b)
            d_max = max(d, d_1, d_2)
            if d_max == 0:
                rev += 1
                #print(p_1, p_2, 1)
            else:
                rev += d / d_max
                #print(p_1, p_2, d / d_max)
    return rev


#use scipy 2means to get a local split
def split_kmeans(root, leaves, id_count):
    n = len(leaves)
    if n <= 1:
        return None, None, None, None, 0, id_count
    points = np.array([leaf.get_rho() for leaf in leaves])
    kmeans = KMeans(n_clusters=2).fit(points)
    y = kmeans.labels_

    if np.sum(y) == 0:
        y[0] = 1
    if np.sum(y) == n:
        y[0] = 0

    left_leaves = []
    right_leaves = []
    for i in range(n):
        if y[i] < 0.5:
            left_leaves.append(leaves[i])
        else:
            right_leaves.append(leaves[i])
    left_points = points[y<0.5][:]
    right_points = points[y>0.5][:]
    rho_1 = np.average(left_points, axis=0)
    rho_2 = np.average(right_points, axis=0)
    L = len(left_leaves)
    R = len(right_leaves)
    if L == 1:
        left_node = left_leaves[0]
    else:
        id_count -= 1
        left_node = Node(id=id_count, left=None, right=None, count=L, rho=rho_1)

    if R == 1:
        right_node = right_leaves[0]
    else:
        id_count -= 1
        right_node = Node(id=id_count, left=None, right=None, count=R, rho=rho_2)

    root.left = left_node
    root.right = right_node

    rev = split_revenue(left_points, right_points)
    return left_node, right_node, left_leaves, right_leaves, rev, id_count

#calls scipy implementation of average linkage
def avlk(points):
    dist = get_dist_matrix(points)
    Z = pdist(dist)
    cluster_matrix = scipy_avlk(Z)
    scipy_root = to_tree(cluster_matrix)
    return scipy_root

#calls scipy implementation of single linkage
def sglk(points):
    dist = get_dist_matrix(points)
    Z = pdist(dist)
    cluster_matrix = scipy_sglk(Z)
    scipy_root = to_tree(cluster_matrix)
    return scipy_root

#use bisecting kmeans to contruct a tree
def bisect_kmeans(points):
    n = points.shape[0]
    leaves = [Node(id=i, left=None, right=None, count=1, rho=points[i][:]) for i in range(n)]
    id_count = 2 * n - 2
    root = Node(id=id_count, left=None, right=None, count=n, rho=np.average(points, axis=0))
    node_list = [(root, leaves)]
    total_rev = 0

    while len(node_list) > 0:
        this_root = node_list[0][0]
        this_leaves = node_list[0][1]
        del node_list[0]
        left_node, right_node, left_leaves, right_leaves, rev, id_count = split_kmeans(this_root, this_leaves, id_count)
        total_rev += rev
        if left_node is not None:
            node_list.append((left_node, left_leaves))
        if right_node is not None:
            node_list.append((right_node, right_leaves))

    return root, total_rev

#use Random algorithm to construct a tree
def bisect_random(points):
    #produce a tree using Random algorithm
    n = points.shape[0]
    leaves = [Node(id=i, left=None, right=None, count=1, rho=points[i][:]) for i in range(n)]
    id_count = 2 * n - 2
    root = Node(id=id_count, left=None, right=None, count=n, rho=np.average(points, axis=0))
    node_list = [(root, leaves)]
    total_rev = 0

    while len(node_list) > 0:
        this_root = node_list[0][0]
        this_leaves = node_list[0][1]
        del node_list[0]

        left_node, right_node, left_leaves, right_leaves, rev, id_count = split_random(this_root, this_leaves, id_count)

        total_rev += rev
        if left_node is not None:
            node_list.append((left_node, left_leaves))
        if right_node is not None:
            node_list.append((right_node, right_leaves))

    return root, total_rev

if __name__ == "__main__":
    filename = "./bank.csv"
    data = load_data(filename)
    num = 1000
    num_instances = 5

    #filenames for saving objective function values
    split_file = "./split_objs_1000.txt"
    ckmm_file = "./ckmm_objs_1000.txt"

    split_f = open(split_file, "w")
    ckmm_f = open(ckmm_file, "w")

    #write the names of the four algorithms
    split_f.write("bisecting k-means  average linkage  single linkage  random  bound\n")
    ckmm_f.write("bisecting k-means  average linkage  single linkage  random  bound\n")

    for i in range(num_instances):

        print("This is the instance %d" %i)

        points = subsample(data, num)
        #this is for scaling adult data, some dimension has a much larger scale than others, which might reduce the data to almost one dimension
        #for i in range(num):
            #points[i][1] = points[i][1] / 10000

        #the upper bound for hierarchical revenue objective is always the number of pairs
        print("the upper bound for revenue:")
        print(num * (num - 1) / 2)

        # test four objective functions
        random_root, random_split_obj = bisect_random(points)
        kmeans_root, kmeans_split_obj = bisect_kmeans(points)
        avlk_root = avlk(points)
        avlk_split_obj = get_split_obj(avlk_root, points)
        sglk_root = sglk(points)
        sglk_split_obj = get_split_obj(sglk_root, points)

        random_ckmm_obj = get_cohen_addad_obj(random_root, points)
        kmeans_ckmm_obj = get_cohen_addad_obj(kmeans_root, points)
        avlk_ckmm_obj = get_cohen_addad_obj(avlk_root, points)
        sglk_ckmm_obj = get_cohen_addad_obj(sglk_root, points)

        print("the split obj for bisecting k-means, average linkage, single linkage and random:")
        print(kmeans_split_obj, avlk_split_obj, sglk_split_obj, random_split_obj)
        split_f.write("{} {} {} {} {}\n".format(kmeans_split_obj, avlk_split_obj, sglk_split_obj, random_split_obj, num * (num - 1) / 2))

        dist = get_dist_matrix(points)
        ckmm_bound = np.sum(dist) / 2 * num

        print("the upper bound for ckmm objective is:")
        print(ckmm_bound)

        print("the CKMM obj for bisecting k-means, average linkage, single linkage and random:")
        print(kmeans_ckmm_obj, avlk_ckmm_obj, sglk_ckmm_obj, random_ckmm_obj)
        ckmm_f.write("{} {} {} {} {}\n".format(kmeans_ckmm_obj, avlk_ckmm_obj, sglk_ckmm_obj, random_ckmm_obj, ckmm_bound))

    split_f.close()
    ckmm_f.close()
