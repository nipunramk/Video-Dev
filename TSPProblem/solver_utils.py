import numpy as np
from manim import *


def get_all_tour_permutations(
    N: int, start: int, max_cap: int = 1000, return_duplicates=False
):
    """
    @param: N, number of cities
    @param: start, starting city
    @param: max_cap, maximum number of tours to return, defaults to 1000.
    @return: list of all possible unique tours from start to end
    """
    tours = []
    seen_vertices = set()

    def generate_permutations(current, current_tour):
        if len(current_tour) == N or len(tours) >= max_cap:
            tours.append(current_tour.copy())
            return

        seen_vertices.add(current)
        for neighbor in get_neighbors(current, N):
            if neighbor not in seen_vertices:
                current_tour.append(neighbor)
                generate_permutations(neighbor, current_tour)
                last_vertex = current_tour.pop()
                if last_vertex in seen_vertices:
                    seen_vertices.remove(last_vertex)

    generate_permutations(start, [start])

    # using a set significantly speeds up this section
    set_non_duplicate_tours = set()
    non_duplicate_tours = []
    for tour in tours:
        # e.g [0, 2, 3, 4] and [0, 4, 3, 2] are the same tour
        if tuple([tour[0]] + tour[1:][::-1]) in set_non_duplicate_tours:
            continue
        else:
            set_non_duplicate_tours.add(tuple(tour))
            non_duplicate_tours.append(tour)

    if return_duplicates:
        # return duplicates but grouped by symmetry
        duplicate_tours = []
        for tour in non_duplicate_tours:
            symm_group = [tour, [tour[0], *tour[1:][::-1]]]
            duplicate_tours.append(symm_group)

        return duplicate_tours
    else:
        return non_duplicate_tours


def get_neighbors(vertex, N):
    return list(range(0, vertex)) + list(range(vertex + 1, N))


# def get_cost_from_permutation(dist_matrix, permutation):
#     cost = 0
#     for i in range(len(permutation)):
#         u, v = i, (i + 1) % len(permutation)
#         cost += dist_matrix[u][v]
#     return cost

# i don't know exactly what your function does, but
# it definitely does not return the permutation's cost.
# i'll leave it commented, but this one is working properly now
def get_cost_from_permutation(dist_matrix, permutation):
    cost = 0
    for t in permutation:
        cost += dist_matrix[t]

    return cost


def get_exact_tsp_solution(dist_matrix):
    from python_tsp.exact import solve_tsp_dynamic_programming

    permutation, distance = solve_tsp_dynamic_programming(dist_matrix)
    return permutation, distance


def get_edges_from_tour(tour):
    """
    @param: tour -- list of vertices that are part of the tour
    @return: list of edges
    """
    edges = []
    for i in range(len(tour)):
        edges.append((tour[i], tour[(i + 1) % len(tour)]))
    return edges


def get_cost_from_edges(edges, dist_matrix):
    return sum([dist_matrix[u][v] for u, v in edges])

def get_random_points_in_frame(N):
    return [get_random_point_in_frame() for _ in range(N)]

def get_random_point_in_frame():
    x = np.random.uniform(-config.frame_x_radius + 2, config.frame_x_radius - 2)
    y = np.random.uniform(-config.frame_y_radius + 0.5, config.frame_y_radius - 0.5)
    return np.array([x, y, 0])

def get_nearest_neighbor_solution(dist_matrix, start=0):
    current = start
    seen = set([current])
    tour = [current]
    unseen_nodes = set(list(range(dist_matrix.shape[0]))).difference(seen)
    total_cost = 0
    while len(unseen_nodes) > 0:
        min_dist_vertex = min(unseen_nodes, key=lambda x: dist_matrix[current][x])
        total_cost += dist_matrix[current][min_dist_vertex]
        tour.append(min_dist_vertex)
        current = min_dist_vertex
        seen.add(current)
        unseen_nodes = set(list(range(dist_matrix.shape[0]))).difference(seen)
    # cost to go back to start
    total_cost += dist_matrix[tour[-1]][tour[0]]
    return tour, total_cost

def get_mst(dist_matrix, v_to_ignore=None):
    # A utility function to find the vertex with
    # minimum distance value, from the set of vertices
    # not yet included in shortest path tree
    vertices_to_consider = list(range(dist_matrix.shape[0]))
    if v_to_ignore is not None:
        vertices_to_consider.remove(v_to_ignore)

    print('Vertices to consider', vertices_to_consider)

    def min_key(key, mst_set):
        # Initialize minim value
        minim = float('inf')
        for v in vertices_to_consider:
            if key[v] < minim and mst_set[v] == False:
                minim = key[v]
                min_index = v
 
        return min_index
 
    # Key values used to pick minimum weight edge in cut
    key = [float('inf')] * dist_matrix.shape[0]
    parent = [None] * dist_matrix.shape[0] # Array to store constructed MST
    # Make key 0 so that this vertex is picked as first vertex
    key[vertices_to_consider[0]] = 0
    mst_set = [False] * dist_matrix.shape[0]
    parent[vertices_to_consider[0]] = -1 # First node is always the root of

    for _ in range(len(vertices_to_consider)):

        # Pick the minimum distance vertex from
        # the set of vertices not yet processed.
        # u is always equal to src in first iteration
        u = min_key(key, mst_set)

        # Put the minimum distance vertex in
        # the shortest path tree
        mst_set[u] = True

        # Update dist value of the adjacent vertices
        # of the picked vertex only if the current
        # distance is greater than new distance and
        # the vertex in not in the shortest path tree
        for v in vertices_to_consider:
            # dist_matrix[u][v] is non zero only for adjacent vertices of m
            # mst_set[v] is false for vertices not yet included in MST
            # Update the key only if dist_matrix[u][v] is smaller than key[v]
            if dist_matrix[u][v] > 0 and mst_set[v] == False and key[v] > dist_matrix[u][v]:
                key[v] = dist_matrix[u][v]
                parent[v] = u
    cost = 0
    mst_edges = []
    for i in range(1, len(vertices_to_consider)):
        mst_edges.append((parent[vertices_to_consider[i]], vertices_to_consider[i]))
        cost += dist_matrix[parent[vertices_to_consider[i]]][i]
    return mst_edges, cost

def get_1_tree(dist_matrix, v_to_ignore):
    mst_edges, cost = get_mst(dist_matrix, v_to_ignore=v_to_ignore)
    closest_vertices = sorted([v for v in range(dist_matrix.shape[0]) if v != v_to_ignore], key=lambda x: dist_matrix[x][v_to_ignore])
    additional_edges = [(v_to_ignore, closest_vertices[0]), (v_to_ignore, closest_vertices[1])]
    one_tree_edges = mst_edges + additional_edges
    one_tree_cost = sum([dist_matrix[u][v] for u, v in one_tree_edges])
    return mst_edges, cost, one_tree_edges, one_tree_cost
