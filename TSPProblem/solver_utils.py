import numpy as np

def get_all_tour_permutations(N: int, start: int):
    """
    @param: N, number of cities
    @param: start, starting city
    @return: list of all possible unique tours from start to end
    """
    tours = []
    seen_vertices = set()
    def generate_permutations(current, current_tour):
        if len(current_tour) == N:
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

    set_all_tours = set([tuple(tour) for tour in tours])
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

    return non_duplicate_tours

def get_neighbors(vertex, N):
    return list(range(0, vertex)) + list(range(vertex + 1, N))

def get_cost_from_permutation(dist_matrix, permutation):
    cost = 0
    for i in range(len(permutation)):
        u, v = i, (i + 1) % len(permutation)
        cost += dist_matrix[u][v]
    return cost

def get_exact_tsp_solution(dist_matrix):
    from python_tsp.exact import solve_tsp_dynamic_programming
    permutation, distance = solve_tsp_dynamic_programming(distance_matrix)
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