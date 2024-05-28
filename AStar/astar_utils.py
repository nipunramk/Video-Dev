import numpy as np
from manim import config
from heapq import heappush, heappop


def solve_astar(graph, start, goal, g_func=None, h_func=None):
    # A* algorithm
    # https://en.wikipedia.org/wiki/A*_search_algorithm
    # https://www.redblobgames.com/pathfinding/a-star/introduction.html

    # The set of nodes already evaluated
    closed_set = set()

    # The set of currently discovered nodes that are not evaluated yet.
    # Initially, only the start node is known.
    heap = []
    heappush(heap, (h_func(graph.vertices[start], graph.vertices[goal]), start))

    # For each node, which node it can most efficiently be reached from.
    # If a node can be reached from many nodes, came_from will eventually contain the
    # most efficient previous step.
    came_from = {}
    g_score = {v: float("inf") for v in graph.vertices}
    g_score[start] = 0
    f_score = {v: float("inf") for v in graph.vertices}
    f_score[start] = h_func(graph.vertices[start], graph.vertices[goal])

    while len(heap) > 0:
        current = heappop(heap)[1]

        if current == goal:
            return reconstruct_path(graph, came_from, current)

        closed_set.add(current)
        for neighbor in graph.get_neighbors(current):
            if neighbor in closed_set:
                continue

            tentative_g_score = (
                g_score[current] + graph.dist_matrix[current][neighbor]
                if g_func is None
                else g_func(current, neighbor)
            )
            if tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + h_func(
                    graph.vertices[neighbor], graph.vertices[goal]
                )
                heappush(heap, (f_score[neighbor], neighbor))


def solve_uniform_cost_search(graph, start, goal):
    # TODO TEST
    return solve_astar(graph, start, goal, g_func=None, h_func=lambda u, v: 0)


def solve_greedy_best_first_search(graph, start, goal):
    # TODO TEST
    return solve_astar(
        graph, start, goal, g_func=lambda u, v: 0, h_func=euclidean_distance
    )


def reconstruct_path(graph, came_from, current):
    cost = 0
    total_path = [current]
    while current in came_from:
        u, v = current, came_from[current]
        distance = euclidean_distance(graph.vertices[u], graph.vertices[v])
        current = came_from[current]
        total_path.insert(0, current)
        cost += distance
    return total_path, cost


def euclidean_distance(v1, v2):
    return np.linalg.norm(v1.get_center() - v2.get_center())


def manhattan_distance(v1, v2):
    return np.sum(np.abs(v1.get_center() - v2.get_center()))


def get_random_layout(N):
    random_points_in_frame = get_random_points_in_frame(N)
    return {v: point for v, point in zip(range(N), random_points_in_frame)}


def get_random_points_in_frame(N):
    return [get_random_point_in_frame() for _ in range(N)]


def get_random_point_in_frame():
    x = np.random.uniform(-config.frame_x_radius + 2, config.frame_x_radius - 2)
    y = np.random.uniform(-config.frame_y_radius + 0.5, config.frame_y_radius - 0.5)
    return np.array([x, y, 0])
