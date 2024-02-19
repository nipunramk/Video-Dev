from manim import (
    ImageMobject,
    LaggedStart,
    Scene,
    Text,
    VGroup,
    config,
    LabeledDot,
    Dot,
    Line,
    TipableVMobject,
    Graph,
    WHITE,
    BLACK,
    FadeIn,
    FadeOut,
    LEFT,
    RIGHT,
    UP,
    DOWN,
)
from astar_utils import get_random_layout, solve_astar, euclidean_distance
from heapq import heappush, heappop
import sys

### THIS IS A WORKAROUND FOR NOW
### IT REQUIRES RUNNING MANIM FROM INSIDE DIRECTORY VIDEO-DEV
sys.path.insert(1, "common/")
from reducible_colors import *
from classes import CustomLabel
from functions import get_glowing_surround_rect, get_glowing_surround_circle
import numpy as np

np.random.seed(23)

# assets/bg-video.png is background image for general scenes
# assets/transition-bg.png is background image for transition scenes


class AGraph(Graph):
    def __init__(
        self,
        vertices,
        edges,
        dist_matrix=None,
        vertex_config={
            "stroke_color": REDUCIBLE_PURPLE,
            "stroke_width": 3,
            "fill_color": REDUCIBLE_PURPLE_DARK_FILL,
            "fill_opacity": 1,
        },
        edge_config={
            "color": REDUCIBLE_VIOLET,
            "stroke_width": 3,
        },
        labels=True,
        label_scale=0.6,
        label_color=WHITE,
        **kwargs,
    ):
        self.edges = edges
        if labels:
            labels = {
                k: CustomLabel(str(k), scale=label_scale).set_color(label_color)
                for k in vertices
            }
            edge_config["buff"] = LabeledDot(list(labels.values())[0]).radius
            self.labels = labels
        else:
            edge_config["buff"] = Dot().radius
            self.labels = None
        ### Manim bug where buff has no effect for some reason on standard lines
        super().__init__(
            vertices,
            edges,
            vertex_config=vertex_config,
            edge_config=edge_config,
            labels=labels,
            **kwargs,
        )
        ### RED-103: Vertices should be the same size (even for larger labels)
        for v in self.vertices:
            if v >= 10:
                self.vertices[v].scale_to_fit_height(self.vertices[0].height)
        self.edge_config = edge_config
        if dist_matrix is None:
            self.dist_matrix = np.zeros((len(vertices), len(vertices)))
            for u, v in self.edges:
                distance = np.linalg.norm(
                    self.vertices[u].get_center() - self.vertices[v].get_center()
                )
                self.dist_matrix[u][v] = distance
                self.dist_matrix[v][u] = distance
        else:
            self.dist_matrix = dist_matrix

    def get_all_edges(self, edge_type: TipableVMobject = Line, buff=None):
        edge_dict = {}
        for edge in self.edges:
            u, v = edge
            edge_dict[edge] = self.create_edge(u, v, edge_type=edge_type, buff=buff)
            edge_dict[(v, u)] = edge_dict[edge]
        return edge_dict

    def get_neighbors(self, vertex):
        return [
            v
            for v in self.vertices
            if (vertex, v) in self.edges or (v, vertex) in self.edges
        ]

    def get_adjacency_list(self):
        adjacency_list = {}
        for u in self.vertices:
            adjacency_list[u] = self.get_neighbors(u)
        return adjacency_list

    def create_edge(self, u, v, edge_type: TipableVMobject = Line, buff=None):
        return edge_type(
            self.vertices[u].get_center(),
            self.vertices[v].get_center(),
            color=self.edge_config["color"],
            stroke_width=self.edge_config["stroke_width"],
            buff=self.edge_config["buff"] if buff is None else buff,
        )

    def get_edge_weight_labels(self, scale=0.3, num_decimal_places=1):
        dist_label_dict = {}
        for edge, edge_mob in self.get_all_edges().items():
            u, v = edge
            dist_label = self.get_dist_label(
                edge_mob,
                self.dist_matrix[u][v],
                scale=scale,
                num_decimal_places=num_decimal_places,
            )
            dist_label_dict[edge] = dist_label
        return dist_label_dict

    def get_dist_label(self, edge_mob, distance, scale=0.3, num_decimal_places=1):
        return (
            Text(str(np.round(distance, num_decimal_places)), font=REDUCIBLE_MONO)
            .set_stroke(BLACK, width=8, background=True, opacity=0.8)
            .scale(scale)
            .move_to(edge_mob.point_from_proportion(0.5))
        )

    def get_dist_matrix(self):
        return self.dist_matrix

    def get_neighboring_edges(self, vertex, buff=None):
        edges = [(vertex, other) for other in get_neighbors(vertex, len(self.vertices))]
        return {edge: self.create_edge(edge[0], edge[1], buff=buff) for edge in edges}

    def get_edges_from_list(self, edges):
        edge_dict = {}
        for edge in edges:
            u, v = edge
            edge_mob = self.create_edge(u, v)
            edge_dict[edge] = edge_mob
        return edge_dict


class AStarNode:
    def __init__(self, vertex, prev_node, g_score, h_score):
        self.vertex = vertex
        self.prev_node = prev_node
        self.g_score = g_score
        self.h_score = h_score
        self.f_score = g_score + h_score

    def __str__(self):
        return f"Vertex: {self.vertex}, prev_node: {self.prev_node} g_score: {self.g_score}, h_score: {self.h_score}, f_score: {self.f_score}"


class AstarAnimationTools(Scene):
    """
    This is a scene that provides tools for animating A* search

    Every method in this class returns a sequence of animations that can be used to animate A* search

    The methods in this class are:

    show_neighbors: This method takes a vertex and a graph and returns a sequence of mobjects that scenes can use to show the neighbors of the vertex
    show_start: This method takes a start vertex and a graph and returns a sequence of mobobjects that scenes can use to show the start vertex
    show_goal: This method takes a goal vertex and a graph and returns a sequence of mobjects that scenes can use to show the goal vertex
    show_path: This method takes a path and a graph and returns a sequence of mobjects that scenes can use to show the path
    show_nodes_expanded: This method takes a sequence of nodes expanded and a graph and returns a sequence of mobjects that scenes can use to show the nodes expanded
    """

    def show_start(self, start, graph, color=REDUCIBLE_YELLOW):
        """
        This method takes a start vertex and a graph and returns a sequence of animations that shows the start vertex
        """
        return [get_glowing_surround_circle(graph.vertices[start], color=color)]

    def show_goal(self, goal, graph, color=REDUCIBLE_YELLOW):
        """
        This method takes a goal vertex and a graph and returns a sequence of animations that shows the goal vertex
        """
        return [get_glowing_surround_circle(graph.vertices[goal], color=color)]

    def show_neighbors(self, vertex, graph, color=REDUCIBLE_YELLOW):
        """
        This method takes a vertex and a graph and returns a sequence of animations that shows the neighbors of the vertex
        """
        neighbors = graph.get_neighbors(vertex)
        return [
            get_glowing_surround_circle(graph.vertices[v], color=color)
            for v in neighbors
        ]

    def show_path(self, path, graph, color=REDUCIBLE_YELLOW):
        """
        This method takes a path and a graph and returns a sequence of mobjects that shows the path
        """
        # show vertices and edges of the path
        path_mobjects = []
        all_edges = graph.get_all_edges()
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            path_mobjects.append(
                get_glowing_surround_circle(graph.vertices[u], color=color)
            )
            path_mobjects.append(all_edges[(u, v)])

        path_mobjects.append(get_glowing_surround_circle(graph.vertices[path[-1]]))
        return path_mobjects

    def show_nodes_expanded(
        self, graph, start, goal, g_func=None, h_func=None, color=REDUCIBLE_YELLOW
    ):
        """
        Runs astar algorithm and incrementally adds mobjects of vertices and edges that are expanded that later scenes can render
        """
        # A* algorithm
        # https://en.wikipedia.org/wiki/A*_search_algorithm

        # The set of nodes already evaluated
        closed_set = set()

        # The set of currently discovered nodes that are not evaluated yet.
        # Initially, only the start node is known.
        heap = []
        start_node = AStarNode(
            start, None, 0, h_func(graph.vertices[start], graph.vertices[goal])
        )
        heappush(
            heap, (h_func(graph.vertices[start], graph.vertices[goal]), start_node)
        )

        # For each node, which node it can most efficiently be reached from.
        # If a node can be reached from many nodes, came_from will eventually contain the
        # most efficient previous step.
        came_from = {}
        g_score = {v: float("inf") for v in graph.vertices}
        g_score[start] = 0
        f_score = {v: float("inf") for v in graph.vertices}
        f_score[start] = h_func(graph.vertices[start], graph.vertices[goal])

        mobjects = []
        all_edges = graph.get_all_edges()
        while len(heap) > 0:
            current = heappop(heap)[1]
            if current.prev_node is None:
                mobjects.append(
                    get_glowing_surround_circle(graph.vertices[current.vertex])
                )
            else:
                incoming_edge = all_edges[(current.prev_node, current.vertex)]
                mobjects.append(incoming_edge)
                mobjects.append(
                    get_glowing_surround_circle(graph.vertices[current.vertex])
                )

            if current.vertex == goal:
                return mobjects

            closed_set.add(current.vertex)
            for neighbor in graph.get_neighbors(current.vertex):
                if neighbor in closed_set:
                    continue

                tentative_g_score = (
                    g_score[current.vertex]
                    + graph.dist_matrix[current.vertex][neighbor]
                    if g_func is None
                    else g_func(current.vertex, neighbor)
                )
                if tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current.vertex
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + h_func(
                        graph.vertices[neighbor], graph.vertices[goal]
                    )
                    neighbor_node = AStarNode(
                        neighbor,
                        current.vertex,
                        g_score[neighbor],
                        h_func(graph.vertices[neighbor], graph.vertices[goal]),
                    )
                    heappush(heap, (f_score[neighbor], neighbor_node))

    def construct(self):
        # Test methods here on an example graph
        layout = {
            0: LEFT * 3.3 + UP * 3.2,
            1: RIGHT * 3.3 + UP * 2.7,
            2: LEFT * 2.5,
            3: RIGHT * 3.5,
            4: LEFT * 2.8 + DOWN * 3.1,
            5: RIGHT * 3.2 + DOWN * 3,
            6: DOWN * 3.3,
            7: RIGHT * 2 + DOWN * 2.4,
        }
        vertices = [0, 1, 2, 3, 4, 5, 6, 7]
        edges = [
            (0, 1),
            (0, 2),
            (1, 2),
            (1, 3),
            (2, 3),
            (2, 4),
            (3, 5),
            (4, 6),
            (5, 6),
            (2, 7),
            (3, 7),
            (4, 7),
            (6, 7),
        ]
        graph = AGraph(vertices, edges, layout=layout)
        self.add(graph)
        self.wait()

        # Test show_start
        mobjects = self.show_start(0, graph, color=REDUCIBLE_CHARM)
        self.play(FadeIn(*mobjects))
        self.wait()
        self.play(FadeOut(*mobjects))
        self.wait()

        # Test show_goal
        mobjects = self.show_goal(6, graph, color=REDUCIBLE_GREEN)
        self.play(FadeIn(*mobjects))
        self.wait()
        self.play(FadeOut(*mobjects))
        self.wait()

        # Test show_neighbors
        mobjects = self.show_neighbors(2, graph)
        self.play(FadeIn(*mobjects))
        self.wait()
        # Fade out above animations
        self.play(*[FadeOut(mobject) for mobject in mobjects])

        # Test show_path
        path = [0, 1, 3, 5, 6]
        mobjects = self.show_path(path, graph)
        for mob in mobjects:
            if isinstance(mob, TipableVMobject):
                self.play(mob.animate.set_stroke(REDUCIBLE_YELLOW, width=8))
            else:
                self.play(FadeIn(mob))
        self.wait()
        # undo previous animations
        fadeout_animatioms = []
        for mob in mobjects:
            if isinstance(mob, TipableVMobject):
                fadeout_animatioms.append(
                    mob.animate.set_stroke(
                        REDUCIBLE_VIOLET,
                        width=3,
                    )
                )
            else:
                fadeout_animatioms.append(FadeOut(mob))
        self.play(*fadeout_animatioms)
        self.wait()
        dist_label_dict = graph.get_edge_weight_labels()
        self.play(*[FadeIn(label) for label in dist_label_dict.values()])
        for label in dist_label_dict.values():
            self.add_foreground_mobject(label)
        # test show_nodes_expanded
        nodes_expanded = self.show_nodes_expanded(
            graph, 0, 6, h_func=euclidean_distance
        )
        for mob in nodes_expanded:
            if isinstance(mob, TipableVMobject):
                self.play(mob.animate.set_stroke(REDUCIBLE_YELLOW, width=8))
            else:
                self.play(FadeIn(mob))
        self.wait()

        # undo previous animations
        fadeout_animatioms = []
        for mob in nodes_expanded:
            if isinstance(mob, TipableVMobject):
                fadeout_animatioms.append(
                    mob.animate.set_stroke(
                        REDUCIBLE_VIOLET,
                        width=3,
                    )
                )
            else:
                fadeout_animatioms.append(FadeOut(mob))
        self.play(*fadeout_animatioms)
        self.wait()

        # test show_nodes_expanded with UCS
        nodes_expanded = self.show_nodes_expanded(graph, 0, 6, h_func=lambda u, v: 0)
        for mob in nodes_expanded:
            if isinstance(mob, TipableVMobject):
                self.play(mob.animate.set_stroke(REDUCIBLE_YELLOW, width=8))
            else:
                self.play(FadeIn(mob))
        self.wait()

        # undo previous animations
        fadeout_animatioms = []
        for mob in nodes_expanded:
            if isinstance(mob, TipableVMobject):
                fadeout_animatioms.append(
                    mob.animate.set_stroke(
                        REDUCIBLE_VIOLET,
                        width=3,
                    )
                )
            else:
                fadeout_animatioms.append(FadeOut(mob))
        self.play(*fadeout_animatioms)
        self.wait()


class AstarTester(Scene):
    def construct(self):
        layout = {
            0: LEFT * 3.3 + UP * 3.2,
            1: RIGHT * 3.3 + UP * 2.7,
            2: LEFT * 2.5,
            3: RIGHT * 3.5,
            4: LEFT * 2.8 + DOWN * 3.1,
            5: RIGHT * 3.2 + DOWN * 3,
            6: DOWN * 3.3,
            7: RIGHT * 2 + DOWN * 2.4,
        }
        bg = ImageMobject("assets/bg-video.png").scale_to_fit_width(config.frame_width)
        self.add(bg)
        vertices = [0, 1, 2, 3, 4, 5, 6, 7]
        edges = [
            (0, 1),
            (0, 2),
            (1, 2),
            (1, 3),
            (2, 3),
            (2, 4),
            (3, 5),
            (4, 6),
            (5, 6),
            (2, 7),
            (3, 7),
            (4, 7),
            (6, 7),
        ]
        graph = AGraph(vertices, edges, layout=layout)
        dist_label_dict = graph.get_edge_weight_labels()
        self.play(FadeIn(graph), *[FadeIn(label) for label in dist_label_dict.values()])
        self.wait()
        shortest_path, cost = solve_astar(graph, 0, 6, h_func=euclidean_distance)
        print("Shortest path and cost:", shortest_path, cost)

        ucs_shortest_path, ucs_cost = solve_astar(graph, 0, 6, h_func=lambda u, v: 0)
        print("Shortest path and cost:", ucs_shortest_path, ucs_cost)

        greedy_shortest_path, greedy_cost = solve_astar(
            graph, 0, 6, g_func=lambda u, v: 0, h_func=euclidean_distance
        )
        print("Shortest path and cost:", greedy_shortest_path, greedy_cost)

        shortest_path, cost = solve_astar(graph, 1, 6, h_func=euclidean_distance)
        print("Shortest path and cost:", shortest_path, cost)

        ucs_shortest_path, ucs_cost = solve_astar(graph, 1, 6, h_func=lambda u, v: 0)
        print("Shortest path and cost:", ucs_shortest_path, ucs_cost)

        greedy_shortest_path, greedy_cost = solve_astar(
            graph, 1, 6, g_func=lambda u, v: 0, h_func=euclidean_distance
        )
        print("Shortest path and cost:", greedy_shortest_path, greedy_cost)
