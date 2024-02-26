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
    Arrow,
    ORIGIN,
    Write,
    DashedLine,
    SurroundingRectangle,
    MEDIUM,
    DL,
    UL,
    UR,
    DR,
    SMALL_BUFF,
    MED_SMALL_BUFF,
    MED_LARGE_BUFF,
    DEFAULT_MOBJECT_TO_MOBJECT_BUFFER,
    ArcBetweenPoints,
    PI,
    DashedVMobject,
    Indicate,
    Transform, TransformFromCopy, Tex, Brace, GrowFromCenter,
    Square, Rectangle, ReplacementTransform
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
        edge_type=Line,
        edge_buff=None,
        labels=True,
        label_scale=0.6,
        label_color=WHITE,
        edge_to_prop={},
        **kwargs,
    ):
        self.edges = edges
        self.edge_to_prop = edge_to_prop
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

        self.edge_dict = {}
        for edge in self.edges:
            u, v = edge
            self.edge_dict[edge] = self.create_edge(
                u, v, edge_type=edge_type, buff=edge_buff
            )
            self.edge_dict[(v, u)] = self.edge_dict[edge]

    def get_all_edges(self):
        return self.edge_dict

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
            if (v, u) in dist_label_dict:
                continue
            dist_label = self.get_dist_label(
                edge_mob,
                self.dist_matrix[u][v],
                scale=scale,
                num_decimal_places=num_decimal_places,
                proportion=(
                    self.edge_to_prop[edge] if edge in self.edge_to_prop else 0.5
                ),
            )
            dist_label_dict[edge] = dist_label
        return dist_label_dict

    def get_dist_label(
        self, edge_mob, distance, scale=0.3, num_decimal_places=1, proportion=0.5
    ):
        return (
            Text(str(np.round(distance, num_decimal_places)), font=REDUCIBLE_MONO)
            .set_stroke(BLACK, width=8, background=True, opacity=0.8)
            .scale(scale)
            .move_to(edge_mob.point_from_proportion(proportion))
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

    def __eq__(self, other):
        return self.vertex == other.vertex and self.f_score == other.f_score

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

        path_mobjects.append(
            get_glowing_surround_circle(graph.vertices[path[-1]], color=color)
        )
        return path_mobjects

    def show_nodes_expanded(
        self, graph, start, goal, g_func=None, h_func=None, color=REDUCIBLE_YELLOW, include_heap_mobs=False,
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
        heap_mob = self.make_heap(heap.copy())
        heap_state = (heap, heap_mob)

        # For each node, which node it can most efficiently be reached from.
        # If a node can be reached from many nodes, came_from will eventually contain the
        # most efficient previous step.
        came_from = {}
        g_score = {v: float("inf") for v in graph.vertices}
        g_score[start] = 0
        f_score = {v: float("inf") for v in graph.vertices}
        f_score[start] = h_func(graph.vertices[start], graph.vertices[goal])

        mobjects = []
        if include_heap_mobs:
            mobjects.append(heap_state)
        all_edges = graph.get_all_edges()
        while len(heap) > 0:
            current = heappop(heap)[1]
            if include_heap_mobs:
                mobjects.append((sorted(heap, key=lambda x: x[0]), self.make_heap(heap.copy())))
            if current.prev_node is None:
                mobjects.append(
                    get_glowing_surround_circle(graph.vertices[current.vertex])
                )
            else:
                incoming_edge = all_edges[(current.prev_node.vertex, current.vertex)]
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
                        current,
                        g_score[neighbor],
                        h_func(graph.vertices[neighbor], graph.vertices[goal]),
                    )
                    heappush(heap, (f_score[neighbor], neighbor_node))
            
            if include_heap_mobs:
                # DEBUG HERE
                mobjects.append((sorted(heap, key=lambda x: x[0]), self.make_heap(heap.copy())))

    def filter_layout(self, layout, min_distance_between_points=0.2):
        filtered_layout = {}
        new_index = 0
        for v in layout:
            if all(
                np.linalg.norm(layout[v] - layout[u]) > min_distance_between_points
                for u in filtered_layout
            ):
                filtered_layout[new_index] = layout[v]
                new_index += 1
        return filtered_layout

    def get_large_random_graph(
        self,
        N,
        dist_threshold=3,
        p=0.3,
        min_distance_between_points=0.2,
        vertex_config={
            "stroke_color": REDUCIBLE_PURPLE,
            "stroke_width": 3,
            "fill_color": REDUCIBLE_PURPLE_DARK_FILL,
            "fill_opacity": 1,
        },
        labels=False,
    ):
        layout = get_random_layout(N)
        layout = self.filter_layout(
            layout, min_distance_between_points=min_distance_between_points
        )
        vertices = list(layout.keys())
        edges = []
        for u in vertices:
            for v in vertices:
                if (
                    u != v
                    and np.linalg.norm(layout[u] - layout[v]) < dist_threshold
                    and np.random.rand() < p
                ):
                    edges.append((u, v))
        return AGraph(
            vertices, edges, layout=layout, labels=labels, vertex_config=vertex_config
        )
    
    def get_current_path(self, node):
        path = [node.vertex]
        while node.prev_node is not None:
            path.append(node.prev_node.vertex)
            node = node.prev_node
        return path[::-1]

    def make_heap_node(self, node, weight, color=WHITE):
        current_path = self.get_current_path(node)
        node = node.vertex
        node_rect = Rectangle(height=0.7, width=1.1, color=color)
        node = Text(str(node), font=REDUCIBLE_MONO, color=color).scale(0.5).move_to(node_rect)
        weight_rectange = Rectangle(width=node_rect.width, height=0.3, color=color)
        weight = round(weight, 1)
        weight = Text(str(weight), font=REDUCIBLE_MONO, color=color).scale(0.3).move_to(weight_rectange)
        node_mob = VGroup(node, node_rect)
        weight_mob = VGroup(weight, weight_rectange)
        path_rect = Rectangle(width=node_rect.width, height=0.3, color=color)
        path_text = Text("->".join(map(str, current_path)), font=REDUCIBLE_MONO, color=color).scale(0.3).move_to(path_rect)
        if path_text.width > path_rect.width:
            path_text.scale_to_fit_width(path_rect.width - SMALL_BUFF * 2)
        path_mob = VGroup(path_text, path_rect)
        if len(current_path) > 1:
            return VGroup(path_mob, weight_mob, node_mob).arrange(DOWN, buff=0)
        return VGroup(weight_mob, node_mob).arrange(DOWN, buff=0)
    
    def make_heap(self, heap, color=WHITE, buff_between_items=MED_SMALL_BUFF):
        heap = sorted(heap, key=lambda x: x[0])
        heap_mobs = []
        for weight, node in heap:
            heap_mobs.append(self.make_heap_node(node, weight, color=color))
        if len(heap_mobs) > 1:
            return VGroup(*heap_mobs).arrange(DOWN, buff=buff_between_items)
        return VGroup(*heap_mobs)


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


class UCSLargeGraph(AstarAnimationTools):

    def construct(self):
        # 23 -> 45
        large_graph = self.get_large_random_graph(
            60, dist_threshold=3, p=0.5, min_distance_between_points=0.4
        )
        self.play(FadeIn(large_graph))
        self.wait()
        for mob in large_graph.vertices.values():
            self.add_foreground_mobject(mob)

        start = self.show_start(23, large_graph, color=REDUCIBLE_CHARM)
        self.play(FadeIn(*start))
        self.wait()

        goal = self.show_goal(45, large_graph, color=REDUCIBLE_GREEN)
        self.play(FadeIn(*goal))
        self.wait()
        # UCS nodes exampled
        nodes_expanded = self.show_nodes_expanded(
            large_graph, 23, 45, h_func=lambda u, v: 0
        )
        animations = []
        # exclude the first animation becase we already highlighted the start node
        for mob in nodes_expanded[1:]:
            if isinstance(mob, TipableVMobject):
                animations.append(mob.animate.set_stroke(REDUCIBLE_YELLOW, width=4))
            else:
                animations.append(FadeIn(mob))

        self.play(
            LaggedStart(*animations),
            run_time=30,
        )
        self.wait()
        path, _ = solve_astar(large_graph, 23, 45, h_func=lambda u, v: 0)
        # show optimal path
        optimal_path = self.show_path(path, large_graph, color=REDUCIBLE_GREEN_LIGHTER)
        animations = []
        for mob in optimal_path:
            if isinstance(mob, TipableVMobject):
                animations.append(
                    mob.animate.set_stroke(REDUCIBLE_GREEN_LIGHTER, width=4)
                )
            else:
                animations.append(FadeIn(mob))
        self.play(LaggedStart(*animations), run_time=5)
        self.wait()

        la_node = large_graph.vertices[39]
        dallas_node = large_graph.vertices[23]
        nyc_node = large_graph.vertices[45]

        dallas_text = Text("Dallas", font=REDUCIBLE_MONO, color=WHITE).scale(0.3)
        la_text = Text("Los Angeles", font=REDUCIBLE_MONO, color=WHITE).scale(0.3)
        nyc_text = Text("New York City", font=REDUCIBLE_MONO, color=WHITE).scale(0.3)

        dallas_arrow = Arrow(UL, dallas_node.get_center()).set_color(WHITE)
        nyc_arrow = Arrow(
            nyc_node.get_center() + DR * 1, nyc_node.get_center()
        ).set_color(WHITE)
        la_arrow = Arrow(la_node.get_center() + UL * 1, la_node.get_center()).set_color(
            WHITE
        )

        dallas_text.next_to(dallas_arrow, UL)
        nyc_text.next_to(nyc_arrow, DOWN).shift(RIGHT * SMALL_BUFF * 3)
        la_text.next_to(la_arrow, UL, buff=SMALL_BUFF)

        self.play(FadeIn(dallas_arrow), FadeIn(dallas_text))
        self.wait()
        self.play(FadeIn(la_arrow), FadeIn(la_text))
        self.wait()
        self.play(FadeIn(nyc_arrow), FadeIn(nyc_text))
        self.wait()


class GreedyApproachVsUCSBroad(AstarAnimationTools):
    def construct(self):
        large_graph = self.get_large_random_graph(
            60, dist_threshold=3, p=0.5, min_distance_between_points=0.4
        )
        self.play(FadeIn(large_graph))
        self.wait()
        for mob in large_graph.vertices.values():
            self.add_foreground_mobject(mob)

        start = self.show_start(23, large_graph, color=REDUCIBLE_CHARM)
        self.play(FadeIn(*start))
        self.wait()

        goal = self.show_goal(45, large_graph, color=REDUCIBLE_GREEN)
        self.play(FadeIn(*goal))
        self.wait()
        # Greedy nodes exampled
        nodes_expanded = self.show_nodes_expanded(
            large_graph,
            23,
            45,
            g_func=lambda u, v: 0,
            h_func=euclidean_distance,
        )
        animations = []
        # exclude the first animation becase we already highlighted the start node
        for mob in nodes_expanded[1:]:
            if isinstance(mob, TipableVMobject):
                animations.append(mob.animate.set_stroke(REDUCIBLE_YELLOW, width=4))
            else:
                animations.append(FadeIn(mob))

        self.play(
            LaggedStart(*animations),
            run_time=30,
        )
        self.wait()
        path, _ = solve_astar(
            large_graph, 23, 45, g_func=lambda u, v: 0, h_func=euclidean_distance
        )
        # show greedy path
        greedy_path = self.show_path(path, large_graph, color=REDUCIBLE_ORANGE)
        animations = []
        for mob in greedy_path:
            if isinstance(mob, TipableVMobject):
                animations.append(mob.animate.set_stroke(REDUCIBLE_ORANGE, width=4))
            else:
                animations.append(FadeIn(mob))
        self.play(LaggedStart(*animations), run_time=5)
        self.wait()

        key_difference = (
            Text(
                "The greedy approach does not bother exploring westward",
                font=REDUCIBLE_FONT,
                color=WHITE,
            )
            .scale(0.5)
            .move_to(UP * 3.5)
        )

        self.play(
            FadeIn(key_difference),
            *[mob.animate.shift(DOWN * 0.2) for mob in self.mobjects],
        )
        self.wait()

        self.clear()
        self.wait()

        greedy_title = (
            Text("Greedy Approach", font=REDUCIBLE_FONT).scale(0.7).move_to(UP * 3.5)
        )
        con = (
            Text("- Not guaranteed to find the shortest path", font=REDUCIBLE_FONT)
            .scale(0.5)
            .next_to(greedy_title, DOWN)
        )
        pro = (
            Text("- Finds path quickly", font=REDUCIBLE_FONT)
            .scale(0.5)
            .next_to(con, DOWN)
        )

        details = (
            VGroup(con, pro)
            .arrange(DOWN, aligned_edge=LEFT)
            .next_to(greedy_title, DOWN)
        )
        self.play(FadeIn(greedy_title), FadeIn(details))
        self.wait()

        self.play(
            greedy_title.animate.shift(LEFT * 3.5), details.animate.shift(LEFT * 3.5)
        )
        self.wait()

        ucs_title = (
            Text("Uniform Cost Search", font=REDUCIBLE_FONT)
            .scale(0.7)
            .move_to(RIGHT * 3.5 + UP * 3.5)
        )
        pro = (
            Text("- Guaranteed to find the shortest path", font=REDUCIBLE_FONT)
            .scale(0.5)
            .next_to(ucs_title, DOWN)
        )
        con = Text("- Can be slow", font=REDUCIBLE_FONT).scale(0.5).next_to(pro, DOWN)
        ucs_details = (
            VGroup(pro, con).arrange(DOWN, aligned_edge=LEFT).next_to(ucs_title, DOWN)
        )
        self.play(FadeIn(ucs_title), FadeIn(ucs_details))
        self.wait()

        # Create a line with greedy approach on the left, and UCS on the right
        # midpoint is A* search
        line = Line(LEFT * 4.5, RIGHT * 4.5, color=REDUCIBLE_PURPLE).move_to(DOWN * 2.5)
        left_tick = (
            Line(UP * 0.1, DOWN * 0.1)
            .move_to(line.get_start())
            .set_color(REDUCIBLE_VIOLET)
        )
        right_tick = (
            Line(UP * 0.1, DOWN * 0.1)
            .move_to(line.get_end())
            .set_color(REDUCIBLE_VIOLET)
        )
        greedy_approach = greedy_title.copy().scale(0.7).next_to(left_tick, DOWN)
        ucs_approach = ucs_title.copy().scale(0.7).next_to(right_tick, DOWN)
        self.play(
            FadeIn(line),
            FadeIn(left_tick),
            FadeIn(right_tick),
        )
        self.wait()
        self.play(FadeIn(greedy_approach), FadeIn(ucs_approach))
        middle_tick = (
            Line(UP * 0.1, DOWN * 0.1)
            .move_to(line.get_center())
            .set_color(REDUCIBLE_GREEN)
        )
        astar_text = (
            Text("A* Search", font=REDUCIBLE_FONT).scale(0.5).next_to(middle_tick, DOWN)
        )
        self.play(FadeIn(middle_tick), FadeIn(astar_text))
        self.wait()


class GreedyApproach(AstarAnimationTools):
    def construct(self):
        graph = self.get_large_random_graph(
            25,
            dist_threshold=3,
            p=0.7,
            min_distance_between_points=1,
            vertex_config={
                "stroke_color": REDUCIBLE_PURPLE,
                "stroke_width": 3,
                "fill_color": REDUCIBLE_PURPLE_DARK_FILL,
                "fill_opacity": 1,
                "radius": 0.2,  # make these slightly larger
            },
            labels=False,
        )
        self.play(FadeIn(graph))
        self.wait()

        # for mob in graph.vertices.values():
        #     self.add_foreground_mobject(mob)

        # show general direction of path from start to end
        start, end = 15, 11

        begin = self.show_start(start, graph, color=REDUCIBLE_CHARM)
        self.play(FadeIn(*begin))
        self.wait()

        goal = self.show_goal(end, graph, color=REDUCIBLE_GREEN)
        self.play(FadeIn(*goal))
        self.wait()

        arrow = (
            Arrow(graph.vertices[start].get_center(), graph.vertices[end].get_center())
            .scale(0.5)
            .set_color(REDUCIBLE_YELLOW)
        )
        self.play(FadeIn(arrow))
        self.wait()
        self.play(FadeOut(arrow))
        self.wait()

        l1 = self.show_eucledian_distance(start, end, graph)
        l2 = self.show_eucledian_distance(3, end, graph)
        l3 = self.show_eucledian_distance(8, end, graph)
        l4 = self.show_eucledian_distance(12, end, graph)

        remaining_labels = []
        for vertex in graph.vertices:
            if vertex not in [start, 3, 8, 12]:
                distance = euclidean_distance(
                    graph.vertices[vertex], graph.vertices[end]
                )
                label = Text(
                    f"{distance:.1f}",
                    font=REDUCIBLE_MONO,
                    color=WHITE,
                )
                label.move_to(graph.vertices[vertex].get_center())
                label.scale_to_fit_height(graph.vertices[vertex].height * 0.25)
                remaining_labels.append(label)
        self.play(*[FadeIn(label) for label in remaining_labels])
        self.wait()

        all_labels = [l1, l2, l3, l4] + remaining_labels
        for mob in list(graph.vertices.values()) + all_labels:
            self.add_foreground_mobject(mob)

        # Greedy nodes example
        self.show_greedy_example(start, end, graph)

        # shift all mobjects in frame by 1.5 units to the left
        self.play(*[mob.animate.shift(LEFT * 1.5) for mob in self.mobjects])
        self.wait()
        # explain why greedy approach is not optimal
        scale = 0.35
        label_B = (
            Text("B", font=REDUCIBLE_MONO, color=WHITE)
            .scale(scale)
            .next_to(graph.vertices[1], LEFT)
        )
        label_A = (
            Text("A", font=REDUCIBLE_MONO, color=WHITE)
            .scale(scale)
            .next_to(graph.vertices[4], DOWN)
        )
        label_C = (
            Text("C", font=REDUCIBLE_MONO, color=WHITE)
            .scale(scale)
            .next_to(graph.vertices[6], RIGHT)
        )
        label_G = (
            Text("G", font=REDUCIBLE_MONO, color=WHITE)
            .scale(scale)
            .next_to(graph.vertices[11], RIGHT)
        )

        self.play(FadeIn(label_A), FadeIn(label_B), FadeIn(label_C), FadeIn(label_G))
        surround_rect = SurroundingRectangle(
            VGroup(graph.vertices[1], graph.vertices[6]), color=REDUCIBLE_YELLOW
        )
        self.play(Write(surround_rect))
        self.wait()
        reason = Text(
            "Greedy approach chose C from A instead of B",
            font=REDUCIBLE_FONT,
            weight=MEDIUM,
            color=WHITE,
        )
        reason.scale(0.5).move_to(RIGHT * 3.5 + UP * 3)
        self.play(FadeIn(reason))
        self.wait()

        explanation = Text(
            "Dist(C, G) < Dist(B, G)", font=REDUCIBLE_MONO, weight=MEDIUM, color=WHITE
        ).scale(0.3)
        explanation.next_to(reason, DOWN)
        self.play(FadeIn(explanation))
        self.wait()

        reason2 = (
            Text(
                "Does not account for Dist(A, B) and Dist(A, C)",
                font=REDUCIBLE_MONO,
                weight=MEDIUM,
                color=WHITE,
            )
            .scale(0.3)
            .next_to(explanation, DOWN)
        )

        self.play(FadeIn(reason2))
        self.wait()

    def show_eucledian_distance(self, start, end, graph):
        # dotted line between start and end
        line = DashedLine(
            graph.vertices[start].get_center(),
            graph.vertices[end].get_center(),
            stroke_width=2,
            stroke_color=REDUCIBLE_YELLOW,
            stroke_opacity=0.5,
        )
        # label for the distance
        distance = euclidean_distance(graph.vertices[start], graph.vertices[end])
        label = Text(
            f"{distance:.1f}",
            font=REDUCIBLE_MONO,
            color=WHITE,
        )
        label.move_to(graph.vertices[start].get_center())
        label.scale_to_fit_height(graph.vertices[start].height * 0.25)
        self.play(Write(line))
        self.play(FadeIn(label))
        self.wait()
        self.remove(line)
        return label

    def show_greedy_example(self, start, end, graph):
        # Greedy nodes exampled
        nodes_expanded = self.show_nodes_expanded(
            graph, start, end, g_func=lambda u, v: 0, h_func=euclidean_distance
        )
        animations = []
        # exclude the first animation becase we already highlighted the start node
        for mob in nodes_expanded[1:]:
            if isinstance(mob, TipableVMobject):
                animations.append(mob.animate.set_stroke(REDUCIBLE_YELLOW, width=4))
            else:
                animations.append(FadeIn(mob))

        for anim in animations:
            self.play(anim)
        self.wait()

        optimal_path, _ = solve_astar(graph, start, end, h_func=euclidean_distance)
        # show optimal path
        optimal_path = self.show_path(
            optimal_path, graph, color=REDUCIBLE_GREEN_LIGHTER
        )
        animations = []
        for mob in optimal_path:
            if isinstance(mob, TipableVMobject):
                animations.append(
                    mob.animate.set_stroke(REDUCIBLE_GREEN_LIGHTER, width=4)
                )
            else:
                animations.append(FadeIn(mob))
        self.play(LaggedStart(*animations), run_time=5)

        self.wait()


class MotivateUniformCostSearch(AstarAnimationTools):
    def get_graph(self):
        # from https://docs.manim.community/en/stable/reference/manim.mobject.graph.Graph.html#manim.mobject.graph.Graph
        edges = []
        partitions = []
        c = 0
        layers = [2, 3, 3, 2]  # the number of neurons in each layer

        for i in layers:
            partitions.append(list(range(c + 1, c + i + 1)))
            c += i
        for i, v in enumerate(layers[1:]):
            last = sum(layers[: i + 1])
            for j in range(v):
                for k in range(last - layers[i], last):
                    edges.append((k + 1, j + last + 1))

        vertices = np.arange(1, sum(layers) + 1)

        graph = Graph(
            vertices,
            edges,
            layout="partite",
            partitions=partitions,
            layout_scale=3,
            vertex_config={"radius": 0.20},
            labels=True,
        )
        # zero index
        new_vertices = [v - 1 for v in vertices]
        new_edges = [(u - 1, v - 1) for u, v in edges]
        new_edges.remove((1, 2))
        new_edges.remove((0, 4))
        new_edges.remove((4, 5))
        new_edges.remove((2, 7))

        layout_with_small_amount_of_noise = {
            v
            - 1: graph.vertices[v].get_center()
            + np.array([np.random.normal(-0.2, 0.2), np.random.normal(-0.2, 0.2), 0])
            for v in graph.vertices
        }
        # some minor adjustments
        layout_with_small_amount_of_noise[0] += LEFT * 0.5
        layout_with_small_amount_of_noise[1] += LEFT * 0.5
        layout_with_small_amount_of_noise[2] += LEFT * 0.2
        layout_with_small_amount_of_noise[3] += LEFT * 0.5
        layout_with_small_amount_of_noise[8] += RIGHT * 0.5
        layout_with_small_amount_of_noise[9] += RIGHT * 0.5

        graph = AGraph(
            new_vertices,
            new_edges,
            layout=layout_with_small_amount_of_noise,
            labels=True,
            edge_to_prop={
                (2, 6): 0.2,
                (3, 5): 0.2,
                (3, 7): 0.3,
                (5, 8): 0.75,
                (5, 9): 0.3,
                (6, 8): 0.7,
                (6, 9): 0.2,
                (7, 8): 0.8,
            },
        )
        return graph

    def construct(self):
        graph = self.get_graph()
        
        dist_label_dict = graph.get_edge_weight_labels()
        self.play(
            FadeIn(graph),
            *[FadeIn(label) for label in dist_label_dict.values()],
        )

        self.wait()

        start = self.show_start(0, graph, color=REDUCIBLE_CHARM)
        self.play(FadeIn(*start))
        self.wait()
        goal = self.show_goal(8, graph, color=REDUCIBLE_GREEN)
        self.play(FadeIn(*goal))
        self.wait()

        # make a curved arc that splits the graph between node 8 and it's neighbords (5, 6, and 7)
        arc = ArcBetweenPoints(
            graph.vertices[8].get_center(), graph.vertices[5].get_center()
        )
        start = arc.point_from_proportion(0.5)
        end = graph.vertices[9].get_center() + UP * 0.5

        arc_split = (
            DashedVMobject(ArcBetweenPoints(start, end, angle=PI / 2))
            .shift(RIGHT * SMALL_BUFF)
            .set_color(REDUCIBLE_GREEN_LIGHTER)
        )
        self.play(FadeIn(arc_split))
        self.wait()

        # add all edge weight labels to the foreground
        for label in dist_label_dict.values():
            self.add_foreground_mobject(label)
        self.wait()

        # show neighbors of goal node
        neighbors = self.show_neighbors(8, graph)
        self.play(FadeIn(*neighbors))
        self.wait()

        # show shortest path from 0 -> 5
        path, _ = solve_astar(graph, 0, 5, h_func=euclidean_distance)
        optimal_path = self.show_path(path, graph, color=REDUCIBLE_GREEN_LIGHTER)
        animations = []
        # ignore start because it's already highlighted
        for mob in optimal_path[1:]:
            if isinstance(mob, TipableVMobject):
                animations.append(
                    mob.animate.set_stroke(REDUCIBLE_GREEN_LIGHTER, width=4)
                )
            else:
                animations.append(FadeIn(mob))
        self.play(LaggedStart(*animations), run_time=3)
        self.wait()

        # show the shortest path from 0 -> 6
        path, _ = solve_astar(graph, 0, 6, h_func=euclidean_distance)
        optimal_path = self.show_path(path, graph, color=REDUCIBLE_YELLOW)
        animations = []
        # ignore start because it's already highlighted
        for mob in optimal_path[1:]:
            if isinstance(mob, TipableVMobject):
                animations.append(
                    mob.animate.set_stroke(REDUCIBLE_YELLOW, width=4)
                )
            else:
                animations.append(FadeIn(mob))
        self.play(LaggedStart(*animations), run_time=3)
        self.wait()

        # show the shortest path from 0 -> 7
        path, _ = solve_astar(graph, 0, 7, h_func=euclidean_distance)
        optimal_path = self.show_path(path, graph, color=REDUCIBLE_ORANGE)
        animations = []
        # ignore start because it's already highlighted
        for i, mob in enumerate(optimal_path[1:]):
            if isinstance(mob, TipableVMobject):
                if i == 0:
                    # set stroke to a gradient of ORANGE and YELLOW
                    animations.append(
                        mob.animate.set_stroke(color=[REDUCIBLE_ORANGE, REDUCIBLE_YELLOW], width=4)
                    )
                else:
                    animations.append(
                        mob.animate.set_stroke(REDUCIBLE_ORANGE, width=4)
                    )
            else:
                if i == 1:
                    mob = get_glowing_surround_circle(graph.vertices[3], color=REDUCIBLE_ORANGE, buff_max=0.075)
                animations.append(FadeIn(mob))
        self.play(LaggedStart(*animations), run_time=3)
        self.wait()

        # shift up all mobjects by 1 unit
        self.play(*[mob.animate.shift(UP * 1.5) for mob in self.mobjects])
        self.wait()
        

        defintions1 = Tex(r"$\text{Let } C(A \rightarrow B) \text{ be the cost of the shortest path from } A \text{ to } B$")
        definiions2 = Tex(r"$\text{Let } d(A, B) \text{ be the distance of the edge between } A \text{ and } B$")
        defintions1.scale(0.65).next_to(graph, DOWN)
        definiions2.scale(0.65).next_to(defintions1, DOWN)
        self.play(FadeIn(defintions1))
        self.wait()
        self.play(FadeIn(definiions2))
        self.wait()

        cost_function = Tex(r"$C(0 \rightarrow 8) = min\{$", 
                            r"$C(0 \rightarrow 5)$", r"$\, + \,$", r"$d(5, 8)$", r"$,$", 
                            r"$\: C(0 \rightarrow 6)$", r"$\, + \,$", r"$d(6, 8)$", r"$,$", 
                            r"$\: C(0 \rightarrow 7)$", r"$\, + \,$", r"$d(7, 8)$", 
                            r"$\}$")
        cost_function.scale(0.65).next_to(definiions2, DOWN)
        self.play(Write(cost_function))
        self.wait()

        self.play(
            cost_function[1].animate.set_color(REDUCIBLE_GREEN_LIGHTER),
            cost_function[5].animate.set_color(REDUCIBLE_YELLOW),
            cost_function[9].animate.set_color(REDUCIBLE_ORANGE)
        )
        self.wait()

        braces1 = Brace(cost_function[1], DOWN)
        braces2 = Brace(cost_function[5], DOWN)
        braces3 = Brace(cost_function[9], DOWN)

        self.play(
            GrowFromCenter(braces1),
            GrowFromCenter(braces2),
            GrowFromCenter(braces3),
        )
        self.wait()

        c_0_5 = [Tex("2.4 + 2.0"), Tex("4.4")]
        c_0_6 = [Tex("2.2 + 2.6"), Tex("4.8")]
        c_0_7 = [Tex("2.2 + 3.1"), Tex("5.3")]

        for all_text in [c_0_5, c_0_6, c_0_7]:
            for text in all_text:
                text.scale(0.65)

        [t.next_to(braces1, DOWN, buff=SMALL_BUFF) for t in c_0_5]
        [t.next_to(braces2, DOWN, buff=SMALL_BUFF) for t in c_0_6]
        [t.next_to(braces3, DOWN, buff=SMALL_BUFF) for t in c_0_7]

        self.play(*[FadeIn(c_0_5[0]), FadeIn(c_0_6[0]), FadeIn(c_0_7[0])])
        self.wait()
        self.play(*[Transform(c_0_5[0], c_0_5[1]), Transform(c_0_6[0], c_0_6[1]), Transform(c_0_7[0], c_0_7[1])])
        self.wait()

        brace4 = Brace(cost_function[3], DOWN)
        brace5 = Brace(cost_function[7], DOWN)
        brace6 = Brace(cost_function[11], DOWN)

        self.play(GrowFromCenter(brace4), GrowFromCenter(brace5), GrowFromCenter(brace6))
        self.wait()
        d_5_8 = Tex("2.8").scale(0.65).next_to(brace4, DOWN, buff=SMALL_BUFF)
        d_6_8 = Tex("2.8").scale(0.65).next_to(brace5, DOWN, buff=SMALL_BUFF)
        d_7_8 = Tex("4.4").scale(0.65).next_to(brace6, DOWN, buff=SMALL_BUFF)

        shift_down = DOWN * 0.55
        min_func = cost_function[0][-5:].copy().shift(shift_down)
        plus1 = cost_function[2].copy().shift(shift_down)
        comma1 = cost_function[4].copy().shift(shift_down)
        plus2 = cost_function[6].copy().shift(shift_down)
        comma2 = cost_function[8].copy().shift(shift_down)
        plus3 = cost_function[10].copy().shift(shift_down)
        end_bracket = cost_function[-1].copy().shift(shift_down)
        # Transform min_func. plus1, plus2, and plus3 using TransformFromCopy
        self.play(FadeIn(d_5_8), FadeIn(d_6_8), FadeIn(d_7_8))
        self.wait()
        shift_up = UP * 0.3

        self.play(FadeOut(braces1), FadeOut(braces2), FadeOut(braces3), FadeOut(brace4), FadeOut(brace5), FadeOut(brace6),
                  TransformFromCopy(cost_function[0][-4:], min_func), 
                  TransformFromCopy(cost_function[2], plus1), 
                  TransformFromCopy(cost_function[6], plus2), 
                  TransformFromCopy(cost_function[10], plus3),
                  TransformFromCopy(cost_function[-1], end_bracket),
                  TransformFromCopy(cost_function[4], comma1),
                  TransformFromCopy(cost_function[8], comma2),
                  c_0_5[0].animate.shift(shift_up),
                  c_0_6[0].animate.shift(shift_up),
                  c_0_7[0].animate.shift(shift_up),
                  d_5_8.animate.shift(shift_up),
                  d_6_8.animate.shift(shift_up),
                  d_7_8.animate.shift(shift_up),
                  )

        self.wait()

        animations = [graph.edge_dict[(5, 8)].shift(UP * 1.5).animate.set_stroke(color=REDUCIBLE_GREEN_LIGHTER, width=4),
                      FadeIn(get_glowing_surround_circle(graph.vertices[8], color=REDUCIBLE_GREEN_LIGHTER)),]
        final_path_cost = Tex(" = 7.2").scale(0.65).next_to(min_func, DOWN, aligned_edge=LEFT)
        self.play(FadeIn(final_path_cost), LaggedStart(*animations))
        self.wait()


class UniformCostSearchDetailDemo(MotivateUniformCostSearch):

    def construct(self):
        graph = self.get_graph()
        
        dist_label_dict = graph.get_edge_weight_labels()
        self.play(
            FadeIn(graph),
            *[FadeIn(label) for label in dist_label_dict.values()],
        )

        self.wait()
        for label in dist_label_dict.values():
            self.add_foreground_mobject(label)

        start = self.show_start(0, graph, color=REDUCIBLE_CHARM)
        self.play(FadeIn(*start))
        self.wait()
        goal = self.show_goal(8, graph, color=REDUCIBLE_GREEN)
        self.play(FadeIn(*goal))
        self.wait()

        nodes_expanded = self.show_nodes_expanded(graph, 0, 8, h_func=lambda u, v: 0, include_heap_mobs=True)
        edge_animations = []
        vertex_animations = []
        heap_mobjects = []
        heaps = []
        count_vertex = 0
        for mob in nodes_expanded:
            if isinstance(mob, TipableVMobject):
                edge_animations.append(mob.animate.set_stroke(REDUCIBLE_YELLOW, width=4))
            elif isinstance(mob, tuple):
                heaps.append(mob[0])
                heap_mobjects.append(mob[1].scale(0.8).move_to(LEFT * 6).to_edge(UP))
            else:
                if count_vertex > 0:
                    vertex_animations.append(FadeIn(mob))
                count_vertex += 1
        for edge_anim, vertex_anim in zip(edge_animations, vertex_animations):
            self.play(edge_anim, vertex_anim)

        heap_animations = []
        heap_animations.append([FadeIn(heap_mobjects[0])])
        assert len(heaps) == len(heap_mobjects), "Heaps and heap mobjects are not of the same length"
        for i in range(1, len(heaps)):
            previous_heap, current_heap = heaps[i - 1], heaps[i]
            previous_heap_mob, current_heap_mob = heap_mobjects[i - 1], heap_mobjects[i]
            heap_animations.append(self.get_heap_animations(previous_heap, current_heap, previous_heap_mob, current_heap_mob))
        
        for animation_group in heap_animations:
            self.play(*animation_group)
            

    def get_heap_animations(self, previous_heap, new_heap, previous_heap_mob, new_heap_mob):
        animations = []
        for prev_element, prev_mob in zip(previous_heap, previous_heap_mob):
            for new_element, new_mob in zip(new_heap, new_heap_mob):
                prev_astar_node, new_astar_node = prev_element[1], new_element[1]
                if prev_astar_node == new_astar_node:
                    animations.append(ReplacementTransform(prev_mob, new_mob))
                    continue
        new_heap_astar_nodes = [element[1] for element in new_heap]
        for prev_element, prev_mob in zip(previous_heap, previous_heap_mob):
            if prev_element[1] not in new_heap_astar_nodes:
                animations.append(FadeOut(prev_mob, shift=RIGHT))

        previous_heap_astar_nodes = [element[1] for element in previous_heap]
        for new_element, new_mob in zip(new_heap, new_heap_mob):
            if new_element[1] not in previous_heap_astar_nodes:
                animations.append(FadeIn(new_mob))
        return animations
