import sys
import copy

### THIS IS A WORKAROUND FOR NOW
### IT REQUIRES RUNNING MANIM FROM INSIDE DIRECTORY VIDEO-DEV
sys.path.insert(1, "common/")

from manim import *
from manim.mobject.geometry.tips import ArrowTriangleFilledTip
from reducible_colors import *
from functions import *
import itertools
from solver_utils import *
from typing import Hashable, Iterable
from classes import *

np.random.seed(23)


class TSPGraph(Graph):
    def __init__(
        self,
        vertices,
        dist_matrix=None,
        vertex_config={
            "stroke_color": REDUCIBLE_PURPLE,
            "stroke_width": 3,
            "fill_color": REDUCIBLE_PURPLE,
            "fill_opacity": 0.5,
        },
        edge_config={
            "color": REDUCIBLE_VIOLET,
            "stroke_width": 3,
        },
        labels=True,
        **kwargs,
    ):
        edges = []
        if labels:
            labels = {k: CustomLabel(str(k), scale=0.6) for k in vertices}
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
            for u, v in itertools.combinations(vertices, 2):
                distance = np.linalg.norm(
                    self.vertices[u].get_center() - self.vertices[v].get_center()
                )
                self.dist_matrix[u][v] = distance
                self.dist_matrix[v][u] = distance
        else:
            self.dist_matrix = dist_matrix

    def get_all_edges(self, buff=None):
        edge_dict = {}
        for edge in itertools.combinations(self.vertices.keys(), 2):
            u, v = edge
            edge_dict[edge] = self.create_edge(u, v, buff=buff)
        return edge_dict

    def create_edge(self, u, v, buff=None):
        return Line(
            self.vertices[u].get_center(),
            self.vertices[v].get_center(),
            color=self.edge_config["color"],
            stroke_width=self.edge_config["stroke_width"],
            buff=self.edge_config["buff"] if buff is None else buff,
        )

    def get_tour_edges(self, tour):
        """
        @param: tour -- sequence of vertices where all vertices are part of the tour (no repetitions)
        """
        edges = get_edges_from_tour(tour)
        edge_dict = {}
        for edge in edges:
            u, v = edge
            edge_mob = self.create_edge(u, v)
            edge_dict[edge] = edge_mob
        return edge_dict

    def get_tour_dist_labels(self, edge_dict, scale=0.3, num_decimal_places=1):
        dist_label_dict = {}
        for edge in edge_dict:
            u, v = edge
            dist_label = self.get_dist_label(
                edge_dict[edge],
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

    def get_neighboring_edges(self, vertex):
        edges = [(vertex, other) for other in get_neighbors(vertex, len(self.vertices))]
        return {edge: self.create_edge(edge[0], edge[1]) for edge in edges}

    def get_edges_from_list(self, edges):
        edge_dict = {}
        for edge in edges:
            u, v = edge
            edge_mob = self.create_edge(u, v)
            edge_dict[edge] = edge_mob
        return edge_dict


class TSPTester(Scene):
    def construct(self):
        big_graph = TSPGraph(range(12), layout_scale=2.4, layout="circular")
        all_edges_bg = big_graph.get_all_edges()
        self.play(FadeIn(big_graph))
        self.wait()

        self.play(*[FadeIn(edge) for edge in all_edges_bg.values()])
        self.wait()


class NearestNeighbor(Scene):
    def construct(self):
        NUM_VERTICES = 10
        layout = self.get_random_layout(NUM_VERTICES)
        # MANUAL ADJUSTMENTS FOR BETTER INSTRUCTIONAL EXAMPLE
        layout[7] = RIGHT * 3.5 + UP * 2

        graph = TSPGraph(
            list(range(NUM_VERTICES)),
            layout=layout,
        )
        self.play(FadeIn(graph))
        self.wait()

        graph_with_tour_edges = self.demo_nearest_neighbor(graph)

        self.compare_nn_with_optimal(graph_with_tour_edges, graph)

        self.clear()

        self.show_many_large_graph_nn_solutions()

    def demo_nearest_neighbor(self, graph):
        glowing_circle = get_glowing_surround_circle(graph.vertices[0])
        self.play(FadeIn(glowing_circle))
        self.wait()

        neighboring_edges = graph.get_neighboring_edges(0)

        self.play(
            LaggedStartMap(Write, [(edge) for edge in neighboring_edges.values()])
        )
        self.wait()

        tour, cost = get_nearest_neighbor_solution(graph.get_dist_matrix())
        tour_edges = graph.get_tour_edges(tour)
        seen = set([tour[0]])
        prev = tour[0]
        residual_edges = {}
        for vertex in tour[1:]:
            self.play(tour_edges[(prev, vertex)].animate.set_color(REDUCIBLE_YELLOW))
            self.wait()
            seen.add(vertex)
            new_glowing_circle = get_glowing_surround_circle(graph.vertices[vertex])
            new_neighboring_edges = graph.get_neighboring_edges(vertex)
            for key in new_neighboring_edges.copy():
                if key[1] in seen and key[1] != vertex:
                    del new_neighboring_edges[key]
            filtered_prev_edges = [
                edge_key
                for edge_key, edge in neighboring_edges.items()
                if edge_key != (prev, vertex) and edge_key != (vertex, prev)
            ]
            self.play(
                FadeOut(glowing_circle),
                FadeIn(new_glowing_circle),
                *[
                    FadeOut(neighboring_edges[edge_key])
                    for edge_key in filtered_prev_edges
                ],
            )
            self.wait()
            filtered_new_edges = [
                edge_key
                for edge_key, edge in new_neighboring_edges.items()
                if edge_key != (prev, vertex) and edge_key != (vertex, prev)
            ]

            if len(filtered_new_edges) > 0:
                self.play(
                    *[
                        FadeIn(new_neighboring_edges[edge_key])
                        for edge_key in filtered_new_edges
                    ]
                )
                self.wait()
            residual_edges[(prev, vertex)] = neighboring_edges[(prev, vertex)]
            neighboring_edges = new_neighboring_edges
            glowing_circle = new_glowing_circle
            prev = vertex

        for edge in residual_edges.values():
            self.remove(edge)
        self.wait()
        # final edge connecting back to start
        tour_edges[(tour[-1], tour[0])].set_color(REDUCIBLE_YELLOW)
        self.play(
            Write(tour_edges[(tour[-1], tour[0])]),
            FadeOut(glowing_circle),
        )
        self.wait()

        graph_with_tour_edges = self.get_graph_tour_group(graph, tour_edges)
        return graph_with_tour_edges

    def compare_nn_with_optimal(self, graph_with_tour_edges, original_graph):
        nn_tour, nn_cost = get_nearest_neighbor_solution(
            original_graph.get_dist_matrix()
        )
        optimal_tour, optimal_cost = get_exact_tsp_solution(
            original_graph.get_dist_matrix()
        )
        optimal_graph = original_graph.copy()
        optimal_edges = optimal_graph.get_tour_edges(optimal_tour)

        shift_amount = 3.2
        scale = 0.6
        self.play(graph_with_tour_edges.animate.scale(scale).shift(LEFT * shift_amount))
        self.wait()

        optimal_graph_tour = self.get_graph_tour_group(optimal_graph, optimal_edges)
        optimal_graph_tour.scale(scale).shift(RIGHT * shift_amount)
        nn_text = self.get_distance_text(nn_cost).next_to(
            graph_with_tour_edges, UP, buff=1
        )
        optimal_text = self.get_distance_text(optimal_cost).next_to(
            optimal_graph_tour, UP, buff=1
        )

        self.play(FadeIn(nn_text))

        self.play(FadeIn(optimal_graph_tour))
        self.wait()

        self.play(
            FadeIn(optimal_text),
        )
        self.wait()

        nn_heuristic = Text(
            "Nearest Neighbor (NN) Heuristic", font=REDUCIBLE_FONT, weight=BOLD
        )
        nn_heuristic.scale(0.8)
        nn_heuristic.move_to(DOWN * 2.5)

        self.play(Write(nn_heuristic))
        self.wait()

        surround_rect_0_2_3_5_original = SurroundingRectangle(
            VGroup(
                *[
                    original_graph.vertices[0],
                    original_graph.vertices[2],
                    original_graph.vertices[3],
                    original_graph.vertices[5],
                ]
            )
        ).set_color(REDUCIBLE_CHARM)

        surround_rect_0_2_3_5_optimal = SurroundingRectangle(
            VGroup(
                *[
                    optimal_graph.vertices[0],
                    optimal_graph.vertices[2],
                    optimal_graph.vertices[3],
                    optimal_graph.vertices[5],
                ]
            )
        ).set_color(REDUCIBLE_GREEN_LIGHTER)

        surround_rect_1_4_6_8_original = SurroundingRectangle(
            VGroup(
                *[
                    original_graph.vertices[1],
                    original_graph.vertices[4],
                    original_graph.vertices[6],
                    original_graph.vertices[8],
                ]
            )
        ).set_color(REDUCIBLE_CHARM)

        surround_rect_1_4_6_8_optimal = SurroundingRectangle(
            VGroup(
                *[
                    optimal_graph.vertices[1],
                    optimal_graph.vertices[4],
                    optimal_graph.vertices[6],
                    optimal_graph.vertices[8],
                ]
            )
        ).set_color(REDUCIBLE_GREEN_LIGHTER)

        self.play(
            Write(surround_rect_0_2_3_5_optimal),
            Write(surround_rect_0_2_3_5_original),
        )
        self.wait()

        self.play(
            Write(surround_rect_1_4_6_8_optimal),
            Write(surround_rect_1_4_6_8_original),
        )
        self.wait()

        self.play(
            FadeOut(surround_rect_0_2_3_5_optimal),
            FadeOut(surround_rect_0_2_3_5_original),
            FadeOut(surround_rect_1_4_6_8_optimal),
            FadeOut(surround_rect_1_4_6_8_original),
        )
        self.wait()

        how_to_compare = Text(
            "How to measure effectiveness of heuristic approach?", font=REDUCIBLE_FONT
        ).scale(0.6)

        how_to_compare.next_to(nn_heuristic, DOWN)

        self.play(FadeIn(how_to_compare))
        self.wait()

        self.play(FadeOut(nn_heuristic), FadeOut(how_to_compare))

        approx_ratio = (
            Tex(
                r"Approximation ratio $(\alpha) = \frac{\text{heuristic solution}}{\text{optimal solution}}$"
            )
            .scale(0.8)
            .move_to(DOWN * 2.5)
        )

        self.play(FadeIn(approx_ratio))

        self.wait()

        example = Tex(
            r"E.g $\alpha = \frac{28.2}{27.0} \approx 1.044$",
            r"$\rightarrow$ 4.4\% above optimal",
        ).scale(0.7)

        example.next_to(approx_ratio, DOWN)

        self.play(Write(example[0]))
        self.wait()

        self.play(Write(example[1]))
        self.wait()

    def show_many_large_graph_nn_solutions(self):
        NUM_VERTICES = 100
        num_iterations = 10
        average_case = (
            Tex(
                r"On average: $\frac{\text{NN Heuristic}}{\text{1-Tree Lower Bound}} = 1.25$"
            )
            .scale(0.8)
            .move_to(DOWN * 3.5)
        )
        for _ in range(num_iterations):
            graph = TSPGraph(
                list(range(NUM_VERTICES)),
                labels=False,
                layout=self.get_random_layout(NUM_VERTICES),
            )
            tour, nn_cost = get_nearest_neighbor_solution(graph.get_dist_matrix())
            print("NN cost", nn_cost)
            tour_edges = graph.get_tour_edges(tour)
            tour_edges_group = VGroup(*list(tour_edges.values()))
            graph_with_tour_edges = VGroup(graph, tour_edges_group).scale(0.8)
            self.add(graph_with_tour_edges)
            if _ == 5:
                self.play(FadeIn(average_case))
            self.wait()
            self.remove(graph_with_tour_edges)

    def get_distance_text(self, cost, num_decimal_places=1, scale=0.6):
        cost = np.round(cost, num_decimal_places)
        return Text(f"Distance = {cost}", font=REDUCIBLE_MONO).scale(scale)

    def get_random_layout(self, N):
        random_points_in_frame = get_random_points_in_frame(N)
        return {v: point for v, point in zip(range(N), random_points_in_frame)}

    def get_graph_tour_group(self, graph, tour_edges):
        return VGroup(*[graph] + list(tour_edges.values()))

    def label_vertices_for_debugging(self, graph):
        labels = VGroup()
        for v, v_mob in graph.vertices.items():
            label = (
                Text(str(v), font=REDUCIBLE_MONO).scale(0.2).move_to(v_mob.get_center())
            )
            labels.add(label)

        return labels


class LowerBoundTSP(NearestNeighbor):
    def construct(self):
        graph = self.get_graph_with_random_layout(200, radius=0.05)
        tour, nn_cost = get_nearest_neighbor_solution(graph.get_dist_matrix())
        edge_ordering = get_edges_from_tour(tour)
        tour_edges = graph.get_tour_edges(tour)

        self.scale_graph_with_tour(graph, tour_edges, 0.8)

        self.play(
            LaggedStartMap(GrowFromCenter, list(graph.vertices.values())), run_time=2
        )
        self.wait()

        self.play(
            LaggedStartMap(Write, [tour_edges[edge] for edge in edge_ordering]),
            run_time=10,
        )
        self.wait()

        problem = (
            Text(
                "Given any solution, no efficient way to verify optimality!",
                font=REDUCIBLE_FONT,
            )
            .scale(0.5)
            .move_to(DOWN * 3.5)
        )

        self.play(FadeIn(problem))
        self.wait()

        self.clear()

        self.present_lower_bound_idea()

        self.clear()

        tsp_graph, mst_tree, mst_edge_dict = self.intro_mst()

        self.intro_1_tree(tsp_graph, mst_tree, mst_edge_dict)

    def present_lower_bound_idea(self):
        heuristic_solution_mod = Module(["Heuristic", "Solution"], text_weight=BOLD)

        optimal_solution_mod = Module(
            ["Optimal", "Solution"],
            REDUCIBLE_GREEN_DARKER,
            REDUCIBLE_GREEN_LIGHTER,
            text_weight=BOLD,
        )

        lower_bound_mod = Module(
            ["Lower", "Bound"],
            REDUCIBLE_YELLOW_DARKER,
            REDUCIBLE_YELLOW,
            text_weight=BOLD,
        )
        left_geq = MathTex(r"\geq").scale(2)
        VGroup(heuristic_solution_mod, left_geq, optimal_solution_mod).arrange(
            RIGHT, buff=1
        )

        self.play(
            FadeIn(heuristic_solution_mod),
            FadeIn(optimal_solution_mod),
            FadeIn(left_geq),
        )
        self.wait()

        right_geq = MathTex(r"\geq").scale(2)
        new_configuration = (
            VGroup(
                heuristic_solution_mod.copy(),
                left_geq.copy(),
                optimal_solution_mod.copy(),
                right_geq,
                lower_bound_mod,
            )
            .arrange(RIGHT, buff=1)
            .scale(0.7)
        )

        self.play(
            Transform(heuristic_solution_mod, new_configuration[0]),
            Transform(left_geq, new_configuration[1]),
            Transform(optimal_solution_mod, new_configuration[2]),
        )

        self.play(FadeIn(right_geq), FadeIn(lower_bound_mod))
        self.wait()

        curved_arrow_1 = (
            CustomCurvedArrow(
                heuristic_solution_mod.get_top(),
                optimal_solution_mod.get_top(),
                angle=-PI / 4,
            )
            .shift(UP * MED_SMALL_BUFF)
            .set_color(GRAY)
        )

        curved_arrow_2 = (
            CustomCurvedArrow(
                heuristic_solution_mod.get_bottom(),
                lower_bound_mod.get_bottom(),
                angle=PI / 4,
            )
            .shift(DOWN * MED_SMALL_BUFF)
            .set_color(GRAY)
        )

        inefficient_comparison = (
            Text("Intractable comparison", font=REDUCIBLE_FONT)
            .scale(0.6)
            .next_to(curved_arrow_1, UP)
        )
        reasonable_comparison = (
            Text("Reasonable comparison", font=REDUCIBLE_FONT)
            .scale(0.6)
            .next_to(curved_arrow_2, DOWN)
        )
        self.play(Write(curved_arrow_1), FadeIn(inefficient_comparison))
        self.wait()

        self.play(Write(curved_arrow_2), FadeIn(reasonable_comparison))
        self.wait()

        good_lower_bound = (
            Tex(
                r"Good lower bound: maximize $\frac{\text{lower bound}}{\text{optimal}}$"
            )
            .scale(0.8)
            .move_to(UP * 3)
        )

        self.play(FadeIn(good_lower_bound))
        self.wait()

    def intro_mst(self):
        title = (
            Text("Minimum Spanning Tree (MST)", font=REDUCIBLE_FONT, weight=BOLD)
            .scale(0.8)
            .move_to(UP * 3.5)
        )
        NUM_VERTICES = 9
        graph = TSPGraph(
            list(range(NUM_VERTICES)), layout=self.get_random_layout(NUM_VERTICES)
        )
        original_scaled_graph = graph.copy()
        not_connected_graph = graph.copy()
        cycle_graph = graph.copy()
        non_mst_graph = graph.copy()

        mst_edges, cost = get_mst(graph.get_dist_matrix())
        mst_edges_mob = graph.get_edges_from_list(mst_edges)
        mst_edges_group = VGroup(*list(mst_edges_mob.values()))
        mst_tree = VGroup(graph, mst_edges_group)
        mst_tree.scale(0.8)
        self.play(Write(title), FadeIn(graph))
        self.wait()
        self.play(*[GrowFromCenter(edge) for edge in mst_edges_group])
        self.wait()

        definition = (
            Text(
                "Set of edges that connect all vertices with minimum distance and no cycles",
                font=REDUCIBLE_FONT,
            )
            .scale(0.5)
            .move_to(DOWN * 3.5)
        )
        definition[14:21].set_color(REDUCIBLE_YELLOW)
        definition[36:51].set_color(REDUCIBLE_YELLOW)
        definition[-8:].set_color(REDUCIBLE_YELLOW)
        self.play(FadeIn(definition))
        self.wait()

        true_mst_text = Text("True MST", font=REDUCIBLE_FONT).scale(0.7)
        self.play(mst_tree.animate.scale(0.75).shift(LEFT * 3.5))
        true_mst_text.next_to(mst_tree, DOWN)
        self.play(FadeIn(true_mst_text))
        self.wait()

        to_remove_edge = (8, 6)
        mst_edges.remove(to_remove_edge)
        not_connected_edge = not_connected_graph.get_edges_from_list(mst_edges)
        not_connect_graph_group = (
            VGroup(*[not_connected_graph] + list(not_connected_edge.values()))
            .scale(0.6)
            .shift(RIGHT * 3.5)
        )

        not_connected_text = (
            Text("Not connected", font=REDUCIBLE_FONT)
            .scale(0.7)
            .next_to(not_connect_graph_group, DOWN)
        )
        self.play(FadeIn(not_connect_graph_group), FadeIn(not_connected_text))

        surround_rect = SurroundingRectangle(
            VGroup(not_connected_graph.vertices[8], not_connected_graph.vertices[6]),
            color=REDUCIBLE_CHARM,
        )
        self.play(
            Write(surround_rect),
        )
        self.wait()

        to_add_edge = (6, 2)
        prev_removed_edge = not_connected_graph.create_edge(
            to_remove_edge[0],
            to_remove_edge[1],
            buff=graph.vertices[0].width / 2 + SMALL_BUFF / 4,
        )
        new_edge = not_connected_graph.create_edge(
            to_add_edge[0],
            to_add_edge[1],
            buff=graph.vertices[0].width / 2 + SMALL_BUFF / 4,
        )

        cyclic_text = (
            Text("Has cycle", font=REDUCIBLE_FONT)
            .scale(0.7)
            .move_to(not_connected_text.get_center())
        )
        self.play(
            FadeOut(surround_rect),
            Write(prev_removed_edge),
            Write(new_edge),
            ReplacementTransform(not_connected_text, cyclic_text),
        )
        new_surround_rect = SurroundingRectangle(
            VGroup(
                not_connected_graph.vertices[8],
                not_connected_graph.vertices[6],
                not_connected_graph.vertices[2],
                not_connected_graph.vertices[0],
            ),
            color=REDUCIBLE_CHARM,
        )
        self.play(Write(new_surround_rect))
        self.wait()

        non_optimal_edge = not_connected_graph.create_edge(
            5, 7, buff=graph.vertices[0].width / 2 + SMALL_BUFF / 4
        )
        non_optimal_edge.set_color(REDUCIBLE_CHARM)
        non_optimal_text = (
            Text("Spanning tree, but not minimum", font=REDUCIBLE_FONT)
            .scale(0.6)
            .move_to(cyclic_text.get_center())
        )
        self.play(
            FadeOut(new_surround_rect),
            FadeOut(new_edge),
            FadeOut(not_connected_edge[(5, 1)]),
            Write(non_optimal_edge),
            ReplacementTransform(cyclic_text, non_optimal_text),
        )
        self.wait()

        self.clear()

        mst_tree, mst_edge_dict = self.demo_prims_algorithm(
            original_scaled_graph.copy()
        )

        return original_scaled_graph, mst_tree, mst_edge_dict

    def demo_prims_algorithm(self, graph):
        visited = set([0])
        unvisited = set(graph.vertices.keys()).difference(visited)
        all_edges = graph.get_all_edges()
        VGroup(graph, VGroup(*list(all_edges.values()))).scale(0.8).shift(UP * 0.5)
        self.play(
            FadeIn(graph),
        )
        self.wait()

        (
            visited_group,
            unvisited_group,
            visited_dict,
            unvisited_dict,
        ) = self.highlight_visited_univisited(
            graph.vertices, graph.labels, visited, unvisited
        )
        visited_label = (
            Text("Visited", font=REDUCIBLE_FONT).scale(0.5).next_to(visited_group, UP)
        )
        unvisited_label = (
            Text("Unvisited", font=REDUCIBLE_FONT)
            .scale(0.5)
            .next_to(unvisited_group, UP)
        )
        self.play(
            FadeIn(visited_group),
            FadeIn(unvisited_group),
            FadeIn(visited_label),
            FadeIn(unvisited_label),
        )
        self.wait()
        iteration = 0
        highlight_animations = []
        for v in graph.vertices:
            if v in visited:
                highlighted_v = graph.vertices[v].copy()
                highlighted_v[0].set_fill(opacity=0.5).set_stroke(opacity=1)
                highlighted_v[1].set_fill(opacity=1)
                highlight_animations.append(Transform(graph.vertices[v], highlighted_v))
            else:
                un_highlighted_v = graph.vertices[v].copy()
                un_highlighted_v[0].set_fill(opacity=0.2).set_stroke(opacity=0.2)
                un_highlighted_v[1].set_fill(opacity=0.2)
                highlight_animations.append(
                    Transform(graph.vertices[v], un_highlighted_v)
                )
        self.play(*highlight_animations)
        self.wait()
        mst_edges = VGroup()
        mst_edge_dict = {}
        while len(unvisited) > 0:
            neighboring_edges = self.get_neighboring_edges_across_sets(
                visited, unvisited
            )
            for i, edge in enumerate(neighboring_edges):
                if edge not in all_edges:
                    neighboring_edges[i] = (edge[1], edge[0])
            neighboring_edges_mobs = [
                all_edges[edge].set_stroke(opacity=0.3) for edge in neighboring_edges
            ]
            self.play(*[Write(edge) for edge in neighboring_edges_mobs])
            self.wait()
            best_neighbor_edge = min(
                neighboring_edges, key=lambda x: graph.get_dist_matrix()[x[0]][x[1]]
            )
            next_vertex = (
                best_neighbor_edge[1]
                if best_neighbor_edge[1] not in visited
                else best_neighbor_edge[0]
            )
            print("Best neighbor", best_neighbor_edge)
            print("Next vertex", next_vertex)
            self.play(
                ShowPassingFlash(
                    all_edges[best_neighbor_edge]
                    .copy()
                    .set_stroke(width=6)
                    .set_color(REDUCIBLE_YELLOW),
                    time_width=0.5,
                ),
            )
            self.play(
                all_edges[best_neighbor_edge].animate.set_stroke(
                    opacity=1, color=REDUCIBLE_YELLOW
                )
            )
            mst_edges.add(all_edges[best_neighbor_edge])
            mst_edge_dict[best_neighbor_edge] = all_edges[best_neighbor_edge]
            self.wait()

            visited.add(next_vertex)
            unvisited.remove(next_vertex)

            (
                _,
                _,
                new_visited_dict,
                new_unvisited_dict,
            ) = self.highlight_visited_univisited(
                graph.vertices, graph.labels, visited, unvisited
            )
            print(type(graph.vertices[next_vertex][1]))
            highlight_next_vertex = graph.vertices[next_vertex].copy()
            highlight_next_vertex[0].set_fill(opacity=0.5).set_stroke(opacity=1)
            highlight_next_vertex[1].set_fill(opacity=1)
            self.play(
                FadeOut(
                    *[
                        all_edges[edge]
                        for edge in neighboring_edges
                        if edge != best_neighbor_edge
                    ]
                ),
                Transform(graph.vertices[next_vertex], highlight_next_vertex),
                *[
                    Transform(visited_dict[v], new_visited_dict[v])
                    for v in visited.difference(set([next_vertex]))
                ],
                *[
                    Transform(unvisited_dict[v], new_unvisited_dict[v])
                    for v in unvisited
                ],
                ReplacementTransform(
                    unvisited_dict[next_vertex], new_visited_dict[next_vertex]
                ),
            )
            self.wait()
            visited_dict[next_vertex] = new_visited_dict[next_vertex]
            del unvisited_dict[next_vertex]

        self.play(
            FadeOut(visited_label),
            FadeOut(unvisited_label),
            *[FadeOut(mob) for mob in visited_dict.values()],
        )
        self.wait()

        mst_tree = VGroup(graph, mst_edges)
        return mst_tree, mst_edge_dict

    def intro_1_tree(self, tsp_graph, mst_tree, mst_edge_dict):
        optimal_tour, optimal_cost = get_exact_tsp_solution(tsp_graph.get_dist_matrix())
        tsp_tour_edges = tsp_graph.get_tour_edges(optimal_tour)
        tsp_tour_edges_group = VGroup(*[edge for edge in tsp_tour_edges.values()])
        tsp_graph_with_tour = VGroup(tsp_graph, tsp_tour_edges_group)
        self.play(mst_tree.animate.scale(0.75).move_to(LEFT * 3.5 + UP * 1))
        self.wait()
        tsp_graph_with_tour.scale_to_fit_height(mst_tree.height).move_to(
            RIGHT * 3.5 + UP * 1
        )

        self.play(FadeIn(tsp_graph_with_tour))
        self.wait()

        mst_cost = Tex(r"MST Cost $<$ TSP Cost").move_to(DOWN * 2)
        self.play(FadeIn(mst_cost))
        self.wait()

        remove_edge = (
            Tex(r"Remove any edge from TSP tour $\rightarrow$ spanning tree $T$")
            .scale(0.7)
            .next_to(mst_cost, DOWN)
        )
        self.play(FadeIn(remove_edge))
        self.wait()
        result = Tex(r"MST cost $\leq$ cost($T$)").scale(0.7)
        result.next_to(remove_edge, DOWN)
        prev_edge = None
        for i, edge in enumerate(tsp_tour_edges):
            if i == 0:
                self.play(FadeOut(tsp_tour_edges[edge]))
            else:
                self.play(
                    FadeIn(tsp_tour_edges[prev_edge]), FadeOut(tsp_tour_edges[edge])
                )
            prev_edge = edge
            self.wait()

        self.play(FadeIn(result))
        self.wait()

        better_lower_bound = (
            Text("Better Lower Bound", font=REDUCIBLE_FONT, weight=BOLD)
            .scale_to_fit_height(mst_cost.height - SMALL_BUFF)
            .move_to(mst_cost.get_center())
            .shift(UP * SMALL_BUFF)
        )
        mst_vertices, mst_edges = mst_tree
        self.play(
            FadeIn(tsp_tour_edges[prev_edge]),
            FadeOut(result),
            FadeOut(remove_edge),
            FadeOut(mst_edges),
            FadeTransform(mst_cost, better_lower_bound),
        )
        self.wait()

        step_1 = Tex(r"1. Remove any vertex $v$ and find MST").scale(0.6)
        step_2 = Tex(r"2. Connect two shortest edges to $v$").scale(0.6)
        steps = VGroup(step_1, step_2).arrange(DOWN, aligned_edge=LEFT)
        steps.next_to(better_lower_bound, DOWN)
        self.play(FadeIn(step_1))
        self.wait()

        self.play(FadeOut(mst_vertices.vertices[6]))
        self.wait()

        mst_tree_edges_removed, cost, one_tree_edges, one_tree_cost = get_1_tree(
            mst_vertices.get_dist_matrix(), 6
        )
        all_edges = mst_vertices.get_all_edges(buff=mst_vertices[0].width / 2)
        self.play(
            *[
                GrowFromCenter(
                    self.get_edge(all_edges, edge).set_color(REDUCIBLE_YELLOW)
                )
                for edge in mst_tree_edges_removed
            ]
        )
        self.wait()

        self.play(FadeIn(step_2))
        self.wait()

        self.play(FadeIn(mst_vertices.vertices[6]))
        self.wait()

        self.play(
            *[
                GrowFromCenter(
                    self.get_edge(all_edges, edge).set_color(REDUCIBLE_YELLOW)
                )
                for edge in one_tree_edges
                if edge not in mst_tree_edges_removed
            ]
        )
        self.wait()

        new_result = Tex(r"1-tree cost $\leq$ TSP cost").scale(0.7)
        new_result.next_to(steps, DOWN)
        new_result[0][:6].set_color(REDUCIBLE_YELLOW)
        self.play(FadeIn(new_result))
        self.wait()

        unhiglighted_nodes = {
            v: tsp_graph.vertices[v].copy() for v in tsp_graph.vertices if v != 6
        }
        highlighted_nodes = copy.deepcopy(unhiglighted_nodes)
        for node in unhiglighted_nodes.values():
            node[0].set_fill(opacity=0.2).set_stroke(opacity=0.2)
            node[1].set_fill(opacity=0.2)

        unhiglighted_nodes_mst = {
            v: mst_vertices.vertices[v].copy() for v in mst_vertices.vertices if v != 6
        }
        highlighted_nodes_mst = copy.deepcopy(unhiglighted_nodes_mst)
        for node in unhiglighted_nodes_mst.values():
            node[0].set_fill(opacity=0.2).set_stroke(opacity=0.2)
            node[1].set_fill(opacity=0.2)

        self.play(
            *[
                Transform(tsp_graph.vertices[v], unhiglighted_nodes[v])
                for v in tsp_graph.vertices
                if v != 6
            ],
            *[
                tsp_tour_edges[edge].animate.set_stroke(opacity=0.2)
                for edge in tsp_tour_edges
                if 6 not in edge
            ],
            *[
                Transform(mst_vertices.vertices[v], unhiglighted_nodes_mst[v])
                for v in mst_vertices.vertices
                if v != 6
            ],
            *[
                self.get_edge(all_edges, edge).animate.set_stroke(opacity=0.2)
                for edge in one_tree_edges
                if 6 not in edge
            ],
        )

        self.wait()
        node_6_faded = mst_vertices.vertices[6].copy()
        original_node_6 = mst_vertices.vertices[6].copy()
        node_6_faded[0].set_fill(opacity=0.2).set_stroke(opacity=0.2)
        node_6_faded[1].set_fill(opacity=0.2)

        node_6_faded_tsp = tsp_graph.vertices[6].copy()
        original_node_6_tsp = tsp_graph.vertices[6].copy()
        node_6_faded_tsp[0].set_fill(opacity=0.2).set_stroke(opacity=0.2)
        node_6_faded_tsp[1].set_fill(opacity=0.2)
        self.play(
            *[
                Transform(tsp_graph.vertices[v], highlighted_nodes[v])
                for v in tsp_graph.vertices
                if v != 6
            ],
            *[
                tsp_tour_edges[edge].animate.set_stroke(opacity=1)
                for edge in tsp_tour_edges
                if 6 not in edge
            ],
            *[
                Transform(mst_vertices.vertices[v], highlighted_nodes_mst[v])
                for v in mst_vertices.vertices
                if v != 6
            ],
            *[
                self.get_edge(all_edges, edge).animate.set_stroke(opacity=1)
                for edge in one_tree_edges
                if 6 not in edge
            ],
            Transform(mst_vertices.vertices[6], node_6_faded),
            Transform(tsp_graph.vertices[6], node_6_faded_tsp),
            *[
                tsp_tour_edges[edge].animate.set_stroke(opacity=0.2)
                for edge in tsp_tour_edges
                if 6 in edge
            ],
            *[
                self.get_edge(all_edges, edge).animate.set_stroke(opacity=0.2)
                for edge in one_tree_edges
                if 6 in edge
            ],
        )
        self.wait()

        self.play(
            Transform(mst_vertices.vertices[6], original_node_6),
            Transform(tsp_graph.vertices[6], original_node_6_tsp),
            *[
                tsp_tour_edges[edge].animate.set_stroke(opacity=1)
                for edge in tsp_tour_edges
                if 6 in edge
            ],
            *[
                self.get_edge(all_edges, edge).animate.set_stroke(opacity=1)
                for edge in one_tree_edges
                if 6 in edge
            ],
        )
        self.wait()
        best_one_cost = one_tree_cost
        best_one_tree = 6
        current_one_tree_edges = VGroup(
            *[self.get_edge(all_edges, edge) for edge in one_tree_edges]
        )
        for v_to_ignore in [0, 1, 2, 3, 4, 7, 8, 5]:
            _, _, one_tree_edges, one_tree_cost = get_1_tree(
                mst_vertices.get_dist_matrix(), v_to_ignore
            )
            if one_tree_cost > best_one_cost:
                best_one_tree, best_one_cost = v_to_ignore, one_tree_cost
            all_edges_new = mst_vertices.get_all_edges(
                buff=mst_vertices.vertices[0].width / 2
            )
            new_one_tree_edges = VGroup(
                *[
                    self.get_edge(all_edges_new, edge).set_color(REDUCIBLE_YELLOW)
                    for edge in one_tree_edges
                ]
            )
            self.play(Transform(current_one_tree_edges, new_one_tree_edges))
            self.wait()

        print(
            "Best one tree", best_one_tree, best_one_cost, "Optimal TSP", optimal_cost
        )
        best_cost_1_tree = Text(
            f"Largest 1-tree cost: {np.round(best_one_cost, 1)}", font=REDUCIBLE_MONO
        ).scale(0.5)
        optimal_tsp_cost = Text(
            f"Optimal TSP cost: {np.round(optimal_cost, 1)}", font=REDUCIBLE_MONO
        ).scale(0.5)

        best_cost_1_tree.next_to(mst_tree, DOWN, buff=1)
        optimal_tsp_cost.next_to(tsp_graph, DOWN, buff=1)

        self.play(
            FadeIn(best_cost_1_tree),
            FadeIn(optimal_tsp_cost),
            FadeOut(better_lower_bound),
            FadeOut(new_result),
            FadeOut(steps),
        )
        self.wait()

    def highlight_visited_univisited(
        self, vertices, labels, visited, unvisited, scale=0.7
    ):
        visited_group = VGroup(
            *[vertices[v].copy().scale(scale) for v in visited]
        ).arrange(RIGHT)
        unvisited_group = VGroup(
            *[vertices[v].copy().scale(scale) for v in unvisited]
        ).arrange(RIGHT)
        visited_group.move_to(LEFT * 3.5 + DOWN * 3.5)
        unvisited_group.move_to(RIGHT * 3.5 + DOWN * 3.5)
        for mob in visited_group:
            mob[0].set_fill(opacity=0.5).set_stroke(opacity=1)
            mob[1].set_fill(opacity=1)
        for mob in unvisited_group:
            mob[0].set_fill(opacity=0.2).set_stroke(opacity=0.2)
            mob[1].set_fill(opacity=0.2)
        visited_dict = {v: visited_group[i] for i, v in enumerate(visited)}
        unvisited_dict = {v: unvisited_group[i] for i, v in enumerate(unvisited)}

        return visited_group, unvisited_group, visited_dict, unvisited_dict

    def get_neighboring_edges_across_sets(self, set1, set2):
        edges = []
        for v in set1:
            for u in set2:
                edges.append((v, u))
        return edges

    def get_graph_with_random_layout(self, N, radius=0.1):
        graph = TSPGraph(
            list(range(N)),
            labels=False,
            layout=self.get_random_layout(N),
            vertex_config={
                "stroke_color": REDUCIBLE_PURPLE,
                "stroke_width": 3,
                "fill_color": REDUCIBLE_PURPLE,
                "fill_opacity": 0.5,
                "radius": radius,
            },
            edge_config={
                "color": REDUCIBLE_VIOLET,
                "stroke_width": 2,
            },
        )
        return graph

    def scale_graph_with_tour(self, graph, tour_edges, scale):
        tour_edges_group = VGroup(*list(tour_edges.values()))
        graph_with_tour_edges = VGroup(graph, tour_edges_group).scale(scale)
        return graph_with_tour_edges

    def is_equal(self, edge1, edge2):
        return edge1 == edge2 or (edge1[1], edge1[0]) == edge2

    def get_edge(self, edge_dict, edge):
        if edge in edge_dict:
            return edge_dict[edge]
        else:
            return edge_dict[(edge[1], edge[0])]


class GreedyApproach(LowerBoundTSP):
    def construct(self):
        self.show_greedy_algorithm()

    def show_greedy_algorithm(self):
        np.random.seed(9)
        NUM_VERTICES = 8
        layout = self.get_random_layout(NUM_VERTICES)
        layout[0] += UR * 0.2 + RIGHT * 0.3
        graph = TSPGraph(list(range(NUM_VERTICES)), layout=layout).scale(0.8)

        title = (
            Text("Greedy Heuristic", font=REDUCIBLE_FONT, weight=BOLD)
            .scale(0.8)
            .move_to(UP * 3.5)
        )

        self.play(
            LaggedStart(*[GrowFromCenter(v) for v in graph.vertices.values()]),
            run_time=2,
        )
        self.wait()

        all_edges = graph.get_all_edges(buff=graph.vertices[0].width / 2)

        self.play(*[Write(edge.set_stroke(opacity=0.3)) for edge in all_edges.values()])
        self.wait()

        self.play(*[Unwrite(edge) for edge in all_edges.values()])
        self.wait()

        self.perform_algorithm(graph)

    def perform_algorithm(self, graph):
        all_edges = graph.get_all_edges(buff=graph.vertices[0].width / 2)
        edges_sorted = sorted(
            [edge for edge in all_edges],
            key=lambda x: graph.get_dist_matrix()[x[0]][x[1]],
        )
        added_edges = []
        for edge in edges_sorted:
            degree_map = self.get_degree_map(graph, added_edges)
            if len(added_edges) == len(graph.vertices) - 1:
                degrees_sorted = sorted(
                    list(degree_map.keys()), key=lambda x: degree_map[x]
                )
                final_edge = (degrees_sorted[0], degrees_sorted[1])
                added_edges.append(final_edge)
                edge_mob = all_edges[final_edge].set_stroke(color=REDUCIBLE_YELLOW)
                self.play(Write(edge_mob))
                self.wait()
                break
            u, v = edge
            edge_mob = all_edges[edge].set_stroke(color=REDUCIBLE_YELLOW)
            if degree_map[u] == 2 or degree_map[v] == 2:
                self.play(Write(edge_mob.set_stroke(color=REDUCIBLE_CHARM)))
                self.wait()
                # show degree issue
                if 3 in edge and 4 in edge:
                    surround_rects = [
                        SurroundingRectangle(graph.vertices[i]).set_color(
                            REDUCIBLE_CHARM
                        )
                        for i in [3, 4]
                    ]
                    degree_comment = (
                        Tex(r"Degree $>$ 2").scale(0.7).next_to(surround_rects[0], UP)
                    )
                    self.play(
                        *[Write(rect) for rect in surround_rects],
                        FadeIn(degree_comment),
                    )
                    self.wait()

                    self.play(
                        FadeOut(edge_mob),
                        FadeOut(degree_comment),
                        *[FadeOut(rect) for rect in surround_rects],
                    )
                    self.wait()
                    continue

                self.play(
                    FadeOut(edge_mob),
                )
                self.wait()
                continue

            if self.is_connected(u, v, added_edges):
                print(u, v, "is connected already, so would cause cycle")
                # would create cycle
                self.play(Write(edge_mob.set_stroke(color=REDUCIBLE_CHARM)))
                self.wait()

                surround_rect = SurroundingRectangle(
                    VGroup(
                        graph.vertices[0],
                        graph.vertices[2],
                        graph.vertices[4],
                    )
                ).set_color(REDUCIBLE_CHARM)

                cycle = (
                    Text("Cycle", font=REDUCIBLE_FONT)
                    .scale(0.6)
                    .next_to(surround_rect, UP)
                )

                self.play(Write(surround_rect), FadeIn(cycle))
                self.wait()

                self.play(FadeOut(edge_mob), FadeOut(surround_rect), FadeOut(cycle))
                self.wait()
                continue
            added_edges.append(edge)
            self.play(Write(edge_mob))
            self.wait()

    def get_degree_map(self, graph, edges):
        v_to_degree = {v: 0 for v in graph.vertices}
        for edge in edges:
            u, v = edge
            v_to_degree[u] = v_to_degree.get(u, 0) + 1
            v_to_degree[v] = v_to_degree.get(v, 0) + 1
        return v_to_degree

    def is_connected(self, u, v, edges):
        visited = set()

        def dfs(u):
            visited.add(u)
            for v in self.get_neighbors(u, edges):
                if v not in visited:
                    dfs(v)

        dfs(u)
        print("visited", visited)
        return v in visited

    def get_neighbors(self, v, edges):
        neighbors = []
        for edge in edges:
            if v not in edge:
                continue
            neighbors.append(edge[0] if edge[0] != v else edge[1])
        return neighbors


class GreedApproachExtraText(Scene):
    def construct(self):
        title = Text(
            "Greedy Heuristic Approach", font=REDUCIBLE_FONT, weight=BOLD
        ).scale(0.8)
        title.move_to(UP * 3.5)
        average_case = (
            Tex(
                r"On average: $\frac{\text{Greedy Heuristic}}{\text{1-Tree Lower Bound}} = 1.17$"
            )
            .scale(0.8)
            .move_to(DOWN * 3.5)
        )
        self.play(Write(title))
        self.wait()

        self.play(FadeIn(average_case))
        self.wait()

        self.clear()

        screen_rect_left = ScreenRectangle(height=3)
        screen_rect_right = ScreenRectangle(height=3)
        screen_rects = VGroup(screen_rect_left, screen_rect_right).arrange(
            RIGHT, buff=1
        )
        self.play(FadeIn(screen_rects))
        self.wait()


class Christofides(GreedyApproach):
    def construct(self):
        self.show_christofides()

    def show_christofides(self):
        (
            left_graph,
            mst_edges_mob,
            right_graph,
            tsp_edges_mob,
            left_edges,
            right_edges,
            mst_edges,
            tsp_tour_edges,
        ) = self.mst_step()

        (
            vertices_to_match,
            copied_matching_nodes,
            surround_circle_highlights,
        ) = self.odd_degree_step(
            left_graph,
            left_edges,
            right_graph,
            right_edges,
            tsp_edges_mob,
            mst_edges_mob,
            mst_edges,
            tsp_tour_edges,
        )

        (
            left_perfect_matching_mob,
            right_perfect_matching_mob,
            perfect_match,
        ) = self.min_weight_perfect_matching_step(
            left_graph,
            right_graph,
            vertices_to_match,
            copied_matching_nodes,
            surround_circle_highlights,
        )

        self.eulerian_tour_step(
            left_perfect_matching_mob,
            right_perfect_matching_mob,
            left_graph,
            left_edges,
            copied_matching_nodes,
            perfect_match,
            mst_edges,
        )

        self.summarize_christofides()

    def mst_step(self):
        mst_label = Text("MST", font=REDUCIBLE_FONT, weight=BOLD).scale(0.8)
        tsp_label = Text("Optimal TSP", font=REDUCIBLE_FONT, weight=BOLD).scale(0.8)

        (
            left_graph,
            mst_edges_mob,
            right_graph,
            tsp_edges_mob,
            left_edges,
            right_edges,
            mst_edges,
            tsp_tour_edges,
        ) = self.get_mst_and_tsp_tour(10)
        self.play(
            *[GrowFromCenter(left_graph.vertices[v]) for v in left_graph.vertices],
            *[GrowFromCenter(right_graph.vertices[v]) for v in right_graph.vertices],
            *[GrowFromCenter(edge) for edge in mst_edges_mob],
            *[GrowFromCenter(edge) for edge in tsp_edges_mob],
        )
        self.wait()

        mst_label.next_to(left_graph, UP).to_edge(UP * 2)
        tsp_label.next_to(right_graph, UP).to_edge(UP * 2)

        self.play(
            Write(mst_label),
            Write(tsp_label),
        )
        self.wait()
        long_right_arrow = MathTex(r"\Longrightarrow").move_to(
            (mst_label.get_center() + tsp_label.get_center()) / 2
        )
        long_right_arrow.scale(1.5)
        num_iterations = 5
        NUM_VERTICES = 10
        for i in range(0, num_iterations * 2, 2):
            (
                new_left_graph,
                new_mst_edges_mob,
                new_right_graph,
                new_tsp_edges_mob,
                left_edges,
                right_edges,
                mst_edges,
                tsp_tour_edges,
            ) = self.get_mst_and_tsp_tour(i, NUM_VERTICES=NUM_VERTICES)
            self.play(
                Transform(left_graph, new_left_graph),
                Transform(right_graph, new_right_graph),
                Transform(mst_edges_mob, new_mst_edges_mob),
                Transform(tsp_edges_mob, new_tsp_edges_mob),
            )
            self.wait()

            if i == 8:
                self.play(Write(long_right_arrow))

        (
            new_left_graph,
            new_mst_edges_mob,
            new_right_graph,
            new_tsp_edges_mob,
            left_edges,
            right_edges,
            mst_edges,
            tsp_tour_edges,
        ) = self.get_mst_and_tsp_tour(6, NUM_VERTICES=13)

        self.play(
            FadeTransform(left_graph, new_left_graph),
            FadeTransform(right_graph, new_right_graph),
            FadeTransform(mst_edges_mob, new_mst_edges_mob),
            FadeTransform(tsp_edges_mob, new_tsp_edges_mob),
        )

        self.play(FadeOut(mst_label), FadeOut(tsp_label), FadeOut(long_right_arrow))
        self.wait()

        return (
            new_left_graph,
            new_mst_edges_mob,
            new_right_graph,
            new_tsp_edges_mob,
            left_edges,
            right_edges,
            mst_edges,
            tsp_tour_edges,
        )

    def get_node_with_opacity(self, node, opacity=0.2):
        fill_opacity = opacity
        if opacity == 1:
            fill_opacity = 0.5
        node_copy = node.copy()
        node_copy[0].set_fill(opacity=fill_opacity).set_stroke(opacity=opacity)
        node_copy[1].set_fill(opacity=opacity)
        return node_copy

    def odd_degree_step(
        self,
        left_graph,
        left_edges,
        right_graph,
        right_edges,
        tsp_edges_mob,
        mst_edges_mob,
        mst_edges,
        tsp_tour_edges,
    ):
        tsp_tour_edge_to_index = {edge: i for i, edge in enumerate(tsp_tour_edges)}
        for v in right_graph.vertices:
            neighboring_edges = self.get_neighboring_edges(v, tsp_tour_edges)
            right_node_highlighted = self.get_node_with_opacity(
                right_graph.vertices[v], opacity=1
            )
            other_nodes_indices = [i for i in right_graph.vertices if i != v]
            self.play(
                Transform(right_graph.vertices[v], right_node_highlighted),
                *[
                    tsp_edges_mob[tsp_tour_edge_to_index[edge]].animate.set_stroke(
                        opacity=1
                    )
                    for edge in neighboring_edges
                ],
                *[
                    tsp_edges_mob[tsp_tour_edge_to_index[edge]].animate.set_stroke(
                        opacity=0.2
                    )
                    for edge in tsp_tour_edges
                    if edge not in neighboring_edges
                ],
                *[
                    Transform(
                        right_graph.vertices[i],
                        self.get_node_with_opacity(
                            right_graph.vertices[i], opacity=0.2
                        ),
                    )
                    for i in other_nodes_indices
                ],
            )
            self.wait()

        self.play(
            *[
                tsp_edges_mob[tsp_tour_edge_to_index[edge]].animate.set_stroke(
                    opacity=1
                )
                for edge in tsp_tour_edges
            ],
            *[
                Transform(
                    right_graph.vertices[i],
                    self.get_node_with_opacity(right_graph.vertices[i], opacity=1),
                )
                for i in other_nodes_indices
            ],
        )
        self.wait()

        degree_map = get_degrees_for_all_vertices(
            mst_edges, left_graph.get_dist_matrix()
        )
        vertices_to_match = [v for v in degree_map if degree_map[v] % 2 == 1]
        surround_circle_highlights = [
            get_glowing_surround_circle(left_graph.vertices[v], color=REDUCIBLE_CHARM)
            for v in vertices_to_match
        ]

        self.play(*[FadeIn(circle) for circle in surround_circle_highlights])
        self.wait()
        copied_matching_nodes = [
            right_graph.vertices[v].copy() for v in vertices_to_match
        ]
        self.play(
            FadeOut(right_graph),
            FadeOut(tsp_edges_mob),
            *[
                TransformFromCopy(left_graph.vertices[v], copied_matching_nodes[i])
                for i, v in enumerate(vertices_to_match)
            ],
        )
        self.wait()
        return vertices_to_match, copied_matching_nodes, surround_circle_highlights

    def min_weight_perfect_matching_step(
        self,
        left_graph,
        right_graph,
        vertices_to_match,
        copied_matching_nodes,
        surround_circle_highlights,
    ):
        all_perfect_matches = get_all_perfect_matchings(vertices_to_match)
        right_node_map = {
            v: copied_matching_nodes[i] for i, v in enumerate(vertices_to_match)
        }
        left_perfect_matching_mob = self.get_perfect_matching_edges(
            all_perfect_matches[0], left_graph.vertices
        )
        right_perfect_matching_mob = self.get_perfect_matching_edges(
            all_perfect_matches[0], right_node_map
        )
        self.play(
            FadeIn(left_perfect_matching_mob),
            FadeIn(right_perfect_matching_mob),
        )
        self.wait()

        minimum_weight_label = Text("Minimum Weight Cost: ", font=REDUCIBLE_MONO).scale(
            0.4
        )
        best_cost = get_cost_from_edges(
            all_perfect_matches[0], left_graph.get_dist_matrix()
        )
        minimum_weight_text = (
            Text(str(np.round(best_cost, 2)), font=REDUCIBLE_MONO)
            .scale(0.4)
            .next_to(minimum_weight_label, RIGHT)
        )
        minimum_weight_label_and_cost = VGroup(
            minimum_weight_label, minimum_weight_text
        )
        minimum_weight_label_and_cost.next_to(left_graph, DOWN).to_edge(DOWN * 2)

        current_cost_label = Text("Perfect Matching Cost: ", font=REDUCIBLE_MONO).scale(
            0.4
        )
        current_cost = get_cost_from_edges(
            all_perfect_matches[0], left_graph.get_dist_matrix()
        )
        current_cost_text = (
            Text(str(np.round(current_cost, 2)), font=REDUCIBLE_MONO)
            .scale(0.4)
            .next_to(current_cost_label, RIGHT)
        )
        current_cost_label_and_cost = VGroup(current_cost_label, current_cost_text)
        current_cost_label_and_cost.next_to(right_graph, DOWN).to_edge(DOWN * 2)
        self.play(
            Write(minimum_weight_label_and_cost),
            Write(current_cost_label_and_cost),
        )
        self.wait()

        run_time = 1
        best_index = None
        import time

        for i, perfect_matching in enumerate(all_perfect_matches):
            start_time = time.time()
            if i == 0:
                continue
            new_left_perfect_matching_mob = self.get_perfect_matching_edges(
                perfect_matching, left_graph.vertices
            )
            new_right_perfect_matching_mob = self.get_perfect_matching_edges(
                perfect_matching, right_node_map
            )
            current_cost = get_cost_from_edges(
                perfect_matching, left_graph.get_dist_matrix()
            )
            new_current_cost_text = (
                Text(str(np.round(current_cost, 2)), font=REDUCIBLE_MONO)
                .scale(0.4)
                .next_to(current_cost_label, RIGHT)
            )
            start_time = time.time()
            if current_cost < best_cost:
                best_cost = current_cost
                best_index = i
                new_min_weight_text = (
                    Text(str(np.round(best_cost, 2)), font=REDUCIBLE_MONO)
                    .scale(0.4)
                    .next_to(minimum_weight_label, RIGHT)
                )
                self.play(
                    Transform(left_perfect_matching_mob, new_left_perfect_matching_mob),
                    Transform(
                        right_perfect_matching_mob, new_right_perfect_matching_mob
                    ),
                    Transform(current_cost_text, new_current_cost_text),
                    Transform(minimum_weight_text, new_min_weight_text),
                    run_time=run_time,
                )
            else:
                self.play(
                    Transform(
                        right_perfect_matching_mob, new_right_perfect_matching_mob
                    ),
                    Transform(current_cost_text, new_current_cost_text),
                    run_time=run_time,
                )

            if i < 5:
                self.wait()
            # self.wait(wait_time)
            run_time = run_time * 0.9

        new_right_perfect_matching_mob = self.get_perfect_matching_edges(
            all_perfect_matches[best_index], right_node_map
        )
        self.wait()
        self.play(
            Transform(right_perfect_matching_mob, new_right_perfect_matching_mob),
            FadeOut(current_cost_label_and_cost),
        )
        self.wait()

        self.play(
            *[FadeOut(highlight) for highlight in surround_circle_highlights],
            FadeOut(minimum_weight_label_and_cost),
        )
        self.wait()

        return (
            left_perfect_matching_mob,
            right_perfect_matching_mob,
            all_perfect_matches[best_index],
        )

    def get_perfect_matching_edges(self, perfect_matching, node_map):
        perfect_matching_edges = VGroup()
        for u, v in perfect_matching:
            edge_mob = self.make_edge(node_map[u], node_map[v])
            perfect_matching_edges.add(edge_mob)
        return perfect_matching_edges

    def make_edge(self, node1, node2, stroke_width=3, color=REDUCIBLE_GREEN_LIGHTER):
        return Line(
            node1.get_center(),
            node2.get_center(),
            buff=node1.width / 2,
            stroke_width=stroke_width,
        ).set_color(color)

    def eulerian_tour_step(
        self,
        left_perfect_matching_mob,
        right_perfect_matching_mob,
        left_graph,
        left_edges,
        copied_matching_nodes,
        perfect_matching,
        mst_edges,
    ):
        mst_edges_mob = VGroup(*[self.get_edge(left_edges, edge) for edge in mst_edges])
        multigraph = VGroup(left_graph, mst_edges_mob, left_perfect_matching_mob)

        multigraph_title = (
            Text(
                "MST and Minimum Weight Perfect Matching MultiGraph",
                font=REDUCIBLE_FONT,
                weight=BOLD,
            )
            .scale(0.6)
            .move_to(UP * 3.2)
        )
        self.play(
            *[FadeOut(node) for node in copied_matching_nodes],
            FadeOut(right_perfect_matching_mob),
            multigraph.animate.scale(1.2).move_to(ORIGIN),
            FadeIn(multigraph_title),
        )
        self.wait()

        self.play(
            *[
                self.get_edge(left_edges, edge).animate.set_stroke(width=6)
                for edge in mst_edges
            ]
        )
        self.wait()

        duplicate_edge_t_1 = Text("Duplicate Edge", font=REDUCIBLE_FONT).scale(0.4)
        duplicate_edge_t_2 = duplicate_edge_t_1.copy()

        left_arrow = Arrow(
            ORIGIN, UL * 1.2, max_tip_length_to_length_ratio=0.15
        ).set_color(REDUCIBLE_VIOLET)
        right_arrow = Arrow(
            ORIGIN, RIGHT * 1.5, max_tip_length_to_length_ratio=0.15
        ).set_color(REDUCIBLE_VIOLET)

        duplicate_edge_t_1.next_to(left_arrow, DR)
        right_arrow_group = (
            VGroup(duplicate_edge_t_1, right_arrow)
            .arrange(RIGHT)
            .next_to(self.get_edge(left_edges, (3, 6)), LEFT)
        )
        left_arrow_group = VGroup(left_arrow, duplicate_edge_t_2).next_to(
            self.get_edge(left_edges, (9, 11)), DR, buff=SMALL_BUFF / 2
        )

        self.play(
            FadeIn(right_arrow_group, direction=LEFT),
            FadeIn(left_arrow_group, direction=RIGHT),
        )
        self.wait()

        self.play(
            FadeOut(right_arrow_group, direction=LEFT),
            FadeOut(left_arrow_group, direction=RIGHT),
        )
        self.wait()

        key_observation = (
            Text("All vertices have even degree", font=REDUCIBLE_FONT)
            .scale(0.5)
            .move_to(DOWN * 3.2)
        )

        self.play(FadeIn(key_observation))
        self.wait()

        find_eulerian_text = (
            Text("Find Eulerian Tour of Multigraph", font=REDUCIBLE_FONT, weight=BOLD)
            .scale(0.6)
            .move_to(multigraph_title.get_center())
        )

        self.play(
            FadeTransform(multigraph_title, find_eulerian_text),
            FadeOut(key_observation),
        )
        self.wait()

        eulerian_tour = get_eulerian_tour(mst_edges, perfect_matching, start=0)
        ordered_vertices_tex = self.get_tex_eulerian_tour(eulerian_tour).move_to(
            DOWN * 2.9
        )
        self.play(FadeIn(ordered_vertices_tex))
        self.wait()

        eulerian_tour_edge_map = self.get_eulerian_tour_edges_map(
            left_graph, eulerian_tour
        )

        for edge in eulerian_tour_edge_map:
            self.play(
                Create(eulerian_tour_edge_map[edge]),
                Flash(eulerian_tour_edge_map[edge].copy()),
            )

        self.wait()

        generate_tsp_tour_step = Text(
            "Generate TSP Tour from Eulerian Tour", font=REDUCIBLE_FONT, weight=BOLD
        ).scale(0.6)
        generate_tsp_tour_step.move_to(find_eulerian_text.get_center())

        self.play(
            *[FadeOut(e) for e in eulerian_tour_edge_map.values()],
            *[FadeOut(self.get_edge(left_edges, edge)) for edge in mst_edges],
            FadeOut(left_perfect_matching_mob),
            FadeTransform(find_eulerian_text, generate_tsp_tour_step),
        )
        self.wait()

        self.hamiltonian_tour_step(
            left_graph, eulerian_tour, ordered_vertices_tex, generate_tsp_tour_step
        )

    def get_ordered_vertices(self, eulerian_tour):
        ordered_vertices = []
        for u, v in eulerian_tour:
            ordered_vertices.append(u)
        ordered_vertices.append(v)
        return ordered_vertices

    def get_tex_eulerian_tour(self, eulerian_tour, scale=0.3):
        ordered_vertices = self.get_ordered_vertices(eulerian_tour)
        string = r" -> ".join([str(v) for v in ordered_vertices])
        return Text(string, font=REDUCIBLE_MONO).scale(scale)

    def hamiltonian_tour_step(
        self, left_graph, eulerian_tour, ordered_vertices_tex, generate_tsp_tour_step
    ):
        tsp_tour = get_hamiltonian_tour_from_eulerian(eulerian_tour)
        tsp_tour_edges = get_edges_from_tour(tsp_tour)
        ordered_vertices = self.get_ordered_vertices(eulerian_tour)

        tsp_tour_edges_mob = self.get_tsp_tour_edges_mob(left_graph, tsp_tour_edges)
        index = 0
        visited = set()

        v_to_tex_end_index_map = {}
        current_index = 1
        for i, v in enumerate(ordered_vertices):
            # 1 digit
            if v // 10 == 0:
                v_to_tex_end_index_map[i] = current_index
            else:
                current_index += 1
                v_to_tex_end_index_map[i] = current_index

            current_index += 3
        tsp_tour_edges_index = 0
        glowing_circles = VGroup()
        visited = set()
        crosses = []
        for i, v in enumerate(ordered_vertices):
            glowing_circle = get_glowing_surround_circle(left_graph.vertices[v])
            if i == 0:
                self.play(
                    ordered_vertices_tex[: v_to_tex_end_index_map[i]].animate.set_fill(
                        opacity=1
                    ),
                    ordered_vertices_tex[v_to_tex_end_index_map[i] :].animate.set_fill(
                        opacity=0.5
                    ),
                    FadeIn(glowing_circle),
                )
                glowing_circles.add(glowing_circle)

            elif v in visited and v != ordered_vertices[0]:
                end_index = v_to_tex_end_index_map[i]
                start_index = end_index - 1 - v // 10
                cross = Cross(ordered_vertices_tex[start_index:end_index]).set_color(
                    REDUCIBLE_CHARM
                )
                cross.set_stroke(width=2)
                self.play(
                    ordered_vertices_tex[: v_to_tex_end_index_map[i]].animate.set_fill(
                        opacity=1
                    ),
                    ordered_vertices_tex[v_to_tex_end_index_map[i] :].animate.set_fill(
                        opacity=0.5
                    ),
                )
                self.play(
                    Write(cross),
                )
                self.add_foreground_mobject(cross)
                crosses.append(cross)
            else:
                if v != ordered_vertices[0]:
                    self.play(
                        ordered_vertices_tex[
                            : v_to_tex_end_index_map[i]
                        ].animate.set_fill(opacity=1),
                        ordered_vertices_tex[
                            v_to_tex_end_index_map[i] :
                        ].animate.set_fill(opacity=0.5),
                        Write(tsp_tour_edges_mob[tsp_tour_edges_index]),
                        FadeIn(glowing_circle),
                    )
                    glowing_circles.add(glowing_circle)

                else:
                    self.play(
                        ordered_vertices_tex[
                            : v_to_tex_end_index_map[i]
                        ].animate.set_fill(opacity=1),
                        ordered_vertices_tex[
                            v_to_tex_end_index_map[i] :
                        ].animate.set_fill(opacity=0.5),
                        Write(tsp_tour_edges_mob[tsp_tour_edges_index]),
                    )
                tsp_tour_edges_index += 1
                visited.add(v)

            self.wait()

        tex_tsp_tour = self.get_tex_eulerian_tour(tsp_tour_edges, scale=0.32)
        tex_tsp_tour_label = Text(
            "TSP Tour: ", font=REDUCIBLE_MONO
        ).scale_to_fit_height(tex_tsp_tour.height)
        text_tsp_tour_with_label = VGroup(tex_tsp_tour_label, tex_tsp_tour).arrange(
            RIGHT
        )
        text_tsp_tour_with_label.next_to(ordered_vertices_tex, DOWN)

        self.play(
            FadeOut(glowing_circles),
            FadeIn(text_tsp_tour_with_label),
        )
        self.wait()

        self.play(
            FadeOut(ordered_vertices_tex),
            FadeOut(text_tsp_tour_with_label),
            FadeOut(generate_tsp_tour_step),
            *[FadeOut(c) for c in crosses],
        )
        graph_with_tsp_tour = VGroup(left_graph, tsp_tour_edges_mob)
        christofides_cost = get_cost_from_edges(
            tsp_tour_edges, left_graph.get_dist_matrix()
        )
        self.play(graph_with_tsp_tour.animate.scale(0.8).move_to(LEFT * 3.5 + UP * 0.5))
        right_graph = left_graph.copy().move_to(RIGHT * 3.5 + UP * 0.5)
        optimal_tsp_tour, optimal_cost = get_exact_tsp_solution(
            left_graph.get_dist_matrix()
        )

        right_graph_tsp_edges_mob = self.get_tsp_tour_edges_mob(
            right_graph, get_edges_from_tour(optimal_tsp_tour)
        )

        self.play(
            FadeIn(right_graph),
            FadeIn(right_graph_tsp_edges_mob),
        )
        self.wait()
        christofides_cost_text = Text(
            f"Christofides tour cost: {np.round(christofides_cost, 2)}",
            font=REDUCIBLE_MONO,
        ).scale(0.5)
        christofides_cost_text.next_to(left_graph, DOWN).to_edge(DOWN * 3)

        optimal_cost_text = Text(
            f"Optimal tour cost: {np.round(optimal_cost, 2)}", font=REDUCIBLE_MONO
        ).scale(0.5)
        optimal_cost_text.next_to(right_graph, DOWN).to_edge(DOWN * 3)

        self.play(
            FadeIn(christofides_cost_text),
            FadeIn(optimal_cost_text),
        )
        self.wait()

        self.clear()

    def summarize_christofides(self):

        christofides_alg = (
            Text("Christofides Algorithm", font=REDUCIBLE_FONT, weight=BOLD)
            .scale(0.8)
            .move_to(UP * 3.3)
        )
        screen_rect = ScreenRectangle(height=3).move_to(LEFT * 3 + UP * 0.5)

        step_1 = Tex(r"1. Find MST $T$ of Graph")
        step_2 = Tex(r"2. Isolate Set of Odd-Degree Vertices $S$")
        step_3 = Tex(r"3. Find Min Weight Perfect Matching $M$ of $S$")
        step_4 = Tex(r"4. Combine $T$ and $M$ into Multigraph $G$")
        step_5 = Tex(r"5. Generate Eulerian Tour of $G$")
        step_6 = Tex(r"6. Generate TSP Tour from Eulerian Tour")

        steps = (
            VGroup(*[step_1, step_2, step_3, step_4, step_5, step_6])
            .scale(0.6)
            .arrange(DOWN, aligned_edge=LEFT)
        )
        steps.move_to(RIGHT * 3.5 + UP * 0.5)

        self.play(Write(christofides_alg))

        on_average_perf = average_case = (
            Tex(
                r"On average: $\frac{\text{Christofides}}{\text{1-Tree Lower Bound}} = 1.1$"
            )
            .scale(0.8)
            .move_to(DOWN * 3)
        )
        for i, step in enumerate(steps):
            if i == 0:
                self.play(FadeIn(screen_rect), FadeIn(step))
            else:
                self.play(FadeIn(step))
            self.wait()

        self.play(FadeIn(on_average_perf))
        self.wait()

        worst_case_perf = (
            Tex(r"Worst case: $\frac{\text{Christofides}}{\text{Optimal TSP}} = 1.5$")
            .scale(0.8)
            .move_to(DOWN * 3 + RIGHT * 3.5)
        )
        self.play(on_average_perf.animate.shift(LEFT * 3.5), Write(worst_case_perf))
        self.wait()

    def get_eulerian_tour_edges_map(self, left_graph, eulerian_tour):
        all_edges = {}
        for edge in eulerian_tour:
            u, v = edge
            if (v, u) in all_edges:
                all_edges[(u, v)] = self.make_edge(
                    left_graph.vertices[u],
                    left_graph.vertices[v],
                    stroke_width=7,
                    color=REDUCIBLE_PURPLE,
                )
            else:
                all_edges[edge] = self.make_edge(
                    left_graph.vertices[u],
                    left_graph.vertices[v],
                    stroke_width=7,
                    color=REDUCIBLE_VIOLET,
                )

        return all_edges

    def get_tsp_tour_edges_mob(
        self, graph, tsp_tour_edges, stroke_width=3, color=REDUCIBLE_VIOLET
    ):
        return VGroup(
            *[
                self.make_edge(
                    graph.vertices[u],
                    graph.vertices[v],
                    stroke_width=stroke_width,
                    color=REDUCIBLE_VIOLET,
                )
                for u, v in tsp_tour_edges
            ]
        )

    def get_mst_and_tsp_tour(self, seed, NUM_VERTICES=10):
        np.random.seed(seed)
        layout = self.get_random_layout(NUM_VERTICES)
        left_graph = TSPGraph(list(range(NUM_VERTICES)), layout=layout).scale(0.55)
        right_graph = left_graph.copy()

        VGroup(left_graph, right_graph).arrange(RIGHT, buff=1.2)

        left_edges = left_graph.get_all_edges(buff=left_graph.vertices[0].width / 2)
        right_edges = right_graph.get_all_edges(buff=right_graph.vertices[0].width / 2)
        mst_edges, mst_cost = get_mst(left_graph.get_dist_matrix())
        tsp_tour, optimal_cost = get_exact_tsp_solution(right_graph.get_dist_matrix())
        tsp_tour_edges = get_edges_from_tour(tsp_tour)
        mst_edges_mob = VGroup(
            *[
                self.get_edge(left_edges, edge).set_color(REDUCIBLE_YELLOW)
                for edge in mst_edges
            ]
        )
        tsp_edges_mob = VGroup(
            *[self.get_edge(right_edges, edge) for edge in tsp_tour_edges]
        )

        for i, edge in enumerate(tsp_tour_edges):
            if edge in mst_edges or (edge[1], edge[0]) in mst_edges:
                tsp_edges_mob[i].set_color(REDUCIBLE_YELLOW)

        return (
            left_graph,
            mst_edges_mob,
            right_graph,
            tsp_edges_mob,
            left_edges,
            right_edges,
            mst_edges,
            tsp_tour_edges,
        )

    def get_neighboring_edges(self, vertex, edges):
        return [edge for edge in edges if vertex in edge]
