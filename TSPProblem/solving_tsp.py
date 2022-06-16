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
        label_scale=0.6,
        **kwargs,
    ):
        edges = []
        if labels:
            labels = {k: CustomLabel(str(k), scale=label_scale) for k in vertices}
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

    def get_all_edges(self, edge_type: TipableVMobject = Line):
        edge_dict = {}
        for edge in itertools.combinations(self.vertices.keys(), 2):
            u, v = edge
            edge_dict[edge] = self.create_edge(u, v, edge_type=edge_type)
        return edge_dict

    def create_edge(self, u, v, edge_type: TipableVMobject = Line):
        return edge_type(
            self.vertices[u].get_center(),
            self.vertices[v].get_center(),
            color=self.edge_config["color"],
            stroke_width=self.edge_config["stroke_width"],
            buff=self.edge_config["buff"] if buff is None else buff,
        )

    def get_tour_edges(self, tour, edge_type: TipableVMobject = Line):
        """
        @param: tour -- sequence of vertices where all vertices are part of the tour (no repetitions)
        """
        edges = get_edges_from_tour(tour)
        edge_dict = {}
        for edge in edges:
            u, v = edge
            edge_mob = self.create_edge(u, v, edge_type=edge_type)
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

class CustomLabel(Text):
    def __init__(self, label, font="SF Mono", scale=1, weight=BOLD):
        super().__init__(label, font=font, weight=weight)
        self.scale(scale)


class TSPTester(Scene):
    def construct(self):
        big_graph = TSPGraph(range(12), layout_scale=2.4, layout="circular")
        all_edges_bg = big_graph.get_all_edges()
        self.play(
            FadeIn(big_graph)
        )
        self.wait()

        self.play(
            *[FadeIn(edge) for edge in all_edges_bg.values()]
        )
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
                r"On average: $\frac{\text{NN Heuristic}}{\text{Held-Karp Lower Bound}} = 1.25$"
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
            LaggedStartMap(GrowFromCenter, list(graph.vertices.values())),
            run_time=2
        )
        self.wait()

        self.play(
            LaggedStartMap(Write, [tour_edges[edge] for edge in edge_ordering]),
            run_time=10
        )
        self.wait()

        problem = Text("Given any solution, no efficient way to verify optimality!", font=REDUCIBLE_FONT).scale(0.5).move_to(DOWN * 3.5)

        self.play(
            FadeIn(problem)
        )
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
        VGroup(heuristic_solution_mod, left_geq, optimal_solution_mod).arrange(RIGHT, buff=1)

        self.play(
            FadeIn(heuristic_solution_mod),
            FadeIn(optimal_solution_mod),
            FadeIn(left_geq)
        )
        self.wait()

        right_geq = MathTex(r"\geq").scale(2)
        new_configuration = VGroup(heuristic_solution_mod.copy(), left_geq.copy(), optimal_solution_mod.copy(), right_geq, lower_bound_mod).arrange(RIGHT, buff=1).scale(0.7)

        self.play(
            Transform(heuristic_solution_mod, new_configuration[0]),
            Transform(left_geq, new_configuration[1]),
            Transform(optimal_solution_mod, new_configuration[2]),
        )

        self.play(
            FadeIn(right_geq),
            FadeIn(lower_bound_mod)
        )
        self.wait()

        curved_arrow_1 = CustomCurvedArrow(
            heuristic_solution_mod.get_top(),
            optimal_solution_mod.get_top(),
            angle=-PI/4
        ).shift(UP * MED_SMALL_BUFF).set_color(GRAY)

        curved_arrow_2 = CustomCurvedArrow(
            heuristic_solution_mod.get_bottom(),
            lower_bound_mod.get_bottom(),
            angle=PI/4,
        ).shift(DOWN * MED_SMALL_BUFF).set_color(GRAY)

        inefficient_comparison = Text("Intractable comparison", font=REDUCIBLE_FONT).scale(0.6).next_to(curved_arrow_1, UP)
        reasonable_comparison = Text("Reasonable comparison", font=REDUCIBLE_FONT).scale(0.6).next_to(curved_arrow_2, DOWN)
        self.play(
            Write(curved_arrow_1),
            FadeIn(inefficient_comparison)
        )
        self.wait()

        self.play(
            Write(curved_arrow_2),
            FadeIn(reasonable_comparison)
        )
        self.wait()

        good_lower_bound = Tex(r"Good lower bound: maximize $\frac{\text{lower bound}}{\text{optimal}}$").scale(0.8).move_to(UP * 3)

        self.play(
            FadeIn(good_lower_bound)
        )
        self.wait()

    def intro_mst(self):
        title = Text("Minimum Spanning Tree (MST)", font=REDUCIBLE_FONT, weight=BOLD).scale(0.8).move_to(UP * 3.5)
        NUM_VERTICES = 9
        graph = TSPGraph(
            list(range(NUM_VERTICES)),
            layout=self.get_random_layout(NUM_VERTICES)
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
        self.play(
            Write(title),
            FadeIn(graph)
        )
        self.wait()
        self.play(
            *[GrowFromCenter(edge) for edge in mst_edges_group]
        )
        self.wait()

        definition = Text("Set of edges that connect all vertices with minimum distance and no cycles", font=REDUCIBLE_FONT).scale(0.5).move_to(DOWN * 3.5)
        definition[14:21].set_color(REDUCIBLE_YELLOW)
        definition[36:51].set_color(REDUCIBLE_YELLOW)
        definition[-8:].set_color(REDUCIBLE_YELLOW)
        self.play(
            FadeIn(definition)
        )
        self.wait()

        true_mst_text = Text("True MST", font=REDUCIBLE_FONT).scale(0.7)
        self.play(
            mst_tree.animate.scale(0.75).shift(LEFT * 3.5)
        )
        true_mst_text.next_to(mst_tree, DOWN)
        self.play(
            FadeIn(true_mst_text)
        )
        self.wait()

        to_remove_edge = (8, 6)
        mst_edges.remove(to_remove_edge)
        not_connected_edge = not_connected_graph.get_edges_from_list(mst_edges)
        not_connect_graph_group = VGroup(*[not_connected_graph] + list(not_connected_edge.values())).scale(0.6).shift(RIGHT * 3.5)

        not_connected_text = Text("Not connected", font=REDUCIBLE_FONT).scale(0.7).next_to(not_connect_graph_group, DOWN)
        self.play(
            FadeIn(not_connect_graph_group),
            FadeIn(not_connected_text)
        )

        surround_rect = SurroundingRectangle(VGroup(not_connected_graph.vertices[8], not_connected_graph.vertices[6]), color=REDUCIBLE_CHARM)
        self.play(
            Write(surround_rect),
        )
        self.wait()

        to_add_edge = (6, 2)
        prev_removed_edge = not_connected_graph.create_edge(to_remove_edge[0], to_remove_edge[1], buff=graph.vertices[0].width / 2 + SMALL_BUFF / 4)
        new_edge = not_connected_graph.create_edge(to_add_edge[0], to_add_edge[1], buff=graph.vertices[0].width / 2 + SMALL_BUFF / 4)

        cyclic_text = Text("Has cycle", font=REDUCIBLE_FONT).scale(0.7).move_to(not_connected_text.get_center())
        self.play(
            FadeOut(surround_rect),
            Write(prev_removed_edge),
            Write(new_edge),
            ReplacementTransform(not_connected_text, cyclic_text)
        )
        new_surround_rect = SurroundingRectangle(
            VGroup(
                not_connected_graph.vertices[8],
                not_connected_graph.vertices[6],
                not_connected_graph.vertices[2],
                not_connected_graph.vertices[0],
            ),
            color=REDUCIBLE_CHARM
        )
        self.play(
            Write(new_surround_rect)
        )
        self.wait()

        non_optimal_edge = not_connected_graph.create_edge(5, 7, buff=graph.vertices[0].width / 2 + SMALL_BUFF / 4)
        non_optimal_edge.set_color(REDUCIBLE_CHARM)
        non_optimal_text = Text("Spanning tree, but not minimum", font=REDUCIBLE_FONT).scale(0.6).move_to(cyclic_text.get_center())
        self.play(
            FadeOut(new_surround_rect),
            FadeOut(new_edge),
            FadeOut(not_connected_edge[(5, 1)]),
            Write(non_optimal_edge),
            ReplacementTransform(cyclic_text, non_optimal_text)
        )
        self.wait()

        self.clear()

        mst_tree, mst_edge_dict = self.demo_prims_algorithm(original_scaled_graph.copy())

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

        visited_group, unvisited_group, visited_dict, unvisited_dict = self.highlight_visited_univisited(graph.vertices, graph.labels, visited, unvisited)
        visited_label = Text("Visited", font=REDUCIBLE_FONT).scale(0.5).next_to(visited_group, UP)
        unvisited_label = Text("Unvisited", font=REDUCIBLE_FONT).scale(0.5).next_to(unvisited_group, UP)
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
                highlight_animations.append(Transform(graph.vertices[v], un_highlighted_v))
        self.play(
            *highlight_animations
        )
        self.wait()
        mst_edges = VGroup()
        mst_edge_dict = {}
        while len(unvisited) > 0:
            neighboring_edges = self.get_neighboring_edges_across_sets(visited, unvisited)
            for i, edge in enumerate(neighboring_edges):
                if edge not in all_edges:
                    neighboring_edges[i] = (edge[1], edge[0])
            neighboring_edges_mobs = [all_edges[edge].set_stroke(opacity=0.3) for edge in neighboring_edges]
            self.play(
                *[Write(edge) for edge in neighboring_edges_mobs]
            )
            self.wait()
            best_neighbor_edge = min(neighboring_edges, key=lambda x: graph.get_dist_matrix()[x[0]][x[1]])
            next_vertex = best_neighbor_edge[1] if best_neighbor_edge[1] not in visited else best_neighbor_edge[0]
            print('Best neighbor', best_neighbor_edge)
            print('Next vertex', next_vertex)
            self.play(
                ShowPassingFlash(
                    all_edges[best_neighbor_edge].copy().set_stroke(width=6).set_color(REDUCIBLE_YELLOW), time_width=0.5
                ),
            )
            self.play(
                all_edges[best_neighbor_edge].animate.set_stroke(opacity=1, color=REDUCIBLE_YELLOW)
            )
            mst_edges.add(all_edges[best_neighbor_edge])
            mst_edge_dict[best_neighbor_edge] = all_edges[best_neighbor_edge]
            self.wait()

            visited.add(next_vertex)
            unvisited.remove(next_vertex)

            _, _, new_visited_dict, new_unvisited_dict = self.highlight_visited_univisited(graph.vertices, graph.labels, visited, unvisited)
            print(type(graph.vertices[next_vertex][1]))
            highlight_next_vertex = graph.vertices[next_vertex].copy()
            highlight_next_vertex[0].set_fill(opacity=0.5).set_stroke(opacity=1)
            highlight_next_vertex[1].set_fill(opacity=1)
            self.play(
                FadeOut(*[all_edges[edge] for edge in neighboring_edges if edge != best_neighbor_edge]),
                Transform(graph.vertices[next_vertex], highlight_next_vertex),
                *[Transform(visited_dict[v], new_visited_dict[v]) for v in visited.difference(set([next_vertex]))],
                *[Transform(unvisited_dict[v], new_unvisited_dict[v]) for v in unvisited],
                ReplacementTransform(unvisited_dict[next_vertex], new_visited_dict[next_vertex]),
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
        self.play(
            mst_tree.animate.scale(0.75).move_to(LEFT * 3.5 + UP * 1)
        )
        self.wait()
        tsp_graph_with_tour.scale_to_fit_height(mst_tree.height).move_to(RIGHT * 3.5 + UP * 1)


        self.play(
            FadeIn(tsp_graph_with_tour)
        )
        self.wait()

        mst_cost = Tex(r"MST Cost $<$ TSP Cost").move_to(DOWN * 2)
        self.play(
            FadeIn(mst_cost)
        )
        self.wait()

        remove_edge = Tex(r"Remove any edge from TSP tour $\rightarrow$ spanning tree $T$").scale(0.7).next_to(mst_cost, DOWN)
        self.play(
            FadeIn(remove_edge)
        )
        self.wait()
        result = Tex(r"MST cost $\leq$ cost($T$)").scale(0.7)
        result.next_to(remove_edge, DOWN)
        prev_edge = None
        for i, edge in enumerate(tsp_tour_edges):
            if i == 0:
                self.play(
                    FadeOut(tsp_tour_edges[edge])
                )
            else:
                self.play(
                    FadeIn(tsp_tour_edges[prev_edge]),
                    FadeOut(tsp_tour_edges[edge])
                )
            prev_edge = edge
            self.wait()

        self.play(
            FadeIn(result)
        )
        self.wait()

        better_lower_bound = Text("Better Lower Bound", font=REDUCIBLE_FONT, weight=BOLD).scale_to_fit_height(mst_cost.height - SMALL_BUFF).move_to(mst_cost.get_center()).shift(UP * SMALL_BUFF)
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
        self.play(
            FadeIn(step_1)
        )
        self.wait()

        self.play(
            FadeOut(mst_vertices.vertices[6])
        )
        self.wait()

        mst_tree_edges_removed, cost, one_tree_edges, one_tree_cost  = get_1_tree(mst_vertices.get_dist_matrix(), 6)
        all_edges = mst_vertices.get_all_edges(buff=mst_vertices[0].width / 2)
        self.play(
            *[GrowFromCenter(self.get_edge(all_edges, edge).set_color(REDUCIBLE_YELLOW)) for edge in mst_tree_edges_removed]
        )
        self.wait()

        self.play(
            FadeIn(step_2)
        )
        self.wait()

        self.play(
            FadeIn(mst_vertices.vertices[6])
        )
        self.wait()

        self.play(
            *[GrowFromCenter(self.get_edge(all_edges, edge).set_color(REDUCIBLE_YELLOW)) for edge in one_tree_edges if edge not in mst_tree_edges_removed]
        )
        self.wait()

        new_result = Tex(r"1-tree cost $\leq$ TSP cost").scale(0.7)
        new_result.next_to(steps, DOWN)
        new_result[0][:6].set_color(REDUCIBLE_YELLOW)
        self.play(
            FadeIn(new_result)
        )
        self.wait()

        unhiglighted_nodes = {v: tsp_graph.vertices[v].copy() for v in tsp_graph.vertices if v != 6}
        highlighted_nodes = copy.deepcopy(unhiglighted_nodes)
        for node in unhiglighted_nodes.values():
            node[0].set_fill(opacity=0.2).set_stroke(opacity=0.2)
            node[1].set_fill(opacity=0.2)

        unhiglighted_nodes_mst = {v: mst_vertices.vertices[v].copy() for v in mst_vertices.vertices if v != 6}
        highlighted_nodes_mst = copy.deepcopy(unhiglighted_nodes_mst)
        for node in unhiglighted_nodes_mst.values():
            node[0].set_fill(opacity=0.2).set_stroke(opacity=0.2)
            node[1].set_fill(opacity=0.2)

        self.play(
            *[Transform(tsp_graph.vertices[v], unhiglighted_nodes[v]) for v in tsp_graph.vertices if v != 6],
            *[tsp_tour_edges[edge].animate.set_stroke(opacity=0.2) for edge in tsp_tour_edges if 6 not in edge],
            *[Transform(mst_vertices.vertices[v], unhiglighted_nodes_mst[v]) for v in mst_vertices.vertices if v != 6],
            *[self.get_edge(all_edges, edge).animate.set_stroke(opacity=0.2) for edge in one_tree_edges if 6 not in edge],
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
            *[Transform(tsp_graph.vertices[v], highlighted_nodes[v]) for v in tsp_graph.vertices if v != 6],
            *[tsp_tour_edges[edge].animate.set_stroke(opacity=1) for edge in tsp_tour_edges if 6 not in edge],
            *[Transform(mst_vertices.vertices[v], highlighted_nodes_mst[v]) for v in mst_vertices.vertices if v != 6],
            *[self.get_edge(all_edges, edge).animate.set_stroke(opacity=1) for edge in one_tree_edges if 6 not in edge],
            Transform(mst_vertices.vertices[6], node_6_faded),
            Transform(tsp_graph.vertices[6], node_6_faded_tsp),
            *[tsp_tour_edges[edge].animate.set_stroke(opacity=0.2) for edge in tsp_tour_edges if 6 in edge],
            *[self.get_edge(all_edges, edge).animate.set_stroke(opacity=0.2) for edge in one_tree_edges if 6 in edge],
        )
        self.wait()

        self.play(
            Transform(mst_vertices.vertices[6], original_node_6),
            Transform(tsp_graph.vertices[6], original_node_6_tsp),
            *[tsp_tour_edges[edge].animate.set_stroke(opacity=1) for edge in tsp_tour_edges if 6 in edge],
            *[self.get_edge(all_edges, edge).animate.set_stroke(opacity=1) for edge in one_tree_edges if 6 in edge],
        )
        self.wait()
        best_one_cost = one_tree_cost
        best_one_tree = 6
        current_one_tree_edges = VGroup(*[self.get_edge(all_edges, edge) for edge in one_tree_edges])
        for v_to_ignore in [0, 1, 2, 3, 4, 7, 8, 5]:
            _, _, one_tree_edges, one_tree_cost = get_1_tree(mst_vertices.get_dist_matrix(), v_to_ignore)
            if one_tree_cost > best_one_cost:
                best_one_tree, best_one_cost = v_to_ignore, one_tree_cost
            all_edges_new = mst_vertices.get_all_edges(buff=mst_vertices.vertices[0].width / 2)
            new_one_tree_edges = VGroup(*[self.get_edge(all_edges_new, edge).set_color(REDUCIBLE_YELLOW) for edge in one_tree_edges])
            self.play(
                Transform(current_one_tree_edges, new_one_tree_edges)
            )
            self.wait()

        print('Best one tree', best_one_tree, best_one_cost, 'Optimal TSP', optimal_cost)
        best_cost_1_tree = Text(f"Largest 1-tree cost: {np.round(best_one_cost, 1)}", font=REDUCIBLE_MONO).scale(0.5)
        optimal_tsp_cost = Text(f"Optimal TSP cost: {np.round(optimal_cost, 1)}", font=REDUCIBLE_MONO).scale(0.5)

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

    def highlight_visited_univisited(self, vertices, labels, visited, unvisited, scale=0.7):
        visited_group = VGroup(*[vertices[v].copy().scale(scale) for v in visited]).arrange(RIGHT)
        unvisited_group = VGroup(*[vertices[v].copy().scale(scale) for v in unvisited]).arrange(RIGHT)
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
                "radius": radius
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
