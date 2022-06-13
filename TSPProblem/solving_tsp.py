import sys

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
from classes import CustomLabel

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
        else:
            edge_config["buff"] = Dot().radius
        ### Manim bug where buff has no effect for some reason on standard lines
        super().__init__(
            vertices,
            edges,
            vertex_config=vertex_config,
            edge_config=edge_config,
            labels=labels,
            **kwargs,
        )
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
            buff=self.edge_config["buff"],
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
            dist_label = self.get_dist_label(edge_dict[edge], self.dist_matrix[u][v], scale=scale, num_decimal_places=num_decimal_places)
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


class TSPTester(Scene):
    def construct(self):
        graph = TSPGraph([0, 1, 2, 3, 4, 5])
        self.play(FadeIn(graph))
        self.wait()
        print(graph.dist_matrix)

        # all_edges = graph.get_all_edges()

        all_tour_permutations = get_all_tour_permutations(len(graph.vertices), 0)
        for tour in all_tour_permutations:
            tour_edges = graph.get_tour_edges(tour)
            tour_dist_labels = graph.get_tour_dist_labels(tour_edges)
            self.add(*tour_edges.values())
            self.add(*tour_dist_labels.values())
            self.wait()
            self.remove(*tour_edges.values())
            self.remove(*tour_dist_labels.values())

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
        self.play(
            FadeIn(graph)
        )
        self.wait()
        
        graph_with_tour_edges = self.demo_nearest_neighbor(graph)
        
        self.compare_nn_with_optimal(graph_with_tour_edges, graph)

        self.clear()

        self.show_many_large_graph_nn_solutions()
    
    def demo_nearest_neighbor(self, graph):
        glowing_circle = get_glowing_surround_circle(graph.vertices[0])
        self.play(
            FadeIn(glowing_circle)
        )
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
            self.play(
                tour_edges[(prev, vertex)].animate.set_color(REDUCIBLE_YELLOW)
            )
            self.wait()
            seen.add(vertex)
            new_glowing_circle = get_glowing_surround_circle(graph.vertices[vertex])
            new_neighboring_edges = graph.get_neighboring_edges(vertex)
            for key in new_neighboring_edges.copy():
                if key[1] in seen and key[1] != vertex:
                    del new_neighboring_edges[key]
            filtered_prev_edges = [edge_key for edge_key, edge in neighboring_edges.items() if edge_key != (prev, vertex) and edge_key != (vertex, prev)]
            self.play(
                FadeOut(glowing_circle),
                FadeIn(new_glowing_circle),
                *[FadeOut(neighboring_edges[edge_key]) for edge_key in filtered_prev_edges],
            )
            self.wait()
            filtered_new_edges = [edge_key for edge_key, edge in new_neighboring_edges.items() if edge_key != (prev, vertex) and edge_key != (vertex, prev)]

            if len(filtered_new_edges) > 0:
                self.play(
                    *[FadeIn(new_neighboring_edges[edge_key]) for edge_key in filtered_new_edges]
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
        nn_tour, nn_cost = get_nearest_neighbor_solution(original_graph.get_dist_matrix())
        optimal_tour, optimal_cost = get_exact_tsp_solution(original_graph.get_dist_matrix())
        optimal_graph = original_graph.copy()
        optimal_edges = optimal_graph.get_tour_edges(optimal_tour)

        shift_amount = 3.2
        scale = 0.6
        self.play(
            graph_with_tour_edges.animate.scale(scale).shift(LEFT * shift_amount)
        )
        self.wait()

        optimal_graph_tour = self.get_graph_tour_group(optimal_graph, optimal_edges)
        optimal_graph_tour.scale(scale).shift(RIGHT * shift_amount)
        nn_text = self.get_distance_text(nn_cost).next_to(graph_with_tour_edges, UP, buff=1)
        optimal_text = self.get_distance_text(optimal_cost).next_to(optimal_graph_tour, UP, buff=1)

        self.play(
            FadeIn(nn_text)
        )

        self.play(
            FadeIn(optimal_graph_tour)
        )
        self.wait()        


        self.play(
            FadeIn(optimal_text),
        )
        self.wait()

        nn_heuristic = Text("Nearest Neighbor (NN) Heuristic", font=REDUCIBLE_FONT, weight=BOLD)
        nn_heuristic.scale(0.8)
        nn_heuristic.move_to(DOWN * 2.5)

        self.play(
            Write(nn_heuristic)
        )
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

        how_to_compare = Text("How to measure effectiveness of heuristic approach?", font=REDUCIBLE_FONT).scale(0.6)

        how_to_compare.next_to(nn_heuristic, DOWN)

        self.play(
            FadeIn(how_to_compare)
        )
        self.wait()

        self.play(
            FadeOut(nn_heuristic),
            FadeOut(how_to_compare)
        )

        approx_ratio = Tex(r"Approximation ratio $(\alpha) = \frac{\text{heuristic solution}}{\text{optimal solution}}$").scale(0.8).move_to(DOWN * 2.5)

        self.play(
            FadeIn(approx_ratio)
        )

        self.wait()

        example = Tex(r"E.g $\alpha = \frac{28.2}{27.0} \approx 1.044$", r"$\rightarrow$ 4.4\% above optimal").scale(0.7)

        example.next_to(approx_ratio, DOWN)

        self.play(
            Write(example[0])
        )
        self.wait()

        self.play(
            Write(example[1])
        )
        self.wait()

    def show_many_large_graph_nn_solutions(self):
        NUM_VERTICES = 100
        num_iterations = 10
        average_case = Tex(r"On average: $\frac{\text{NN Heuristic}}{\text{Held-Karp Lower Bound}} = 1.25$").scale(0.8).move_to(DOWN * 3.5)
        for _ in range(num_iterations):
            graph = TSPGraph(
                list(range(NUM_VERTICES)),
                labels=False,
                layout=self.get_random_layout(NUM_VERTICES),
            )
            tour, nn_cost = get_nearest_neighbor_solution(graph.get_dist_matrix())
            print('NN cost', nn_cost)
            tour_edges = graph.get_tour_edges(tour)
            tour_edges_group = VGroup(*list(tour_edges.values()))
            graph_with_tour_edges = VGroup(graph, tour_edges_group).scale(0.8)
            self.add(graph_with_tour_edges)
            if _ == 5:
                self.play(
                    FadeIn(average_case)
                )
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
            label = Text(str(v), font=REDUCIBLE_MONO).scale(0.2).move_to(v_mob.get_center())
            labels.add(label)

        return labels

