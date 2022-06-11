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
                self.dist_matrix[u][v] = np.round(distance, decimals=1)
                self.dist_matrix[v][u] = np.round(distance, decimals=1)
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

    def get_tour_dist_labels(self, edge_dict):
        dist_label_dict = {}
        for edge in edge_dict:
            u, v = edge
            dist_label = self.get_dist_label(edge_dict[edge], self.dist_matrix[u][v])
            dist_label_dict[edge] = dist_label
        return dist_label_dict

    def get_dist_label(self, edge_mob, distance, scale=0.3):
        return (
            Text(str(distance), font=REDUCIBLE_MONO)
            .set_stroke(BLACK, width=8, background=True, opacity=0.8)
            .scale(scale)
            .move_to(edge_mob.point_from_proportion(0.5))
        )


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
            # self.add(*tour_dist_labels.values())
            self.wait()
            self.remove(*tour_edges.values())
            # self.remove(*tour_dist_labels.values())


class BruteForce(Scene):
    def construct(self):
        pass
