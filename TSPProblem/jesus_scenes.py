from typing import Iterable
from manim import *
from solving_tsp import TSPGraph
from reducible_colors import *
from functions import *
from classes import *
from solver_utils import *

np.random.seed(1)


class TSPAssumptions(Scene):
    def construct(self):
        # self.intro_PIP()
        # self.play(*[FadeOut(mob) for mob in self.mobjects])
        self.present_TSP_graph()
        # self.play(*[FadeOut(mob) for mob in self.mobjects])

    def intro_PIP(self):
        rect = ScreenRectangle(height=4)
        self.play(Write(rect), run_time=2)

    def present_TSP_graph(self):

        graph = TSPGraph(range(8))

        self.play(Write(graph))
        self.wait()

        all_edges = graph.get_all_edges()

        self.play(LaggedStartMap(Write, all_edges.values()))
        self.wait()

        edge_focus = self.focus_on_edges(
            edges_to_focus_on=[(2, 7)], all_edges=all_edges
        )
        self.play(*edge_focus)

        arrow_up = (
            Arrow(
                start=graph.vertices[2],
                end=graph.vertices[7],
                max_tip_length_to_length_ratio=0.041,
            )
            .move_to(all_edges[(2, 7)].start)
            .shift(DOWN * 0.4)
            .scale(0.2)
            .set_color(WHITE)
        )
        arrow_down = (
            Arrow(
                start=graph.vertices[7],
                end=graph.vertices[2],
                max_tip_length_to_length_ratio=0.041,
            )
            .move_to(all_edges[(2, 7)].end)
            .shift(UP * 0.4)
            .scale(0.2)
            .set_color(WHITE)
        )
        dist_label = graph.get_dist_label(all_edges[(2, 7)], graph.dist_matrix[2, 7])
        self.play(FadeIn(arrow_up, arrow_down, dist_label))
        self.play(
            arrow_up.animate.move_to(all_edges[(2, 7)].end).shift(DOWN * 0.4),
            arrow_down.animate.move_to(all_edges[(2, 7)].start).shift(UP * 0.4),
            ShowPassingFlash(
                all_edges[(2, 7)].copy().set_stroke(width=6).set_color(REDUCIBLE_YELLOW)
            ),
            ShowPassingFlash(
                all_edges[(2, 7)]
                .copy()
                .set_stroke(width=6)
                .flip(RIGHT)
                .flip(DOWN)
                .set_color(REDUCIBLE_YELLOW)
            ),
        )

        self.play(FadeOut(arrow_up), FadeOut(arrow_down))
        self.play(
            graph.animate.shift(DOWN),
            dist_label.animate.shift(DOWN),
            *[l.animate.shift(DOWN) for l in all_edges.values()],
        )
        title = (
            Text(
                "Symmetric Traveling Salesman Problem",
                font=REDUCIBLE_FONT,
                weight=BOLD,
            )
            .scale(0.8)
            .to_edge(UP, buff=1)
        )
        self.play(LaggedStartMap(FadeIn, title))
        self.wait()

        self.play(
            FadeOut(title),
        )

        full_graph = VGroup(*graph.vertices.values(), *all_edges.values(), dist_label)

        self.play(full_graph.animate.move_to(LEFT * 3), FadeOut(dist_label))

        # triangle inequality
        all_labels = {
            t: graph.get_dist_label(e, graph.dist_matrix[t]).set_opacity(0)
            for t, e in all_edges.items()
        }

        for i in range(3):
            triang_vertices = sorted(
                np.random.choice(list(graph.vertices.keys()), size=3, replace=False)
            )
            start_node = triang_vertices[0]
            middle_node = triang_vertices[1]
            end_node = triang_vertices[2]

            triangle_edges = [
                (start_node, end_node),
                (start_node, middle_node),
                (middle_node, end_node),
            ]
            triangle_ineq_edges_focus = self.focus_on_edges(triangle_edges, all_edges)
            labels_to_focus = self.focus_on_labels(triangle_edges, all_labels)
            vertices_to_focus = self.focus_on_vertices(triang_vertices, graph.vertices)

            less_than = Text(
                "always less than", font=REDUCIBLE_FONT, weight=BOLD
            ).scale(0.7)

            arrow_text = Text("→", font=REDUCIBLE_FONT, weight=BOLD)

            direct_path = VGroup(
                graph.vertices[start_node].copy().set_stroke(REDUCIBLE_PURPLE),
                arrow_text.copy(),
                graph.vertices[end_node].copy().set_stroke(REDUCIBLE_PURPLE),
            ).arrange(RIGHT, buff=0.2)

            indirect_path = VGroup(
                graph.vertices[start_node].copy().set_stroke(REDUCIBLE_PURPLE),
                arrow_text.copy(),
                graph.vertices[middle_node].copy().set_stroke(REDUCIBLE_PURPLE),
                arrow_text.copy(),
                graph.vertices[end_node].copy().set_stroke(REDUCIBLE_PURPLE),
            ).arrange(RIGHT, buff=0.2)

            both_paths = (
                VGroup(direct_path, less_than, indirect_path)
                .arrange(DOWN, buff=0.5)
                .shift(RIGHT * 3)
            )

            if i == 0:
                self.play(
                    *labels_to_focus,
                    *triangle_ineq_edges_focus,
                    *vertices_to_focus,
                    FadeIn(direct_path),
                    FadeIn(less_than),
                    FadeIn(indirect_path),
                )
            else:
                self.play(
                    *labels_to_focus,
                    *triangle_ineq_edges_focus,
                    *vertices_to_focus,
                    FadeTransform(last_direct_path, direct_path),
                    FadeTransform(last_indirect_path, indirect_path),
                )

            self.play(
                Succession(
                    AnimationGroup(
                        ShowPassingFlash(
                            all_edges[(start_node, end_node)]
                            .copy()
                            .set_stroke(width=6)
                            .set_color(REDUCIBLE_YELLOW),
                            time_width=0.4,
                        ),
                    ),
                    AnimationGroup(
                        Succession(
                            ShowPassingFlash(
                                all_edges[(start_node, middle_node)]
                                .copy()
                                .set_stroke(width=6)
                                .set_color(REDUCIBLE_YELLOW),
                                time_width=0.6,
                            ),
                            ShowPassingFlash(
                                all_edges[(middle_node, end_node)]
                                .copy()
                                .set_stroke(width=6)
                                .set_color(REDUCIBLE_YELLOW),
                                time_width=0.6,
                            ),
                        ),
                    ),
                ),
            )

            last_direct_path = direct_path
            last_indirect_path = indirect_path

    ### UTIL FUNCS

    def focus_on_edges(
        self, edges_to_focus_on: Iterable[tuple], all_edges: Iterable[tuple]
    ):
        edges_animations = []
        for t, e in all_edges.items():
            if not t in edges_to_focus_on:
                edges_animations.append(e.animate.set_opacity(0.3))
            else:
                edges_animations.append(e.animate.set_opacity(1))

        return edges_animations

    def focus_on_vertices(
        self, edges_to_focus_on: Iterable[tuple], all_edges: Iterable[tuple]
    ):
        edges_animations = []
        for t, e in all_edges.items():
            if not t in edges_to_focus_on:
                edges_animations.append(e.animate.set_stroke(REDUCIBLE_PURPLE))
            else:
                edges_animations.append(e.animate.set_stroke(REDUCIBLE_YELLOW))

        return edges_animations

    def focus_on_labels(self, labels_to_show, all_labels):
        labels_animations = []
        for t, e in all_labels.items():
            if not t in labels_to_show:
                labels_animations.append(e.animate.set_opacity(0))
            else:
                labels_animations.append(e.animate.set_opacity(1))

        return labels_animations

    def show_triangle_inequality(
        self,
        i,
        graph,
        all_edges,
        all_labels,
    ):
        triang_vertices = sorted(
            np.random.choice(list(graph.vertices.keys()), size=3, replace=False)
        )
        start_node = triang_vertices[0]
        middle_node = triang_vertices[1]
        end_node = triang_vertices[2]

        triangle_edges = [
            (start_node, end_node),
            (start_node, middle_node),
            (middle_node, end_node),
        ]
        triangle_ineq_edges_focus = self.focus_on_edges(
            edges_to_focus_on=triangle_edges, all_edges=all_edges
        )
        labels_to_focus = self.show_labels(
            labels_to_show=triangle_edges, all_labels=all_labels
        )
        all_labels = [
            graph.get_dist_label(all_edges[e], graph.dist_matrix[e])
            for e in triangle_edges
        ]

        less_than = Text("always less than", font=REDUCIBLE_FONT, weight=BOLD).scale(
            0.7
        )

        arrow_text = Text("→", font=REDUCIBLE_FONT, weight=BOLD)

        direct_path = VGroup(
            graph.vertices[start_node].copy(),
            arrow_text.copy(),
            graph.vertices[end_node].copy(),
        ).arrange(RIGHT, buff=0.2)

        indirect_path = VGroup(
            graph.vertices[start_node].copy(),
            arrow_text.copy(),
            graph.vertices[middle_node].copy(),
            arrow_text.copy(),
            graph.vertices[end_node].copy(),
        ).arrange(RIGHT, buff=0.2)

        both_paths = (
            VGroup(direct_path, less_than, indirect_path)
            .arrange(DOWN, buff=0.5)
            .shift(RIGHT * 3)
        )

        if i > 0:
            last_direct_path = direct_path
            last_indirect_path = indirect_path

        print(i)
        if i == 0:
            self.play(
                *labels_to_focus,
                *triangle_ineq_edges_focus,
                FadeIn(direct_path),
                FadeIn(indirect_path),
            )
        else:

            self.play(
                *labels_to_focus,
                *triangle_ineq_edges_focus,
                Transform(last_direct_path, direct_path),
                Transform(last_indirect_path, indirect_path),
            )

        self.play(
            Succession(
                AnimationGroup(
                    ShowPassingFlash(
                        all_edges[(start_node, end_node)]
                        .copy()
                        .set_stroke(width=6)
                        .set_color(REDUCIBLE_YELLOW),
                        time_width=0.4,
                    ),
                ),
                FadeIn(less_than),
                AnimationGroup(
                    Succession(
                        ShowPassingFlash(
                            all_edges[(start_node, middle_node)]
                            .copy()
                            .set_stroke(width=6)
                            .set_color(REDUCIBLE_YELLOW),
                            time_width=0.6,
                        ),
                        ShowPassingFlash(
                            all_edges[(middle_node, end_node)]
                            .copy()
                            .set_stroke(width=6)
                            .set_color(REDUCIBLE_YELLOW),
                            time_width=0.6,
                        ),
                    ),
                ),
            ),
        )
