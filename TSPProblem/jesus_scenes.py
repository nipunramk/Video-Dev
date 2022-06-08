from typing import Iterable
from manim import *
from solving_tsp import TSPGraph
from reducible_colors import *
from functions import *
from classes import *
from solver_utils import *


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
            ShowPassingFlash(all_edges[(2, 7)].copy().set_color(REDUCIBLE_YELLOW)),
            ShowPassingFlash(
                all_edges[(2, 7)]
                .copy()
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
            # graph.animate.shift(UP),
            # dist_label.animate.shift(UP),
            FadeOut(title),
            # *[l.animate.shift(UP) for l in all_edges.values()],
        )

        full_graph = VGroup(*graph.vertices.values(), *all_edges.values(), dist_label)

        self.play(full_graph.animate.move_to(LEFT * 3))

        # triangle inequality
        triangle_edges = [(2, 3), (2, 7), (3, 7)]
        triangle_ineq_edges_focus = self.focus_on_edges(
            edges_to_focus_on=triangle_edges, all_edges=all_edges
        )
        triangle_labels = [
            graph.get_dist_label(all_edges[e], graph.dist_matrix[e])
            for e in triangle_edges
        ]

        self.play(*[FadeIn(l) for l in triangle_labels], *triangle_ineq_edges_focus)

        self.wait()
        less_than = Text("always less than", font=REDUCIBLE_FONT, weight=BOLD).scale(
            0.7
        )

        arrow_text = Text("→", font=REDUCIBLE_FONT, weight=BOLD)
        direct_path = VGroup(
            graph.vertices[3].copy(), arrow_text.copy(), graph.vertices[7].copy()
        ).arrange(RIGHT, buff=0.2)
        indirect_path = VGroup(
            graph.vertices[3].copy(),
            arrow_text.copy(),
            graph.vertices[2].copy(),
            arrow_text.copy(),
            graph.vertices[7].copy(),
        ).arrange(RIGHT, buff=0.2)
        both_paths = (
            VGroup(direct_path, less_than, indirect_path)
            .arrange(DOWN, buff=0.5)
            .shift(RIGHT * 3)
        )

        self.play(
            Succession(
                AnimationGroup(
                    FadeIn(direct_path),
                    ShowPassingFlash(
                        all_edges[(3, 7)].copy().set_color(REDUCIBLE_YELLOW),
                        time_width=0.4,
                    ),
                ),
                FadeIn(less_than),
                AnimationGroup(
                    FadeIn(indirect_path),
                    Succession(
                        ShowPassingFlash(
                            all_edges[(2, 3)]
                            .copy()
                            .flip(RIGHT)
                            .flip(DOWN)
                            .set_color(REDUCIBLE_YELLOW),
                            time_width=0.6,
                        ),
                        ShowPassingFlash(
                            all_edges[(2, 7)].copy().set_color(REDUCIBLE_YELLOW),
                            time_width=0.6,
                        ),
                    ),
                ),
            ),
        )

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
