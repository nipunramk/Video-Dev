from pprint import pprint
from typing import Iterable
from manim import *
from solving_tsp import TSPGraph
from reducible_colors import *
from functions import *
from classes import *
from math import factorial, log10
from scipy.special import gamma
from solver_utils import *
from manim.mobject.geometry.tips import ArrowTriangleTip
from itertools import combinations, permutations

np.random.seed(2)
config["assets_dir"] = "assets"


class TSPAssumptions(MovingCameraScene):
    def construct(self):

        self.intro_PIP()
        self.wait()
        self.play(*[FadeOut(mob) for mob in self.mobjects])
        self.wait()

        self.present_TSP_graph()
        self.wait()
        self.play(*[FadeOut(mob) for mob in self.mobjects])

    def intro_PIP(self):
        rect = ScreenRectangle(height=4)
        self.play(Write(rect), run_time=2)

    def present_TSP_graph(self):
        frame = self.camera.frame

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
        self.wait()

        full_graph = VGroup(*graph.vertices.values(), *all_edges.values(), dist_label)

        self.play(
            FadeOut(dist_label), full_graph.animate.move_to(LEFT * 3), run_time=0.7
        )
        self.wait()

        # triangle inequality
        all_labels = {
            t: graph.get_dist_label(e, graph.dist_matrix[t]).set_opacity(0)
            for t, e in all_edges.items()
        }

        for i in range(10):
            if i == 5:
                triang_title = (
                    Text("Triangle Inequality", font=REDUCIBLE_FONT, weight=BOLD)
                    .scale(0.8)
                    .move_to(frame.get_top())
                )
                self.play(FadeIn(triang_title), frame.animate.shift(UP * 0.8))
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
        # end for loop triangle inequality

    ### UTIL FUNCS

    def focus_on_edges(
        self, edges_to_focus_on: Iterable[tuple], all_edges: Iterable[tuple]
    ):
        edges_animations = []

        edges_to_focus_on = list(
            map(lambda t: (t[1], t[0]) if t[0] > t[1] else t, edges_to_focus_on)
        )
        for t, e in all_edges.items():
            if not t in edges_to_focus_on:
                edges_animations.append(e.animate.set_opacity(0.3))
            else:
                edges_animations.append(e.animate.set_opacity(1))

        return edges_animations

    def focus_on_vertices(
        self, vertices_to_focus_on: Iterable[int], all_vertices: Iterable[tuple]
    ):
        edges_animations = []
        for t, e in all_vertices.items():
            if not t in vertices_to_focus_on:
                edges_animations.append(e.animate.set_stroke(REDUCIBLE_PURPLE))
            else:
                edges_animations.append(e.animate.set_stroke(REDUCIBLE_YELLOW))

        return edges_animations

    def focus_on_labels(self, labels_to_show, all_labels):
        labels_animations = []

        labels_to_show = list(
            map(lambda t: (t[1], t[0]) if t[0] > t[1] else t, labels_to_show)
        )
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


class CustomArrow(Line):
    """
    Custom arrow with tip in the middle instead of the end point
    to represent direction but not mistake it with a directed graph.
    """

    def __init__(
        self,
        start=LEFT,
        end=RIGHT,
        stroke_width=6,
        **kwargs,
    ):
        super().__init__(
            start=start,
            end=end,
            stroke_width=stroke_width,
            stroke_color=REDUCIBLE_VIOLET,
            **kwargs,
        )
        self.add_tip()

        self.tip.scale(0.7).move_to(self.point_from_proportion(0.25))

    def add_tip(self, tip=None, tip_shape=None, tip_length=None, at_start=False):
        """
        Overridden method to remove the `reset_endpoints_based_on_tip call`
        so the line actually reaches to the nodes in our particular case.
        """
        if tip is None:
            tip = self.create_tip(tip_shape, tip_length, at_start)
        else:
            self.position_tip(tip, at_start)
        self.asign_tip_attr(tip, at_start)
        self.add(tip)
        return self


class BruteForce(TSPAssumptions):
    def construct(self):

        cities = 5
        graph = TSPGraph(range(cities))
        all_edges = graph.get_all_edges()

        tour_perms = get_all_tour_permutations(cities, 0)

        self.play(Write(graph))
        self.play(LaggedStartMap(Write, all_edges.values()))
        self.play(
            graph.animate.shift(RIGHT * 4),
            VGroup(*all_edges.values()).animate.shift(RIGHT * 4),
            run_time=0.8,
        )
        self.wait()

        empty_mobs = (
            VGroup(*[Dot() for c in range(factorial(cities - 1) // 2)])
            .arrange_in_grid(cols=4, buff=1.5, row_heights=np.repeat(0.9, 10))
            .shift(LEFT * 3 + UP * 0.2)
        )
        permutations_mobs = VGroup()

        for i, tour in enumerate(tour_perms):
            tour_edges = get_edges_from_tour(tour)

            edges_animation = self.focus_on_edges(tour_edges, all_edges)
            self.play(*edges_animation, run_time=1 / (i + 1))

            curr_tour_cost = get_cost_from_permutation(graph.dist_matrix, tour_edges)
            curr_tour = (
                VGroup(
                    *[v.copy() for v in graph.vertices.values()],
                    *[e.copy().set_stroke(width=2) for e in all_edges.values()],
                )
                .scale(0.3)
                .move_to(empty_mobs[i])
            )
            cost_text = (
                Text(f"{curr_tour_cost:.2f}", font=REDUCIBLE_MONO)
                .scale(0.3)
                .next_to(curr_tour, DOWN, buff=0.2)
            )
            permutations_mobs.add(curr_tour, cost_text)

            self.play(FadeIn(curr_tour), FadeIn(cost_text), run_time=1 / (i + 1))

        self.wait()
        self.play(
            LaggedStart(
                FadeOut(permutations_mobs),
                *[FadeOut(e) for e in all_edges.values()],
                graph.animate.move_to(ORIGIN),
            )
        )
        self.wait()

        ##################################33
        # BIG EXAMPLE

        big_cities = 12
        big_graph = TSPGraph(
            range(big_cities), label_scale=0.5, layout_scale=2.4, layout="circular"
        )
        all_edges_bg = big_graph.get_all_edges()

        self.play(FadeTransform(graph, big_graph))
        self.play(LaggedStartMap(Write, all_edges_bg.values()))

        all_tours = get_all_tour_permutations(big_cities, 0, 600)
        edge_tuples_tours = [get_edges_from_tour(tour) for tour in all_tours]
        pprint(len(all_tours))

        # change line here back to 200
        for i, tour_edges in enumerate(edge_tuples_tours[:200]):
            anims = self.focus_on_edges(tour_edges, all_edges_bg)
            self.play(*anims, run_time=1 / (5 * i + 1))

        self.wait()

        self.play(*[FadeOut(e) for e in all_edges_bg.values()])
        self.wait()

        # select random node to start
        for _ in range(10):
            random_indx = np.random.randint(0, big_cities)
            anims = self.focus_on_vertices(
                [random_indx],
                big_graph.vertices,
            )
            self.play(*anims, run_time=0.1)

        # of cities left
        cities_counter_label = Text(
            "# of cities left: ", font=REDUCIBLE_FONT, weight=MEDIUM
        ).scale(0.4)
        cities_counter = Text(str(big_cities), font=REDUCIBLE_MONO, weight=BOLD).scale(
            0.4
        )
        full_label = (
            VGroup(cities_counter_label, cities_counter)
            .arrange(RIGHT, buff=0.1, aligned_edge=UP)
            .next_to(big_graph, RIGHT, aligned_edge=DOWN)
        )

        # start creating the loop step by step
        last_node = random_indx  # take the last one from the other animation
        first_node = last_node

        valid_nodes = list(range(big_cities))
        valid_nodes.remove(last_node)

        self.play(FadeIn(full_label, shift=UP * 0.3))

        path_builder = VGroup()

        # change line here back to big_cities
        for i in range(big_cities):
            if len(valid_nodes) == 0:
                # we finished, so we go back home and break out of the loop
                new_counter = (
                    Text(str(len(valid_nodes)), font=REDUCIBLE_MONO, weight=BOLD)
                    .scale(0.4)
                    .move_to(cities_counter)
                )
                anims = self.focus_on_vertices(
                    [last_node],
                    big_graph.vertices,
                )

                self.play(Transform(cities_counter, new_counter), *anims, run_time=0.1)

                edge = big_graph.create_edge(last_node, first_node)
                path_builder.add(edge)
                self.wait()
                self.play(
                    Write(edge),
                )
                break

            # start from random index
            anims = self.focus_on_vertices(
                [last_node],
                big_graph.vertices,
            )
            new_counter = (
                Text(str(len(valid_nodes)), font=REDUCIBLE_MONO, weight=BOLD)
                .scale(0.4)
                .move_to(cities_counter)
            )
            self.play(Transform(cities_counter, new_counter), *anims, run_time=0.1)

            # create all edges from this vertex
            vertex_edges = {
                (last_node, v): big_graph.create_edge(last_node, v).set_opacity(0.5)
                for v in range(big_cities)
                if v != last_node
            }

            self.play(*[Write(e) for e in vertex_edges.values()])
            next_node = np.random.choice(valid_nodes)
            valid_nodes.remove(next_node)

            edge = vertex_edges.pop((last_node, next_node))
            path_builder.add(edge)
            self.play(
                ShowPassingFlash(
                    edge.copy().set_stroke(REDUCIBLE_YELLOW, width=7, opacity=1)
                ),
                edge.animate.set_opacity(1),
                *[e.animate.set_opacity(0) for e in vertex_edges.values()],
            )

            last_node = next_node

        self.wait()

        n_minus_one_factorial = (
            Text("(n - 1)!", font=REDUCIBLE_FONT, weight=BOLD)
            .scale(1.2)
            .set_stroke(width=10, background=True)
        )
        self.play(Write(n_minus_one_factorial, scale=0.95))
        self.wait()
        self.play(
            FadeOut(path_builder),
            FadeOut(n_minus_one_factorial, scale=0.95),
            FadeOut(big_graph),
            FadeOut(full_label, scale=0.95),
        )

        # go back to small example to show combinations
        small_cities = 4
        small_graph = TSPGraph(range(small_cities))

        all_possible_tours = get_all_tour_permutations(
            small_cities, 0, return_duplicates=True
        )

        all_tours = VGroup()
        for tour_symms in all_possible_tours:
            # tour is a list of 2 tours that are symmetric. bear that in mind!
            tour_pairs = VGroup()

            for tour in tour_symms:
                graph = VGroup()
                graph.add(*[v.copy() for v in small_graph.vertices.values()])

                graph.add(
                    *list(
                        small_graph.get_tour_edges(tour, edge_type=CustomArrow).values()
                    )
                ).scale(0.6)
                tour_pairs.add(graph)

            all_tours.add(tour_pairs.arrange(DOWN, buff=0.3))

        all_tours.arrange_in_grid(
            cols=factorial(small_cities - 1) // 2,
            row_heights=np.repeat(2.5, factorial(small_cities - 1) // 2),
            col_widths=np.repeat(3.5, factorial(small_cities - 1) // 2),
        ).scale_to_fit_width(config.frame_width - 4)

        self.play(*[FadeIn(t[0]) for t in all_tours])
        self.wait()
        self.play(*[FadeIn(t[1], shift=DOWN * 0.9) for t in all_tours])

        surr_rect = SurroundingRectangle(
            VGroup(all_tours[0]), color=REDUCIBLE_YELLOW, buff=0.3
        )
        annotation = (
            Text("These two are the same", font=REDUCIBLE_FONT, weight=BOLD)
            .next_to(surr_rect, UP, buff=0.2)
            .scale_to_fit_width(surr_rect.width)
            .set_color(REDUCIBLE_YELLOW)
        )

        def annotation_updater(mob):
            mob.next_to(surr_rect, UP, buff=0.2)

        annotation.add_updater(annotation_updater)

        self.play(Write(surr_rect), Write(annotation))

        self.play(
            surr_rect.animate.move_to(all_tours[1]),
        )
        self.play(
            surr_rect.animate.move_to(all_tours[2]),
        )
        self.wait()

        self.play(
            FadeOut(surr_rect),
            FadeOut(annotation),
            *[FadeOut(t[1], shift=DOWN * 0.9) for t in all_tours],
        )

        annotation_2 = (
            Text(
                "These correspond to half of the total",
                font=REDUCIBLE_FONT,
                weight=MEDIUM,
            )
            .scale(0.8)
            .shift(DOWN)
        )

        bold_template = TexTemplate()
        bold_template.add_to_preamble(r"\usepackage{bm}")
        n_minus_one_factorial_over_two = (
            Tex(r"$\bm{\frac{(n - 1)!}{2}}$", tex_template=bold_template)
            .scale(1.8)
            .next_to(annotation_2, DOWN, buff=0.5)
        )

        self.play(Write(annotation_2))
        self.play(Write(n_minus_one_factorial_over_two))

        self.wait()
        self.play(
            FadeOut(n_minus_one_factorial_over_two),
            FadeOut(annotation_2),
            *[FadeOut(t[0]) for t in all_tours],
        )
        self.wait()

        # show big number
        np.random.seed(110)
        random_graph = TSPGraph(
            range(20),
            layout="circular",
        ).shift(UP * 0.7)

        # scale down vertices
        [v.scale(0.7) for v in random_graph.vertices.values()]

        # recalculate edges
        all_edges = random_graph.get_all_edges(buff=random_graph.vertices[0].width / 2)

        # set edges opacity down
        [e.set_opacity(0.3) for e in all_edges.values()]

        twenty_factorial = (
            Text(
                f"(20 - 1)! / 2 = {factorial(20  - 1) // 2:,}".replace(",", " "),
                font=REDUCIBLE_MONO,
                weight=BOLD,
            )
            .scale(0.7)
            .next_to(random_graph, DOWN, buff=1)
        )

        self.play(FadeIn(random_graph))
        self.play(LaggedStartMap(Write, all_edges.values()))

        self.play(AddTextLetterByLetter(twenty_factorial))
        self.wait()

    def get_random_layout(self, N):
        random_points_in_frame = get_random_points_in_frame(N)
        return {v: point for v, point in zip(range(N), random_points_in_frame)}


class ProblemComplexity(TSPAssumptions):
    def construct(self):
        self.dynamic_programming_simulation()
        self.wait()
        self.play(*[FadeOut(mob) for mob in self.mobjects])

        self.np_hard_problems()
        self.wait()

        self.plot_graphs()
        self.wait()
        self.play(*[FadeOut(mob) for mob in self.mobjects])

    def dynamic_programming_simulation(self):
        cities = 4

        graph = TSPGraph(range(cities)).shift(RIGHT * 3)
        all_edges = graph.get_all_edges()

        # make the whole graph a bit bigger
        VGroup(graph, *all_edges.values()).scale(1.4)

        all_labels = {
            t: graph.get_dist_label(e, graph.dist_matrix[t])
            for t, e in all_edges.items()
        }

        [e.set_opacity(0) for t, e in all_edges.items()]

        self.play(LaggedStartMap(FadeIn, graph))

        cities_list = list(range(1, cities))
        start_city = 0

        curr_tour_txt = Text("Current tour:", font=REDUCIBLE_FONT).scale(0.6)
        best_subtour_txt = Text("Best subtour:", font=REDUCIBLE_FONT).scale(0.6)
        curr_cost_txt = Text("Current cost:", font=REDUCIBLE_FONT).scale(0.6)
        best_cost_txt = Text("Best cost:", font=REDUCIBLE_FONT).scale(0.6)

        text_vg = (
            VGroup(curr_tour_txt, curr_cost_txt, best_subtour_txt, best_cost_txt)
            .arrange(DOWN, aligned_edge=LEFT, buff=0.3)
            .to_edge(LEFT)
        )

        curr_tour_str = Text(f"", font=REDUCIBLE_MONO).next_to(curr_tour_txt)
        best_tour_str = Text(f"", font=REDUCIBLE_MONO).next_to(best_subtour_txt)
        curr_cost_str = Text(f"", font=REDUCIBLE_MONO).next_to(curr_cost_txt)
        best_cost_str = Text(f"", font=REDUCIBLE_MONO).next_to(best_cost_txt)

        explanation = Text("").next_to(text_vg, UP, buff=1, aligned_edge=LEFT)

        self.play(FadeIn(text_vg))
        for i in range(cities - 1):
            costs = {}
            internal_perms = list(permutations(cities_list, i + 1))

            for sub_tour in internal_perms:
                tour = [*sub_tour, start_city]

                tour_edges = graph.get_tour_edges(tour)

                # remove the last one since we are talking about sub tours
                tour_edge_tuples = get_edges_from_tour(tour)[:-1]

                curr_cost = get_cost_from_edges(tour_edge_tuples, graph.dist_matrix)
                print(tour_edge_tuples, curr_cost)

                costs[tuple(tour)] = curr_cost

                new_curr_tour = (
                    Text(f"{tour}", font=REDUCIBLE_MONO)
                    .scale(0.6)
                    .next_to(curr_tour_txt)
                )

                new_curr_cost = (
                    Text(f"{np.round(curr_cost, 1)}", font=REDUCIBLE_MONO)
                    .scale(0.6)
                    .next_to(curr_cost_txt)
                )

                explanation_str = f"Going from {tour[0]} to {tour[-1]} through {i} {'cities' if i != 1 else 'city'}"
                new_explanation = (
                    Text(
                        explanation_str,
                        font=REDUCIBLE_FONT,
                        t2f={
                            str(tour[0]): REDUCIBLE_MONO,
                            str(tour[-1]): REDUCIBLE_MONO,
                            str(i): REDUCIBLE_MONO,
                        },
                    )
                    .scale(0.6)
                    .next_to(text_vg, UP, buff=1, aligned_edge=LEFT)
                )

                edges_anims = self.focus_on_edges(tour_edges, all_edges=all_edges)
                labels_anims = self.focus_on_labels(tour_edge_tuples, all_labels)

                self.play(
                    *edges_anims,
                    *labels_anims,
                    Transform(curr_tour_str, new_curr_tour),
                    Transform(curr_cost_str, new_curr_cost),
                    Transform(explanation, new_explanation),
                    run_time=0.5,
                )

            # find best subtour and display it
            best_subtour, best_cost = min(costs.items(), key=lambda x: x[1])
            print(costs, best_subtour, best_cost)

            new_best_tour = (
                Text(f"{tour}", font=REDUCIBLE_MONO)
                .scale(0.6)
                .next_to(best_subtour_txt)
            )

            new_best_cost = (
                Text(f"{np.round(best_cost, 1)}", font=REDUCIBLE_MONO)
                .scale(0.6)
                .next_to(best_cost_txt)
            )

            self.play(
                Transform(best_tour_str, new_best_tour),
                Transform(best_cost_str, new_best_cost),
            )

            self.wait()

    def np_hard_problems(self):
        # explanation about NP problems
        tsp_problem = (
            Module(["Traveling Salesman", "Problem"], text_weight=BOLD)
            .scale(0.8)
            .shift(DOWN * 0.7)
        )
        np_hard_problems = Module(
            "NP Hard Problems",
            fill_color=REDUCIBLE_GREEN_DARKER,
            stroke_color=REDUCIBLE_GREEN_LIGHTER,
            width=12,
            height=6,
            text_weight=BOLD,
            text_scale=0.6,
            text_position=UP,
        )
        problems = [
            "Integer Programming",
            "Knapsack Problem",
            "Bin Packing",
            "Complete Coloring",
        ]
        modules = VGroup(*[Module(p, text_weight=BOLD).scale(0.6) for p in problems])
        modules[0].move_to(tsp_problem).shift(LEFT * 2 + UP * 1.5)
        modules[1].move_to(tsp_problem).shift(LEFT * 2 + DOWN * 1.2)
        modules[2].move_to(tsp_problem).shift(RIGHT * 1.5 + UP * 1.2)
        modules[3].move_to(tsp_problem).shift(RIGHT * 1.9 + DOWN * 1)

        self.play(Write(np_hard_problems))
        self.play(FadeIn(tsp_problem, scale=1.05))
        self.wait()

        self.play(
            LaggedStart(*[FadeIn(m, scale=1.05) for m in modules], lag_ratio=1),
            run_time=5,
        )
        self.wait()
        self.play(*[FadeOut(m, scale=0.95) for m in self.mobjects])

    def plot_graphs(self):

        # plot graphs
        eval_range = [0, 20]
        num_plane = Axes(
            x_range=eval_range,
            y_range=[0, 400],
            y_length=10,
            x_length=15,
            tips=False,
            axis_config={"include_ticks": False},
        ).to_corner(DL)

        bold_template = TexTemplate()
        bold_template.add_to_preamble(r"\usepackage{bm}")

        constant_plot = (
            num_plane.plot(lambda x: 5, x_range=eval_range)
            .set_color(REDUCIBLE_BLUE)
            .set_stroke(width=5)
        )
        constant_tag = (
            Tex(r"$\bm{O(1)}$", tex_template=bold_template)
            .set_color(REDUCIBLE_BLUE)
            .scale(0.6)
            .next_to(constant_plot, UP)
        )

        linear_plot = (
            num_plane.plot(lambda x: x, x_range=eval_range)
            .set_color(REDUCIBLE_PURPLE)
            .set_stroke(width=5)
        )
        linear_tag = (
            Tex(r"$\bm{O(n)}$", tex_template=bold_template)
            .set_color(REDUCIBLE_PURPLE)
            .scale(0.6)
            .next_to(linear_plot.point_from_proportion(0.7), UP)
        )

        quad_plot = (
            num_plane.plot(lambda x: x ** 2, x_range=eval_range)
            .set_color(REDUCIBLE_VIOLET)
            .set_stroke(width=5)
        )
        quad_tag = (
            Tex(r"$\bm{O(n^2)}$", tex_template=bold_template)
            .set_color(REDUCIBLE_VIOLET)
            .scale(0.6)
            .next_to(quad_plot.point_from_proportion(0.5), RIGHT)
        )

        poly_plot = (
            num_plane.plot(lambda x: 3 * x ** 2 + 2 * x, x_range=eval_range)
            .set_color(REDUCIBLE_YELLOW)
            .set_stroke(width=5)
        )
        poly_tag = (
            Tex(r"$\bm{O(3n^2+2n)}$", tex_template=bold_template)
            .set_color(REDUCIBLE_YELLOW)
            .scale(0.6)
            .next_to(poly_plot.point_from_proportion(0.25), RIGHT)
        )

        exponential_plot = (
            num_plane.plot(lambda x: 2 ** x, x_range=[0, 10])
            .set_color(REDUCIBLE_ORANGE)
            .set_stroke(width=5)
        )
        exp_tag = (
            Tex(r"$\bm{O(2^n)}$", tex_template=bold_template)
            .set_color(REDUCIBLE_ORANGE)
            .scale(0.6)
            .next_to(exponential_plot.point_from_proportion(0.2), RIGHT)
        )

        factorial_plot = (
            num_plane.plot(
                lambda x: gamma(x) if x > 1 else x ** 2,
                x_range=[0, 10],
            )
            .set_color(REDUCIBLE_CHARM)
            .set_stroke(width=5)
        )

        factorial_tag = (
            Tex(r"$\bm{O(n!)}$", tex_template=bold_template)
            .set_color(REDUCIBLE_CHARM)
            .scale(0.6)
            .next_to(factorial_plot.point_from_proportion(0.001), LEFT)
        )

        plots = [
            constant_plot,
            linear_plot,
            quad_plot,
            poly_plot,
            exponential_plot,
            factorial_plot,
        ]
        tags = [
            constant_tag,
            linear_tag,
            quad_tag,
            poly_tag,
            exp_tag,
            factorial_tag,
        ]
        self.play(
            Write(
                num_plane.x_axis,
            ),
            Write(
                num_plane.y_axis,
            ),
        )
        self.play(LaggedStart(*[Write(p) for p in plots]))
        self.play(*[FadeIn(t, scale=0.95) for t in tags])

        self.play(
            constant_plot.animate.set_stroke(opacity=0.3),
            linear_plot.animate.set_stroke(opacity=0.3),
            quad_plot.animate.set_stroke(opacity=0.3),
            poly_plot.animate.set_stroke(opacity=0.3),
            factorial_plot.animate.set_stroke(opacity=0.3),
            constant_tag.animate.set_opacity(0.3),
            linear_tag.animate.set_opacity(0.3),
            quad_tag.animate.set_opacity(0.3),
            poly_tag.animate.set_opacity(0.3),
            factorial_tag.animate.set_opacity(0.3),
        )


class TransitionOtherApproaches(TSPAssumptions):
    def construct(self):
        bg = ImageMobject("usa-map-satellite-markers.png").scale_to_fit_width(
            config.frame_width
        )
        dark_filter = ScreenRectangle(height=10).set_fill(BLACK, opacity=0.2)
        self.play(FadeIn(bg), FadeIn(dark_filter))
        self.wait()
        # self.add(
        #     Axes(
        #         axis_config={"include_numbers": True},
        #         x_length=config.frame_width,
        #         y_length=config.frame_height,
        #         tips=False,
        #     )
        # )

        city_coords = [
            (-5.95, 0.7, 0),  # SF
            (-5.8, 0.55, 0),  # san jose
            (-4.9, -0.45, 0),  # LA
            (-4.6, -0.87, 0),  # San diego
            (-3.35, -0.65, 0),  # phoenix
            (0.45, -0.85, 0),  # dallas
            (0, -1.8, 0),  # san antonio
            (0.8, -1.75, 0),  # houston
            (2.74, 2, 0),  # chicago
            (6.1, 1.65, 0),  # new york
            (5.85, 1.35, 0),  # philadelphia
        ]
        nodes_per_core = 10
        total_number_of_nodes = len(city_coords) * nodes_per_core
        graph = TSPGraph(
            range(total_number_of_nodes),
            vertex_config={
                "fill_opacity": 1,
                "fill_color": REDUCIBLE_YELLOW,
                "stroke_color": REDUCIBLE_PURPLE_DARKER,
                "stroke_width": 0,
            },
            labels=False,
            layout=self.get_city_layout(
                core_cities=city_coords, nodes_per_core=nodes_per_core
            ),
        )

        # first len(city_coords) are actual cities, so we give them more relevance
        [
            v.scale(0.8)
            .set_fill(REDUCIBLE_VIOLET)
            .set_stroke(REDUCIBLE_PURPLE_DARKER, width=3)
            for v in list(graph.vertices.values())[: len(city_coords)]
        ]

        # these are the tiny nodes around every city
        [v.scale(0.3) for v in list(graph.vertices.values())[len(city_coords) :]]
        self.add_foreground_mobjects(*list(graph.vertices.values())[: len(city_coords)])

        self.play(
            LaggedStart(*[FadeIn(v, scale=0.7) for v in graph.vertices.values()]),
            run_time=3,
        )
        self.wait()

        all_edges = graph.get_all_edges(
            buff=graph.vertices[len(city_coords) + 1].width / 2
        )
        [e.set_opacity(0) for e in all_edges.values()]

        tour_perms = get_all_tour_permutations(total_number_of_nodes, 0, max_cap=1)
        print(len(tour_perms))

        edges_perms = [get_edges_from_tour(t) for t in tour_perms]

        cost_indicator = (
            Text(
                f"Distance: {get_cost_from_edges(edges_perms[0], graph.dist_matrix):.2f}",
                font=REDUCIBLE_FONT,
                t2f={
                    f"{get_cost_from_edges(edges_perms[0], graph.dist_matrix):.2f}": REDUCIBLE_MONO
                },
                weight=BOLD,
            )
            .set_stroke(width=4, background=True)
            .scale(0.6)
            .to_corner(DL, buff=0.5)
        )

        # by changing the slice size you can create a longer or shorter example
        for i, tour_edges in enumerate(edges_perms[:5]):

            # the all_edges dict only stores the edges in ascending order
            tour_edges = list(
                map(lambda x: x if x[0] < x[1] else (x[1], x[0]), tour_edges)
            )
            cost = get_cost_from_edges(tour_edges, graph.dist_matrix)
            new_cost_indicator = (
                Text(
                    f"Distance: {cost:.2f}",
                    font=REDUCIBLE_FONT,
                    t2f={f"{cost:.2f}": REDUCIBLE_MONO},
                    weight=BOLD,
                )
                .set_stroke(width=4, background=True)
                .scale(0.6)
                .to_corner(DL, buff=0.5)
            )

            if i == 0:
                anims = LaggedStart(
                    *[Write(all_edges[e].set_opacity(1)) for e in tour_edges]
                )
                self.play(
                    anims,
                    Transform(cost_indicator, new_cost_indicator),
                    run_time=1 / (5 * i + 1),
                )

            else:
                anims = self.focus_on_edges(
                    tour_edges,
                    all_edges=all_edges,
                )

                self.play(
                    *anims,
                    Transform(cost_indicator, new_cost_indicator),
                    run_time=1 / (5 * i + 1),
                )

        print("Finished looping through tours")

        self.wait()
        # transition to nearest neighbour example

        self.play(*[all_edges[e].animate.set_opacity(0) for e in tour_edges])
        print("finished deleting edges")
        self.wait()

        nn_tour, nn_cost = get_nearest_neighbor_solution(graph.dist_matrix)
        nn_edges = get_edges_from_tour(nn_tour)
        nn_edges = list(map(lambda x: x if x[0] < x[1] else (x[1], x[0]), nn_edges))

        nn_cost_indicator = (
            Text(
                f"Distance: {nn_cost:.2f}",
                font=REDUCIBLE_FONT,
                t2f={f"{nn_cost:.2f}": REDUCIBLE_MONO},
                weight=BOLD,
            )
            .set_stroke(width=4, background=True)
            .scale(0.6)
            .to_corner(DL, buff=0.5)
        )

        self.play(
            dark_filter.animate.set_opacity(0.4),
            LaggedStart(*[all_edges[e].animate.set_opacity(1) for e in nn_edges]),
            Transform(cost_indicator, nn_cost_indicator),
        )

    def get_specific_layout(self, *coords):
        # dict with v number and coordinate
        return {v: coord for v, coord in enumerate(coords)}

    def get_normal_dist_layout(self, N):

        x_values = np.random.normal(-0.3, 1.8, size=N)
        y_values = np.random.normal(0, 0.7, size=N)

        return {
            v: (point[0], point[1], 0)
            for v, point in enumerate(zip(x_values, y_values))
        }

    def get_city_layout(self, core_cities: Iterable[tuple], nodes_per_core: int = 10):
        v_dict = {v: coord for v, coord in enumerate(core_cities)}

        # for each city, we are gona spread nodes_per_core more cities around it
        index = len(core_cities)
        for v in core_cities:
            x_values = np.random.normal(v[0], 0.1, size=nodes_per_core)
            y_values = np.random.normal(v[1], 0.1, size=nodes_per_core)
            for x, y in zip(x_values, y_values):
                v_dict[index] = (x, y, 0)
                index += 1

        return v_dict

    def focus_on_edges(
        self,
        edges_to_focus_on: Iterable[tuple],
        all_edges: Iterable[tuple],
        min_opacity=0.0001,
    ):
        edges_animations = []

        edges_to_focus_on = list(
            map(lambda t: (t[1], t[0]) if t[0] > t[1] else t, edges_to_focus_on)
        )
        for t, e in all_edges.items():
            if not t in edges_to_focus_on:
                edges_animations.append(e.animate.set_opacity(min_opacity))
            else:
                edges_animations.append(e.animate.set_opacity(1))

        return edges_animations
