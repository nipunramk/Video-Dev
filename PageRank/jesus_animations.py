import sys


sys.path.insert(1, "common/")

from manim import *

from markov_chain import *
from reducible_colors import *


class CustomMatrix(VGroup):
    def __init__(self, matrix: np.ndarray, color=REDUCIBLE_VIOLET) -> None:

        self.shape = matrix.shape
        self.matrix = matrix
        numbers_to_text = [
            Text(f"{num:.1f}", font=REDUCIBLE_MONO, color=color)
            for num in matrix.ravel()
        ]
        dist_matrix = VGroup(*numbers_to_text)
        dist_matrix.arrange_in_grid(n_rows=self.shape[0], buff=0.4)

        super().__init__(dist_matrix)


class TransitionMatrix(MovingCameraScene):
    def construct(self):
        markov_ch = MarkovChain(
            5,
            edges=[
                (2, 0),
                (3, 0),
                (4, 0),
                (2, 3),
                (0, 3),
                (3, 4),
                (4, 1),
                (2, 1),
                (0, 2),
                (1, 2),
            ],
        )

        markov_ch_mob = MarkovChainGraph(
            markov_ch,
            curved_edge_config={"radius": 2},
            layout_scale=2.6,
        )

        markov_ch_sim = MarkovChainSimulator(markov_ch, markov_ch_mob, num_users=50)
        users = markov_ch_sim.get_users()

        dot_product = (
            MathTex(
                r"\pi_{n+1}(0) &= \pi_{n}(3) * P(3, 0)\\ &+ \pi_{n}(2) * P(2, 0)\\ &+ \pi_{n}(4) * P(4, 0)"
            )
            .scale(0.7)
            .shift(LEFT * 5.5 + DOWN * 1.5)
        )
        annotation = (
            Text(
                "Probability of ending in state 0 in the next step:",
                font=REDUCIBLE_FONT,
                weight=BOLD,
            )
            .scale_to_fit_width(dot_product.width)
            .next_to(dot_product, UP, buff=0.15)
        )

        trans_matrix_mob = self.matrix_to_mob(markov_ch.get_transition_matrix())

        p_equals = (
            MarkupText("P<sub>0</sub> = ", font=REDUCIBLE_FONT, weight=BOLD)
            .scale(0.3)
            .next_to(trans_matrix_mob, LEFT)
        )

        vertices_down = VGroup(
            *[dot.copy().scale(0.4) for dot in markov_ch_mob.vertices.values()]
        ).arrange(DOWN, buff=0.05)
        vertices_right = VGroup(
            *[dot.copy() for dot in markov_ch_mob.vertices.values()]
        ).arrange(RIGHT)

        matrix = (
            VGroup(p_equals, vertices_down, trans_matrix_mob)
            .arrange(RIGHT, buff=0.1)
            .scale(1.5)
            .to_edge(LEFT, buff=-0.5)
            .shift(DOWN * 0.6)
        )

        prob_labels = markov_ch_mob.get_transition_labels(scale=0.2)

        ################# ANIMATIONS #################

        self.play(Write(markov_ch_mob), run_time=1)

        self.wait()

        self.play(self.focus_on(markov_ch_mob, buff=2.8).shift(LEFT * 2.5))

        self.play(
            Indicate(markov_ch_mob.vertices[0]),
        )

        self.play(
            markov_ch_mob.vertices[1].animate.set_opacity(0.3),
            markov_ch_mob.edges[(2, 1)].animate.set_opacity(0.3),
            markov_ch_mob.edges[(4, 1)].animate.set_opacity(0.3),
            markov_ch_mob.edges[(3, 4)].animate.set_opacity(0.3),
            markov_ch_mob.edges[(2, 3)].animate.set_opacity(0.3),
            markov_ch_mob.edges[(0, 3)].animate.set_opacity(0.3),
            markov_ch_mob.edges[(0, 2)].animate.set_opacity(0.3),
        )

        self.wait()

        self.play(FadeIn(annotation), run_time=0.6)
        self.play(FadeIn(dot_product))

        self.wait()

        self.play(
            FadeOut(annotation),
            FadeOut(dot_product),
        )
        self.play(
            markov_ch_mob.vertices[1].animate.set_stroke(opacity=1),
            markov_ch_mob.vertices[1].animate.set_opacity(0.5),
            markov_ch_mob._labels[1].animate.set_opacity(1),
            markov_ch_mob.edges[(2, 1)].animate.set_opacity(1),
            markov_ch_mob.edges[(4, 1)].animate.set_opacity(1),
            markov_ch_mob.edges[(3, 4)].animate.set_opacity(1),
            markov_ch_mob.edges[(2, 3)].animate.set_opacity(1),
            markov_ch_mob.edges[(0, 3)].animate.set_opacity(1),
            markov_ch_mob.edges[(0, 2)].animate.set_opacity(1),
        )

        # now lets simulate the chain and show the transition matrix
        self.play(*[FadeIn(user) for user in users])
        self.wait()

        steps = 6
        trans_matrix = markov_ch.get_transition_matrix()
        new_matrix = trans_matrix.copy()

        print(markov_ch_mob.vertices)

        for n in range(steps):
            if n == 0:
                self.play(FadeIn(matrix), FadeIn(prob_labels))
                self.wait()
            else:
                new_matrix = new_matrix @ trans_matrix
                new_transition_matrix = (
                    self.matrix_to_mob(new_matrix)
                    .scale_to_fit_width(trans_matrix_mob.width)
                    .move_to(trans_matrix_mob)
                )

                transition_map = markov_ch_sim.get_lagged_smooth_transition_animations()

                new_iteration_text = (
                    MarkupText(f"P<sub>{n}</sub> = ", font=REDUCIBLE_FONT, weight=BOLD)
                    .scale_to_fit_width(p_equals.width)
                    .move_to(p_equals, LEFT)
                )

                # new_matrix_vg = (
                #     VGroup(new_iteration_text, vertices_down, new_transition_matrix)
                #     .arrange(RIGHT, buff=0.1)
                #     .to_edge(LEFT, buff=-0.5)
                #     .shift(DOWN * 0.3)
                # )

                self.play(
                    *[LaggedStart(*transition_map[i]) for i in markov_ch.get_states()],
                    # Transform(matrix, new_matrix_vg),
                    Transform(p_equals, new_iteration_text),
                    Transform(trans_matrix_mob, new_transition_matrix),
                    run_time=1,
                )
                self.wait()

    def focus_on(self, mobject, buff=2):
        return self.camera.frame.animate.set_width(mobject.width * buff).move_to(
            mobject
        )

    def matrix_to_mob(self, matrix: np.ndarray):
        str_repr = [[f"{a:.2f}" for a in row] for row in matrix]
        return Matrix(
            str_repr,
            left_bracket="(",
            right_bracket=")",
            element_to_mobject=Text,
            element_to_mobject_config={"font": REDUCIBLE_MONO},
            h_buff=2.3,
            v_buff=1.3,
        ).scale(0.2)


class DefineStationaryDist(Scene):
    def construct(self):
        pass
