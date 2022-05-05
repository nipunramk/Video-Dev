import sys
from typing import Iterable

from numpy import sqrt
from math import dist

sys.path.insert(1, "common/")


from manim import *

config["assets_dir"] = "assets"

from markov_chain import *
from reducible_colors import *


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

        ######### DEFINE STATIONARY DISTRIBUTON #########

        self.play(
            self.camera.frame.animate.scale(1.4).shift(LEFT * 1.7),
            FadeOut(matrix),
            *[FadeOut(user) for user in users],
            FadeOut(prob_labels),
        )
        self.wait()

        stationary_dist_annotation = (
            Text("A distribution is stationary if:", font=REDUCIBLE_FONT, weight=BOLD)
            .scale(0.65)
            .next_to(markov_ch_mob, LEFT, buff=4.5, aligned_edge=RIGHT)
        )
        stationary_dist_tex = (
            MathTex("\pi_{n+1} = \pi_{n} P")
            .scale_to_fit_width(stationary_dist_annotation.width)
            .next_to(stationary_dist_annotation, DOWN)
        )
        self.play(Write(stationary_dist_annotation), run_time=0.8)
        self.play(FadeIn(stationary_dist_tex))

        self.wait()

        count_labels = self.get_current_count_mobs(
            markov_chain_g=markov_ch_mob, markov_chain_sim=markov_ch_sim, use_dist=True
        )

        self.wait()

        self.play(
            *[FadeIn(l) for l in count_labels.values()],
            *[FadeIn(u) for u in users],
        )

        for i in range(2):
            transition_animations = markov_ch_sim.get_instant_transition_animations()
            count_labels, count_transforms = self.update_count_labels(
                count_labels, markov_ch_mob, markov_ch_sim, use_dist=True
            )
            run_time = 1 / sqrt(i + 1)
            self.play(*transition_animations + count_transforms, run_time=run_time)

        self.wait()

        self.play(
            FadeOut(stationary_dist_annotation),
            FadeOut(stationary_dist_tex),
            *[FadeOut(u) for u in users],
            *[FadeOut(l) for l in count_labels.values()],
        )

        ############ IMPORTANT QUESTIONS ############

        """
        it’s important to address the question of whether a unique distribution even exists for a Markov chain. 
        And, a critical point in our model is if any initial distribution eventually converges to the stationary
        distribution
        """

        question_1 = (
            Text(
                "→ Is there a stationary distribution?",
                font=REDUCIBLE_FONT,
                weight=BOLD,
            )
            .scale(0.6)
            .next_to(markov_ch_mob, LEFT, buff=2)
            .shift(UP * 2)
        )

        question_2 = (
            MarkupText(
                """
                → Does every initial distribution
                converge to the stationary one?
                """,
                font=REDUCIBLE_FONT,
                weight=BOLD,
            )
            .scale(0.6)
            .next_to(question_1, DOWN, buff=1, aligned_edge=LEFT)
        )

        self.play(Write(question_1))
        self.wait()
        self.play(Write(question_2))

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

    def get_current_count_mobs(self, markov_chain_g, markov_chain_sim, use_dist=False):
        vertex_mobs_map = markov_chain_g.vertices
        count_labels = {}
        for v in vertex_mobs_map:
            if not use_dist:
                state_counts = markov_chain_sim.get_state_counts()
                label = Text(str(state_counts[v]), font="SF Mono").scale(0.6)
            else:
                state_counts = markov_chain_sim.get_user_dist(round_val=True)
                label = Text("{0:.2f}".format(state_counts[v]), font="SF Mono").scale(
                    0.6
                )
            label_direction = normalize(
                vertex_mobs_map[v].get_center() - markov_chain_g.get_center()
            )
            label.next_to(vertex_mobs_map[v], label_direction)
            count_labels[v] = label

        return count_labels

    def update_count_labels(
        self, count_labels, markov_chain_g, markov_chain_sim, use_dist=False
    ):
        if count_labels is None:
            count_labels = self.get_current_count_mobs(
                markov_chain_g, markov_chain_sim, use_dist=use_dist
            )
            transforms = [Write(label) for label in count_labels.values()]

        else:
            new_count_labels = self.get_current_count_mobs(
                markov_chain_g, markov_chain_sim, use_dist=use_dist
            )
            transforms = [
                Transform(count_labels[v], new_count_labels[v]) for v in count_labels
            ]

        return count_labels, transforms


class BruteForceMethod(TransitionMatrix):
    def construct(self):

        frame = self.camera.frame
        markov_ch = MarkovChain(
            4,
            edges=[
                (2, 0),
                (2, 3),
                (0, 3),
                (3, 1),
                (2, 1),
                (1, 2),
            ],
            dist=[0.2, 0.5, 0.2, 0.1],
        )

        markov_ch_mob = MarkovChainGraph(
            markov_ch,
            curved_edge_config={"radius": 2, "tip_length": 0.1},
            straight_edge_config={"max_tip_length_to_length_ratio": 0.08},
            layout="circular",
        )

        markov_ch_sim = MarkovChainSimulator(markov_ch, markov_ch_mob, num_users=50)
        users = markov_ch_sim.get_users()

        count_labels = self.get_current_count_mobs(
            markov_chain_g=markov_ch_mob, markov_chain_sim=markov_ch_sim, use_dist=True
        )

        stationary_dist_tex = (
            MathTex("\pi_{n+1} = \pi_{n} P")
            .scale(1.3)
            .next_to(markov_ch_mob, RIGHT, buff=6, aligned_edge=LEFT)
            .shift(UP * 2)
        )
        ############### ANIMATIONS

        self.play(Write(markov_ch_mob))
        self.play(
            LaggedStart(*[FadeIn(u) for u in users]),
            LaggedStart(
                *[FadeIn(l) for l in count_labels.values()],
            ),
            run_time=0.5,
        )

        self.play(frame.animate.shift(RIGHT * 4 + UP * 0.5).scale(1.2))

        title = (
            Text("Brute Force Method", font=REDUCIBLE_FONT, weight=BOLD)
            .scale(1)
            .move_to(frame.get_top())
            .shift(DOWN * 0.9)
        )
        self.play(FadeIn(title))
        self.wait()

        self.play(Write(stationary_dist_tex[0][-1]))
        self.play(Write(stationary_dist_tex[0][5:7]))
        self.play(Write(stationary_dist_tex[0][:5]))

        last_dist = markov_ch_sim.get_user_dist().values()
        last_dist_mob = (
            self.vector_to_mob(last_dist)
            .scale_to_fit_width(stationary_dist_tex[0][5:7].width)
            .next_to(stationary_dist_tex[0][5:7], DOWN, buff=0.4)
        )
        self.play(FadeIn(last_dist_mob))
        self.wait()

        # first iteration
        transition_map = markov_ch_sim.get_lagged_smooth_transition_animations()
        count_labels, count_transforms = self.update_count_labels(
            count_labels, markov_ch_mob, markov_ch_sim, use_dist=True
        )

        current_dist = markov_ch_sim.get_user_dist().values()
        current_dist_mob = (
            self.vector_to_mob(current_dist)
            .scale_to_fit_width(last_dist_mob.width)
            .next_to(stationary_dist_tex[0][:4], DOWN, buff=0.4)
        )
        self.play(
            *[LaggedStart(*transition_map[i]) for i in markov_ch.get_states()],
            *count_transforms,
            FadeIn(current_dist_mob),
        )

        distance = dist(current_dist, last_dist)
        distance_definition = (
            MathTex(r"D(\pi_{n+1}, \pi_{n}) =  ||\pi_{n+1} - \pi_{n}||_2")
            .scale(0.7)
            .next_to(stationary_dist_tex, DOWN, buff=2.5, aligned_edge=LEFT)
        )
        distance_mob = (
            VGroup(
                MathTex("D(\pi_{" + str(1) + "}, \pi_{" + str(0) + "})"),
                MathTex("="),
                Text(f"{distance:.5f}", font=REDUCIBLE_MONO).scale(0.6),
            )
            .arrange(RIGHT, buff=0.2)
            .scale(0.7)
            .next_to(stationary_dist_tex, DOWN, buff=2.5, aligned_edge=LEFT)
        )

        tolerance = 0.001
        tolerance_mob = (
            Text(
                "Threshold = " + str(tolerance),
                font=REDUCIBLE_FONT,
                t2f={str(tolerance): REDUCIBLE_MONO},
            )
            .scale(0.4)
            .next_to(distance_mob, DOWN, buff=0.2, aligned_edge=LEFT)
        )

        self.play(FadeIn(distance_definition))
        self.wait()
        self.play(
            FadeOut(distance_definition, shift=UP * 0.3),
            FadeIn(distance_mob, shift=UP * 0.3),
        )
        self.wait()

        self.play(FadeIn(tolerance_mob, shift=UP * 0.3))

        tick = (
            SVGMobject("check.svg")
            .scale(0.1)
            .set_color(PURE_GREEN)
            .next_to(tolerance_mob, RIGHT, buff=0.3)
        )

        self.wait()
        ## start the loop
        for i in range(2, 100):
            transition_animations = markov_ch_sim.get_instant_transition_animations()

            count_labels, count_transforms = self.update_count_labels(
                count_labels, markov_ch_mob, markov_ch_sim, use_dist=True
            )

            last_dist = current_dist
            current_dist = markov_ch_sim.get_user_dist().values()

            distance = dist(current_dist, last_dist)

            i_str = str(i)
            i_minus_one_str = str(i - 1)
            new_distance_mob = (
                VGroup(
                    MathTex("D(\pi_{" + i_str + "}, \pi_{" + i_minus_one_str + "})"),
                    MathTex("="),
                    Text(f"{distance:.5f}", font=REDUCIBLE_MONO).scale(0.6),
                )
                .arrange(RIGHT, buff=0.2)
                .scale(0.7)
                .next_to(stationary_dist_tex, DOWN, buff=2.5, aligned_edge=LEFT)
            )

            run_time = 0.8 if i < 6 else 1 / i

            if i < 6:
                current_to_last_shift = current_dist_mob.animate.move_to(last_dist_mob)
                fade_last_dist = FadeOut(last_dist_mob)
                last_dist_mob = current_dist_mob

                current_dist_mob = (
                    self.vector_to_mob(current_dist)
                    .scale_to_fit_width(last_dist_mob.width)
                    .next_to(stationary_dist_tex[0][:4], DOWN, buff=0.4)
                )

                self.play(
                    *transition_animations + count_transforms,
                    current_to_last_shift,
                    fade_last_dist,
                    FadeIn(current_dist_mob),
                    FadeTransform(distance_mob, new_distance_mob),
                    run_time=run_time,
                )

                distance_mob = new_distance_mob
            else:

                self.remove(last_dist_mob)
                last_dist_mob = current_dist_mob.move_to(last_dist_mob)

                current_dist_mob = (
                    self.vector_to_mob(current_dist)
                    .scale_to_fit_width(last_dist_mob.width)
                    .next_to(stationary_dist_tex[0][:4], DOWN, buff=0.4)
                )

                self.add(current_dist_mob)

                self.play(
                    *transition_animations + count_transforms,
                    FadeTransform(distance_mob, new_distance_mob),
                    run_time=run_time,
                )
                distance_mob = new_distance_mob

            if distance <= tolerance:
                found_iteration = (
                    Text(
                        f"iteration: {str(i)}",
                        font=REDUCIBLE_FONT,
                        t2f={str(i): REDUCIBLE_MONO},
                    )
                    .scale(0.3)
                    .next_to(tick, RIGHT, buff=0.1)
                )
                self.play(
                    FadeIn(tick, shift=UP * 0.3),
                    FadeIn(found_iteration, shift=UP * 0.3),
                )

                # get out of the loop
                break

        self.wait()

        ### the final distribution is:

        self.play(
            FadeOut(distance_mob),
            FadeOut(tolerance_mob),
            FadeOut(found_iteration),
            FadeOut(tick),
            FadeOut(last_dist_mob),
            current_dist_mob.animate.next_to(stationary_dist_tex, DOWN, buff=1.5).scale(
                2
            ),
        )
        self.wait()
        vertices_down = (
            VGroup(*[dot.copy().scale(0.8) for dot in markov_ch_mob.vertices.values()])
            .arrange(DOWN, buff=0.3)
            .next_to(current_dist_mob.copy().shift(RIGHT * 0.25), LEFT, buff=0.2)
        )
        self.play(FadeIn(vertices_down), current_dist_mob.animate.shift(RIGHT * 0.25))

    def vector_to_mob(self, vector: Iterable):
        str_repr = np.array([f"{a:.2f}" for a in vector]).reshape(-1, 1)
        return Matrix(
            str_repr,
            left_bracket="[",
            right_bracket="]",
            element_to_mobject=Text,
            element_to_mobject_config={"font": REDUCIBLE_MONO},
            h_buff=2.3,
            v_buff=1.3,
        )
