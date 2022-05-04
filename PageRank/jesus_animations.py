import sys

from numpy import sqrt


sys.path.insert(1, "common/")

from manim import *

from markov_chain import *
from reducible_colors import *


class TransitionMatrix(MovingCameraScene):
    def construct(self):
        frame = self.camera.frame
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
                r"\pi_{n+1}(0) &= \pi_{n}(3) \cdot P(3, 0)\\ &+ \pi_{n}(2) \cdot P(2, 0)\\ &+ \pi_{n}(4) \cdot P(4, 0)"
            )
            .scale(0.7)
            .shift(LEFT * 5.5 + DOWN * 1.5)
        )
        # annotation = (
        #     Text(
        #         "Probability of ending in state 0 in the next step:",
        #         font=REDUCIBLE_FONT,
        #         weight=BOLD,
        #     )
        #     .scale_to_fit_width(dot_product.width)
        #     .next_to(dot_product, UP, buff=0.15)
        # )

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

        # isolate node 0
        mask_0 = (
            Difference(
                Rectangle(height=10, width=20),
                markov_ch_mob.vertices[0].copy().scale(1.05),
            )
            .set_color(BLACK)
            .set_stroke(width=0)
            .set_opacity(0.7)
        )
        self.play(FadeIn(mask_0))
        self.wait()
        self.play(FadeOut(mask_0))

        mask_but_0 = Rectangle(width=20, height=20)
        # only way to create a mask of several mobjects is to
        # keep poking the holes on the mask one by one
        for v in list(markov_ch_mob.vertices.values())[1:]:
            mask_but_0 = Difference(mask_but_0, v.copy().scale(1.05))

        mask_but_0.set_color(BLACK).set_stroke(width=0).set_opacity(0.7)

        self.play(FadeIn(mask_but_0))
        self.wait()
        self.play(FadeOut(mask_but_0))

        self.play(
            markov_ch_mob.vertices[1].animate.set_opacity(0.3),
            markov_ch_mob.edges[(2, 1)].animate.set_opacity(0.3),
            markov_ch_mob.edges[(1, 2)].animate.set_opacity(0.3),
            markov_ch_mob.edges[(4, 1)].animate.set_opacity(0.3),
            markov_ch_mob.edges[(3, 4)].animate.set_opacity(0.3),
            markov_ch_mob.edges[(2, 3)].animate.set_opacity(0.3),
            markov_ch_mob.edges[(0, 3)].animate.set_opacity(0.3),
            markov_ch_mob.edges[(0, 2)].animate.set_opacity(0.3),
        )

        self.wait()

        pi_dists = []
        for s in markov_ch.get_states():
            state = markov_ch_mob.vertices[s]
            label_direction = normalize(state.get_center() - markov_ch_mob.get_center())
            pi_dists.append(
                MathTex(f"\pi({s})")
                .scale(0.6)
                .next_to(state, label_direction, buff=0.1)
            )

        pi_dists_vg = VGroup(*pi_dists)

        self.play(FadeIn(pi_dists_vg))

        labels = markov_ch_mob.get_transition_labels(scale=0.2)
        self.play(FadeIn(VGroup(*labels)))

        pi_next_0 = MathTex("\pi_{n+1}(0)").scale(0.8)

        math_str = [
            "\pi_{n}" + f"({i})" + f"&\cdot P({i},0)"
            for i in range(len(markov_ch.get_states()))
        ]

        dot_prod_mob = MathTex("\\\\".join(math_str)).scale(0.6)

        brace = Brace(dot_prod_mob, LEFT)
        equation_explanation = (
            VGroup(pi_next_0, brace, dot_prod_mob)
            .arrange(RIGHT, buff=0.1)
            .move_to(frame.get_left(), aligned_edge=LEFT)
            .shift(RIGHT * 0.5)
        )

        self.play(FadeIn(pi_next_0))
        self.wait()
        # self.add(index_labels(dot_prod_mob[0]).set_opacity(0.8))
        self.play(
            FadeIn(brace),
            FadeIn(dot_prod_mob[0][0:5]),
            FadeIn(dot_prod_mob[0][12:17]),
            FadeIn(dot_prod_mob[0][24:29]),
            FadeIn(dot_prod_mob[0][36:41]),
            FadeIn(dot_prod_mob[0][48:53]),
        )
        self.wait()
        self.play(
            FadeIn(dot_prod_mob[0][5:12]),
            FadeIn(dot_prod_mob[0][17:24]),
            FadeIn(dot_prod_mob[0][29:36]),
            FadeIn(dot_prod_mob[0][41:48]),
            FadeIn(dot_prod_mob[0][53:]),
        )
        self.wait()

        self.play(
            markov_ch_mob.vertices[1].animate.set_stroke(opacity=1),
            markov_ch_mob.vertices[1].animate.set_opacity(0.5),
            markov_ch_mob._labels[1].animate.set_opacity(1),
            markov_ch_mob.edges[(2, 1)].animate.set_opacity(1),
            markov_ch_mob.edges[(1, 2)].animate.set_opacity(1),
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

        self.play(
            *[FadeIn(l) for l in count_labels.values()],
            *[FadeIn(u) for u in users],
        )

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
