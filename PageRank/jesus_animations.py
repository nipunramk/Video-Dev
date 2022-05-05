import sys


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

        trans_matrix_mob = self.matrix_to_mob(markov_ch.get_transition_matrix())

        p_equals = (
            Text("P = ", font=REDUCIBLE_FONT, weight=BOLD)
            .scale(0.3)
            .next_to(trans_matrix_mob, LEFT)
        )

        vertices_down = VGroup(
            *[dot.copy().scale(0.4) for dot in markov_ch_mob.vertices.values()]
        ).arrange(DOWN, buff=0.05)

        matrix = VGroup(p_equals, vertices_down, trans_matrix_mob).arrange(
            RIGHT, buff=0.1
        )

        vertices_right = (
            VGroup(*[dot.copy().scale(0.4) for dot in markov_ch_mob.vertices.values()])
            .arrange(RIGHT, buff=0.27)
            .next_to(trans_matrix_mob, UP, buff=0.1)
        )

        prob_labels = markov_ch_mob.get_transition_labels(scale=0.3)

        ################# ANIMATIONS #################

        self.play(Write(markov_ch_mob), run_time=1)

        self.wait()

        self.play(self.focus_on(markov_ch_mob, buff=3.2).shift(LEFT * 3))

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
        plus_signs = (
            VGroup(*[Tex("+").scale(0.7) for _ in range(4)])
            .arrange(DOWN, buff=0.22)
            .next_to(dot_prod_mob, RIGHT, buff=0.1, aligned_edge=UP)
            .shift(DOWN * 0.02)
        )

        self.play(FadeIn(pi_next_0))

        self.wait()

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

        self.play(FadeIn(plus_signs))

        self.wait()

        full_equation = VGroup(equation_explanation, plus_signs)

        ##### camera pans down for explanation
        self.play(frame.animate.shift(DOWN * 7), run_time=1.5)

        self.wait()

        math_notation_title = (
            Text("Some bits of mathematical notation", font=REDUCIBLE_FONT, weight=BOLD)
            .scale(0.5)
            .move_to(frame.get_corner(UL), aligned_edge=UL)
            .shift(DR * 0.5)
        )
        self.play(FadeIn(math_notation_title, shift=UP * 0.3), FadeOut(full_equation))

        dist_definition = (
            MathTex(
                # r"\vec{\pi_n} = [\pi_n(0), \pi_n(1), \pi_n(2), \pi_n(3), \pi_n(4) ]",
                r"\vec{\pi_n} = \begin{bmatrix} \pi_n(0) & \pi_n(1) & \pi_n(2) & \pi_n(3) & \pi_n(4) \end{bmatrix}",
            )
            .scale(0.7)
            .move_to(frame.get_center())
        )

        self.play(FadeIn(dist_definition, shift=UP * 0.3))
        self.wait()

        self.play(dist_definition.animate.shift(UP * 2))
        self.wait()

        trans_column_def = (
            MathTex(
                r"\vec{P_{i,0}} = \begin{bmatrix} P(0,0) \\ P(1,0) \\ P(2,0) \\ P(3,0) \\ P(4,0) \end{bmatrix}"
            )
            .scale(0.8)
            .move_to(frame.get_center())
        )
        self.play(FadeIn(trans_column_def, shift=UP * 0.3))

        self.wait()
        self.play(
            trans_column_def.animate.scale(0.7).next_to(
                dist_definition, DOWN, buff=0.5, coor_mask=[0, 1, 0]
            )
        )
        self.wait()
        next_dist_def = (
            MathTex(r"\vec{\pi}_{n+1}(0) = \vec{\pi_n} \cdot \vec{P_{i,0}}}")
            .scale(1.4)
            .move_to(frame.get_center())
            .shift(DOWN * 1.8)
        )
        self.play(FadeIn(next_dist_def, shift=UP * 0.3))

        self.wait()
        #### camera frame comes back
        self.play(
            frame.animate.shift(UP * 7),
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
            run_time=1.5,
        )

        self.play(
            FadeOut(dist_definition),
            FadeOut(trans_column_def),
            FadeOut(next_dist_def),
            FadeOut(math_notation_title),
        )

        self.wait()

        matrix_complete = (
            VGroup(vertices_right, matrix)
            .scale(1.7)
            .move_to(frame.get_left(), aligned_edge=LEFT)
            .shift(RIGHT * 0.6 + UP * 0.4)
        )

        self.play(FadeIn(matrix_complete), FadeIn(prob_labels))
        self.wait()
        dot_product_def = (
            MathTex(r"\vec{\pi}_{n+1} &= \vec{\pi}_n \cdot P")
            .scale(1.3)
            .next_to(trans_matrix_mob, DOWN, buff=0.5)
        )
        self.play(FadeIn(dot_product_def, shift=UP * 0.3))
        self.wait()

        surr_rect = SurroundingRectangle(
            trans_matrix_mob[0][0 : len(markov_ch.get_states())], color=REDUCIBLE_YELLOW
        )

        not_relevant_labels_tuples = list(
            filter(lambda x: x[0] != 0, markov_ch_mob.labels.keys())
        )
        not_relevant_labels = [
            markov_ch_mob.labels[t] for t in not_relevant_labels_tuples
        ]
        not_relevant_arrows = [
            markov_ch_mob.edges[t] for t in not_relevant_labels_tuples
        ]

        self.play(
            Write(surr_rect),
            *[
                markov_ch_mob.labels[t].animate.set_opacity(0.4)
                for t in not_relevant_labels_tuples
            ],
        )

        for s in markov_ch.get_states()[1:]:
            self.play(
                *[l.animate.set_opacity(1) for l in not_relevant_labels],
                *[arr.animate.set_opacity(1) for arr in not_relevant_arrows],
            )

            not_relevant_labels_tuples = list(
                filter(lambda x: x[0] != s, markov_ch_mob.labels.keys())
            )
            not_relevant_labels = [
                markov_ch_mob.labels[t] for t in not_relevant_labels_tuples
            ]
            not_relevant_arrows = [
                markov_ch_mob.edges[t] for t in not_relevant_labels_tuples
            ]

            print(not_relevant_labels)

            self.play(
                surr_rect.animate.shift(DOWN * 0.44),
                *[l.animate.set_opacity(0.2) for l in not_relevant_labels],
                *[arr.animate.set_opacity(0.3) for arr in not_relevant_arrows],
            )

            self.wait()

        ######### DEFINE STATIONARY DISTRIBUTON #########

        self.play(
            self.camera.frame.animate.scale(1.3).shift(LEFT * 1.2),
            FadeOut(matrix),
            FadeOut(dot_product_def),
            FadeOut(vertices_right),
            FadeOut(prob_labels),
            FadeOut(pi_dists_vg),
        )
        self.wait()

        stationary_dist_annotation = (
            Text("A distribution is stationary if:", font=REDUCIBLE_FONT, weight=BOLD)
            .scale(0.65)
            .next_to(markov_ch_mob, LEFT, buff=4.5, aligned_edge=RIGHT)
            .shift(UP * 0.8)
        )
        stationary_dist_tex = (
            MathTex("\pi = \pi P")
            .scale_to_fit_width(stationary_dist_annotation.width - 2)
            .next_to(stationary_dist_annotation, DOWN, buff=0.8)
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

        # accelerate the simulation so
        # we only show the stationary distribution
        # [markov_ch_sim.transition() for _ in range(10)]
        for i in range(15):
            transition_map = markov_ch_sim.get_lagged_smooth_transition_animations()
            count_labels, count_transforms = self.update_count_labels(
                count_labels, markov_ch_mob, markov_ch_sim, use_dist=True
            )
            self.play(
                *[LaggedStart(*transition_map[i]) for i in markov_ch.get_states()]
                + count_transforms
            )

        self.wait()

        self.play(
            FadeOut(stationary_dist_annotation),
            FadeOut(stationary_dist_tex),
            *[FadeOut(u) for u in users],
            *[FadeOut(l) for l in count_labels.values()],
            self.camera.frame.animate.scale(0.9),
        )

        ############ IMPORTANT QUESTIONS ############

        """
        it's important to address the question of whether a unique distribution even exists for a Markov chain. 
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
            .next_to(markov_ch_mob, LEFT, buff=1.5)
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
            left_bracket="[",
            right_bracket="]",
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
