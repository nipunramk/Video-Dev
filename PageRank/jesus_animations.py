import sys

from networkx.drawing import layout
from networkx.generators.expanders import margulis_gabber_galil_graph

sys.path.insert(1, "common/")

from manim import *

from markov_chain import *
from reducible_colors import *
import numpy as np


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
                (4, 1),
                (2, 1),
                (0, 2),
                (3, 4),
            ],
        )

        markov_ch_mob = MarkovChainGraph(
            markov_ch,
            curved_edge_config={"radius": 2},
            layout_scale=2.6,
        )

        self.play(Write(markov_ch_mob))

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

    def focus_on(self, mobject, buff=2):
        return self.camera.frame.animate.set_width(mobject.width * buff).move_to(
            mobject
        )


class DefineStationaryDist(Scene):
    def construct(self):
        pass


class MarkovChainTester(Scene):
    def construct(self):
        markov_chain = MarkovChain(
            4,
            [(0, 1), (1, 0), (1, 2), (1, 3), (2, 3), (3, 1)],
        )
        print(markov_chain.get_states())
        print(markov_chain.get_edges())
        print(markov_chain.get_current_dist())
        print(markov_chain.get_adjacency_list())
        print(markov_chain.get_transition_matrix())

        markov_chain_g = MarkovChainGraph(
            markov_chain, enable_curved_double_arrows=True
        )
        markov_chain_t_labels = markov_chain_g.get_transition_labels()
        self.play(FadeIn(markov_chain_g), FadeIn(markov_chain_t_labels))
        self.wait()

        markov_chain_sim = MarkovChainSimulator(
            markov_chain, markov_chain_g, num_users=50
        )

        users = markov_chain_sim.get_users()

        self.play(*[FadeIn(user) for user in users])
        self.wait()

        num_steps = 5

        markov_chain_g.toggle_ranking_scale_mapping(enabled=True)

        for _ in range(num_steps):
            (
                transition_map,
                scaling_animations,
            ) = markov_chain_sim.get_lagged_smooth_transition_animations()

            self.remove(markov_chain_sim)
            self.play(
                *[LaggedStart(*transition_map[i]) for i in markov_chain.get_states()]
            )
            self.add(markov_chain_sim)
            self.play(*scaling_animations)
            self.wait()
