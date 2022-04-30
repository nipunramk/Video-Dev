import sys

from networkx.drawing import layout

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
