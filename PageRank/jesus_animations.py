import sys

sys.path.insert(1, "common/")

from manim import *

from markov_chain import *
from reducible_colors import *
import numpy as np


class TransitionMatrix(MovingCameraScene):
    def construct(self):
        markov_ch = MarkovChain(
            5, edges=[(1, 0), (2, 0), (3, 0), (4, 0), (0, 2), (0, 3), (2, 1), (3, 4)]
        )

        markov_ch_mob = MarkovChainGraph(
            markov_ch,
            curved_edge_config={"radius": 1.5},
        )

        self.play(Write(markov_ch_mob))

        prob_labels = markov_ch_mob.get_transition_labels()

        # self.play(Write(prob_labels))

        self.wait()

        self.play(self.focus_on(markov_ch_mob, buff=2.5).shift(LEFT * 2))
        self.play(Indicate(markov_ch_mob.vertices[0]))

    def focus_on(self, mobject, buff=2):
        return self.camera.frame.animate.set_width(mobject.width * buff).move_to(
            mobject
        )


class DefineStationaryDist(Scene):
    def construct(self):
        pass
