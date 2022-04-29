import sys

sys.path.insert(1, "common/")

from manim import *

from markov_chain import *
from reducible_colors import *
import numpy as np


class TransitionMatrix(Scene):
    def construct(self):
        markov_ch = MarkovChain(3, edges=[(0, 1), (0, 2), (1, 2), (2, 1), (1, 0)])

        markov_ch_mob = MarkovChainGraph(
            markov_ch, curved_edge_config={"color": REDUCIBLE_YELLOW}
        ).shift(RIGHT * 3)

        self.play(Write(markov_ch_mob))

        prob_labels = markov_ch_mob.get_transition_labels()

        self.play(Write(prob_labels))

        self.wait()
