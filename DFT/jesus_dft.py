import sys

### THIS IS A WORKAROUND FOR NOW
### IT REQUIRES RUNNING MANIM FROM INSIDE DIRECTORY VIDEO-DEV
sys.path.insert(1, "common/")

from manim import *
from dft_utils import *
from reducible_colors import *


class BeginIntroSampling_002(Scene):
    def construct(self):
        x_min = 0

        x_max = 2 * PI

        cosine_signal = get_cosine_func(freq=7)
        axes, signal_mob = plot_time_domain(cosine_signal, t_max=x_max - PI / 4)
        sampled_dots = get_sampled_dots(signal_mob, axes, x_max=x_max, num_points=7)
        self.play(Write(signal_mob))
        self.play(Write(sampled_dots))
