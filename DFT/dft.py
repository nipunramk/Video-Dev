import sys

### THIS IS A WORKAROUND FOR NOW
### IT REQUIRES RUNNING MANIM FROM INSIDE DIRECTORY VIDEO-DEV
sys.path.insert(1, "common/")

from dft_utils import *


class AnalysisFrequencies(Scene):
    def construct(self):
        time_signal_func = get_cosine_func(freq=2)

        time_domain_graph = self.show_signal(time_signal_func)

        self.play(time_domain_graph.animate.scale(0.7).shift(LEFT * 3.5))
        self.wait()

        analysis_freq_func = get_cosine_func(freq=2)

        analysis_freq_graph = self.show_signal(
            analysis_freq_func, color=FREQ_DOMAIN_COLOR
        ).shift(RIGHT * 3.5)
        self.play(FadeIn(analysis_freq_graph.scale(0.7)))
        self.wait()

    def show_signal(self, time_signal_func, color=TIME_DOMAIN_COLOR):
        time_axis, graph = plot_time_domain(time_signal_func, t_max=2 * PI, color=color)
        sampled_points_dots = get_sampled_dots(graph, time_axis)
        sampled_points_vert_lines = get_vertical_bars_for_samples(graph, time_axis)
        return VGroup(time_axis, graph, sampled_points_dots, sampled_points_vert_lines)
