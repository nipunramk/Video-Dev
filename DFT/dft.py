import sys

### THIS IS A WORKAROUND FOR NOW
### IT REQUIRES RUNNING MANIM FROM INSIDE DIRECTORY VIDEO-DEV
sys.path.insert(1, "common/")

from dft_utils import *

DEFAULT_SCALE = 0.8
DEFAULT_FREQ = 2


class AnalysisFrequencies(Scene):
    def construct(self):
        time_signal_func = get_cosine_func(freq=DEFAULT_FREQ)

        time_domain_graph = self.show_signal(time_signal_func)

        self.play(time_domain_graph.animate.scale(DEFAULT_SCALE).shift(LEFT * 3.5))
        self.wait()

        analysis_freq_func = get_cosine_func(freq=DEFAULT_FREQ)

        analysis_freq_graph = (
            self.show_signal(analysis_freq_func, color=FREQ_DOMAIN_COLOR)
            .shift(RIGHT * 3.5)
            .scale(DEFAULT_SCALE)
        )

        (
            time_axis,
            sampled_points_vert_lines,
            graph,
            sampled_points_dots,
        ) = analysis_freq_graph

        self.play(Write(time_axis), Write(graph))

        self.wait()

        self.play(Write(sampled_points_vert_lines))
        self.wait()

        self.play(FadeIn(sampled_points_dots))
        self.wait()
        self.remove(time_axis, sampled_points_vert_lines, graph, sampled_points_dots)

        (
            changing_analysis_frequency,
            time_freq,
            analysis_freq,
        ) = self.show_changing_analysis_frequency(
            time_domain_graph, analysis_freq_graph
        )

        f_def = (
            Tex(r"$f$ - original signal frequency")
            .scale(DEFAULT_SCALE)
            .next_to(time_domain_graph, DOWN)
        )
        f_hat_def = (
            Tex(r"$\hat{f}$ - analyzing frequency")
            .scale(DEFAULT_SCALE)
            .next_to(changing_analysis_frequency, DOWN)
        )

        self.play(FadeIn(f_def), FadeIn(f_hat_def))

        self.wait()

        self.play(
            time_domain_graph.animate.shift(UP * 1.5),
            changing_analysis_frequency.animate.shift(UP * 1.5),
            time_freq.animate.shift(UP * 1.5),
            analysis_freq.animate.shift(UP * 1.5),
            FadeOut(f_def),
            FadeOut(f_hat_def),
        )
        self.wait()

        signal_samples_vec = MathTex(
            r"\vec{y} = \begin{bmatrix} y_{0} & y_{1} & \cdots & y_{N-1} \end{bmatrix}"
        ).scale(0.8)
        analysis_samples_vec = MathTex(
            r"\vec{a} = \begin{bmatrix} a_{0} & a_{1} & \cdots & a_{N - 1} \end{bmatrix}"
        ).scale(0.8)

        sample_value = MathTex(r"y_k = \cos \left( \frac{2 \pi k f}{N} \right)").scale(
            0.8
        )
        analysis_value = MathTex(
            r"y_k = \cos \left( \frac{2 \pi k \hat{f}}{N} \right)"
        ).scale(0.8)

        signal_samples_vec.next_to(time_domain_graph, DOWN)
        analysis_samples_vec.next_to(changing_analysis_frequency, DOWN)

        sample_value.next_to(signal_samples_vec, DOWN)
        analysis_value.next_to(analysis_samples_vec, DOWN)

        self.play(FadeIn(signal_samples_vec), FadeIn(analysis_samples_vec))
        self.wait()

        self.play(FadeIn(sample_value), FadeIn(analysis_value))
        self.wait()

        freq_equal_case = MathTex(
            r"f = \hat{f} \rightarrow \vec{y} \cdot \vec{a} \neq 0"
        ).scale(1)
        freq_nequal_case = MathTex(
            r"f \neq \hat{f} \rightarrow \vec{y} \cdot \vec{a} = 0"
        ).scale(1)

        cases = (
            VGroup(freq_equal_case, freq_nequal_case).arrange(DOWN).shift(DOWN * 2.7)
        )
        for case in cases:
            self.play(FadeIn(case))
            self.wait()

        self.play(
            FadeOut(cases),
            FadeOut(signal_samples_vec),
            FadeOut(analysis_samples_vec),
            FadeOut(sample_value),
            FadeOut(analysis_value),
        )

        self.change_analysis_freq_with_dot_prod(
            changing_analysis_frequency, analysis_freq[1]
        )

    def show_signal(self, time_signal_func, color=TIME_DOMAIN_COLOR):
        time_axis, graph = plot_time_domain(time_signal_func, t_max=2 * PI, color=color)
        sampled_points_dots = get_sampled_dots(graph, time_axis)
        sampled_points_vert_lines = get_vertical_dashed_lines_for_samples(
            graph, time_axis, color=color
        )
        return VGroup(time_axis, sampled_points_vert_lines, graph, sampled_points_dots)

    def show_changing_analysis_frequency(self, time_domain_graph, analysis_freq_signal):
        frequency_tracker = ValueTracker(2)

        time_freq = (
            VGroup(MathTex("f = "), DecimalNumber(2.00, num_decimal_places=2))
            .arrange(RIGHT)
            .scale(DEFAULT_SCALE)
        )
        time_freq.next_to(time_domain_graph, UP)

        analysis_freq_symbol = MathTex(r"\hat{f} = ")
        analysis_freq_value = DecimalNumber(2.00, num_decimal_places=2)
        analysis_freq_value.add_updater(
            lambda m: m.set_value(frequency_tracker.get_value())
        )
        analysis_freq = (
            VGroup(analysis_freq_symbol, analysis_freq_value)
            .arrange(RIGHT)
            .scale(DEFAULT_SCALE)
        )
        analysis_freq.next_to(analysis_freq_signal, UP)

        def get_signal_of_frequency():
            analysis_freq_func = get_cosine_func(freq=frequency_tracker.get_value())
            new_signal = self.show_signal(analysis_freq_func, color=FREQ_DOMAIN_COLOR)
            return new_signal.shift(RIGHT * 3.5).scale(DEFAULT_SCALE)

        changing_analysis_freq_signal = always_redraw(get_signal_of_frequency)
        self.add(changing_analysis_freq_signal)

        self.play(FadeIn(time_freq))
        self.wait()

        self.play(FadeIn(analysis_freq))
        self.wait()

        self.play(frequency_tracker.animate.set_value(3), rate_func=linear, run_time=2)
        self.play(frequency_tracker.animate.set_value(4), rate_func=linear, run_time=2)
        self.play(frequency_tracker.animate.set_value(5), rate_func=linear, run_time=2)
        self.play(frequency_tracker.animate.set_value(2), rate_func=linear, run_time=6)
        self.wait()

        changing_analysis_freq_signal.clear_updaters()
        return changing_analysis_freq_signal, time_freq, analysis_freq

    def change_analysis_freq_with_dot_prod(
        self, analysis_freq_signal, analysis_freq_value
    ):
        frequency_tracker = ValueTracker(2)
        analysis_freq_value.clear_updaters()

        analysis_freq_value.add_updater(
            lambda m: m.set_value(frequency_tracker.get_value())
        )

        def get_signal_of_frequency():
            analysis_freq_func = get_cosine_func(freq=frequency_tracker.get_value())
            new_signal = self.show_signal(analysis_freq_func, color=FREQ_DOMAIN_COLOR)
            return new_signal.shift(RIGHT * 3.5 + UP * 1.5).scale(DEFAULT_SCALE)

        def get_dot_prod_bar_graph():
            time_domain_func = get_cosine_func(freq=DEFAULT_FREQ)
            analysis_freq_func = get_cosine_func(freq=frequency_tracker.get_value())
            dot_prod = inner_prod(time_domain_func, analysis_freq_func)
            bar_chart = BarChart(
                values=[dot_prod],
                y_range=[-5, 5, 5],
                x_length=3,
                y_length=3,
                bar_width=0.4,
                bar_colors=[REDUCIBLE_GREEN_LIGHTER],
            ).shift(DOWN * 2)
            return bar_chart

        self.remove(analysis_freq_signal)
        changing_analysis_freq_signal = always_redraw(get_signal_of_frequency)
        self.add(changing_analysis_freq_signal)

        bar_chart = always_redraw(get_dot_prod_bar_graph)
        self.add(bar_chart)
        self.wait()

        self.play(frequency_tracker.animate.set_value(0), run_time=4, rate_func=linear)
        self.wait()

        self.play(frequency_tracker.animate.set_value(1), run_time=2, rate_func=linear)
        self.wait()
        self.play(frequency_tracker.animate.set_value(2), run_time=2, rate_func=linear)
        self.wait()
        self.play(frequency_tracker.animate.set_value(3), run_time=2, rate_func=linear)
        self.wait()
        self.play(frequency_tracker.animate.set_value(4), run_time=2, rate_func=linear)
        self.wait()
        self.play(frequency_tracker.animate.set_value(5), run_time=2, rate_func=linear)
        self.wait()
