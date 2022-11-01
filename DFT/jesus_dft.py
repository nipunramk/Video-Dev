import sys

### THIS IS A WORKAROUND FOR NOW
### IT REQUIRES RUNNING MANIM FROM INSIDE DIRECTORY VIDEO-DEV
sys.path.insert(1, "common/")

from manim import *
from dft_utils import *
from reducible_colors import *


class BeginIntroSampling_002(MovingCameraScene):
    def construct(self):
        frame = self.camera.frame
        x_max = TAU
        frequency = 7
        num_points = 7

        cosine_signal = get_cosine_func(freq=frequency)
        axes, signal_mob = plot_time_domain(cosine_signal, t_max=x_max)
        sampled_dots = get_sampled_dots(
            signal_mob, axes, x_max=x_max, num_points=num_points
        ).set_color(REDUCIBLE_YELLOW)

        freq_txt = (
            Text("ƒ = " + str(frequency) + " Hz", font=REDUCIBLE_MONO)
            .next_to(signal_mob, DOWN, aligned_edge=LEFT, buff=1)
            .scale(0.6)
        )
        point_n_txt = (
            Text("N = " + str(num_points), font=REDUCIBLE_MONO)
            .next_to(signal_mob, DOWN, aligned_edge=RIGHT, buff=1)
            .scale(0.6)
        )

        self.play(Write(signal_mob), run_time=2.5)

        self.wait()

        self.play(FadeIn(freq_txt))

        self.wait()

        self.play(Write(sampled_dots))

        self.wait()

        # aliasing
        # show how any multiple of our original freq is an alias
        multiples = 5
        original_signal_mob = signal_mob.copy()
        original_freq_txt = freq_txt.copy()
        aliasing_txt = (
            Text("Aliasing", font=REDUCIBLE_FONT, weight=BOLD)
            .set_stroke(width=8, background=True)
            .to_corner(UR)
        )

        axes, constant_signal_mob = plot_time_domain(get_cosine_func(0), t_max=x_max)

        constant_signal_mob = DashedVMobject(constant_signal_mob).move_to(
            sampled_dots[0], coor_mask=[0, 1, 0]
        )

        self.play(
            signal_mob.animate.set_stroke(opacity=0.25).set_fill(opacity=0),
            FadeIn(constant_signal_mob),
        )
        self.wait()
        self.play(
            FadeOut(constant_signal_mob), signal_mob.animate.set_stroke(opacity=1)
        )

        self.play(FadeIn(aliasing_txt, shift=LEFT))
        for m in range(2, multiples):
            cosine_signal_aa = get_cosine_func(freq=frequency * m)
            axes_aa, signal_mob_aa = plot_time_domain(cosine_signal_aa, t_max=x_max)
            new_freq_txt = (
                Text(
                    "ƒ = " + str(frequency * m) + " Hz",
                    font=REDUCIBLE_MONO,
                )
                .scale(0.6)
                .move_to(freq_txt)
            )

            self.play(
                Transform(signal_mob, signal_mob_aa),
                FadeTransform(freq_txt, new_freq_txt),
                run_time=2,
            )
            self.wait(0.5)
            freq_txt = new_freq_txt

        self.play(FadeOut(aliasing_txt, shift=RIGHT))
        self.wait()
        # reset
        self.play(
            Transform(signal_mob, original_signal_mob),
            FadeTransform(freq_txt, original_freq_txt),
        )
        freq_txt = original_freq_txt
        self.wait()

        # double sampling
        double_sampling = get_sampled_dots(
            signal_mob, axes, x_max=x_max, num_points=frequency * 2
        ).set_color(REDUCIBLE_YELLOW)

        point_n_txt = (
            Text("N = " + str(num_points), font=REDUCIBLE_MONO)
            .next_to(signal_mob, DOWN, aligned_edge=RIGHT, buff=1)
            .scale(0.6)
        )
        point_2n_txt = (
            Text("N = " + str(num_points * 2), font=REDUCIBLE_MONO)
            .next_to(signal_mob, DOWN, aligned_edge=RIGHT, buff=1)
            .scale(0.6)
        )

        # sampling at N = 14
        self.play(FadeIn(point_n_txt, shift=LEFT))
        self.wait()
        self.play(
            Transform(sampled_dots, double_sampling),
            FadeTransform(point_n_txt, point_2n_txt),
        )
        point_n_txt = point_2n_txt
        self.wait()

        # now show how this could be a constant signal
        # shift every point π radians
        period_length = TAU / frequency
        period_quarter = period_length / 4

        double_sampling_offset = get_sampled_dots(
            signal_mob,
            axes,
            x_min=period_quarter,
            x_max=x_max + period_quarter,
            num_points=frequency * 2,
        ).set_color(REDUCIBLE_YELLOW)

        _, constant_sine = plot_time_domain(get_sine_func(0), t_max=x_max)
        c_sine_mob = (
            DashedVMobject(constant_sine)
            .move_to(double_sampling_offset, coor_mask=[0, 1, 0])
            .set_color(REDUCIBLE_YELLOW)
            .set_stroke(opacity=0.5)
        )

        self.play(Transform(sampled_dots, double_sampling_offset), run_time=3)
        self.wait()

        self.play(signal_mob.animate.set_stroke(opacity=0.3), FadeIn(c_sine_mob))
        self.wait()

        self.play(FadeOut(c_sine_mob), signal_mob.animate.set_stroke(opacity=1))
        self.wait()

        # shannon sampling
        shannon_sampling = get_sampled_dots(
            signal_mob, axes, x_max=x_max, num_points=frequency * 2 + 1
        ).set_color(REDUCIBLE_YELLOW)
        point_shannon_txt = (
            Text("N = " + str(num_points * 2 + 1), font=REDUCIBLE_MONO)
            .next_to(signal_mob, DOWN, aligned_edge=RIGHT, buff=1)
            .scale(0.6)
        )
        self.play(
            Transform(sampled_dots, shannon_sampling),
            FadeTransform(point_n_txt, point_shannon_txt),
        )
        point_n_txt = point_shannon_txt

        shannon_text = Text(
            "Shannon-Nyquist Theorem", font=REDUCIBLE_FONT, weight=BOLD
        ).next_to(signal_mob, UP, buff=2)

        self.wait()

        self.play(
            FadeIn(shannon_text, shift=DOWN),
            frame.animate.shift(UP),
            FadeOut(freq_txt, point_n_txt),
            signal_mob.animate.shift(DOWN * 0.8),
            sampled_dots.animate.shift(DOWN * 0.8),
        )
        axes.shift(DOWN * 0.8)

        shannon_theorem = MathTex(
            r"f_{s} \Rightarrow f_{\text{\tiny sample rate}} > 2 \cdot f_{s}"
        ).next_to(shannon_text, DOWN, buff=0.5)
        shannon_theorem_reverse = MathTex(
            r"f_{\text{\tiny sample rate}} \Rightarrow f_{s_{+}} < \frac{f_{\text{\tiny sample rate}}}{2}"
        ).next_to(shannon_theorem, DOWN, buff=0.5)

        self.wait()

        self.play(FadeIn(shannon_theorem))

        self.wait()

        self.play(FadeIn(shannon_theorem_reverse))

        self.wait()


class IntroTimeFreqDomain(MovingCameraScene):
    def construct(self):

        frame = self.camera.frame
        x_max = TAU * 2
        frequency = 7
        num_points = 7

        freq_1 = 2
        freq_2 = 5
        freq_3 = 10

        cos_1 = get_cosine_func(freq=freq_1, amplitude=0.3)
        cos_2 = get_cosine_func(freq=freq_2, amplitude=0.2)
        cos_3 = get_cosine_func(freq=freq_3, amplitude=0.1)
        sum_function = get_sum_functions(cos_1, cos_2, cos_3)

        axes_sum, sum_mob = plot_time_domain(
            sum_function,
            t_max=x_max,
        )

        self.play(Write(sum_mob), run_time=2)

        # this would be the "frequency representation"
        sum_dft_graph = (
            get_fourier_bar_chart(sum_function, height_scale=14.8, n_samples=100)
            .scale(0.4)
            .next_to(sum_mob, DOWN, buff=1)
        )

        cos_1_axes, cos_1_mob = plot_time_domain(
            cos_1,
            t_max=x_max,
        )
        _, cos_2_mob = plot_time_domain(
            cos_2,
            t_max=x_max,
        )
        _, cos_3_mob = plot_time_domain(
            cos_3,
            t_max=x_max,
        )

        self.play(
            sum_mob.animate.shift(UP * 0.5), LaggedStartMap(FadeIn, sum_dft_graph)
        )
        cos_1_dft_graph = (
            get_fourier_bar_chart(cos_1, height_scale=14.8, n_samples=100)
            .scale(0.4)
            .move_to(sum_dft_graph, aligned_edge=DOWN)
        )
        cos_2_dft_graph = (
            get_fourier_bar_chart(cos_2, height_scale=14.8, n_samples=100)
            .scale(0.4)
            .move_to(sum_dft_graph, aligned_edge=DOWN)
        )
        cos_3_dft_graph = (
            get_fourier_bar_chart(cos_3, height_scale=14.8, n_samples=100)
            .scale(0.4)
            .move_to(sum_dft_graph, aligned_edge=DOWN)
        )

        decomposed_sum = (
            VGroup(cos_1_mob, cos_2_mob, cos_3_mob)
            .arrange(DOWN, buff=0.5)
            .scale(0.8)
            .move_to(sum_mob)
        )

        text_cos_1 = (
            Text(f"{freq_1} Hz", font=REDUCIBLE_MONO)
            .next_to(cos_1_mob, LEFT, buff=0, aligned_edge=DOWN)
            .scale(0.5)
        )
        text_cos_2 = (
            Text(f"{freq_2} Hz", font=REDUCIBLE_MONO)
            .next_to(cos_2_mob, LEFT, buff=0, aligned_edge=DOWN)
            .scale(0.5)
        )
        text_cos_3 = (
            Text(f"{freq_3} Hz", font=REDUCIBLE_MONO)
            .next_to(cos_3_mob, LEFT, buff=0, aligned_edge=DOWN)
            .scale(0.5)
        )

        self.wait()

        sum_mob_og = sum_mob.copy()
        sum_dft_graph_og = sum_dft_graph.copy()

        self.play(
            Transform(sum_mob, decomposed_sum),
            FadeIn(text_cos_1, text_cos_2, text_cos_3),
        )
        self.wait()
        self.play(FadeOut(text_cos_1, text_cos_2, text_cos_3, shift=UP * 0.3))

        self.wait()

        self.play(
            sum_mob[1].animate.set_stroke(opacity=0.3),
            sum_mob[2].animate.set_stroke(opacity=0.3),
            Transform(sum_dft_graph, cos_1_dft_graph),
        )

        self.wait()

        self.play(
            sum_mob[0].animate.set_stroke(opacity=0.3),
            sum_mob[1].animate.set_stroke(opacity=1),
            Transform(sum_dft_graph, cos_2_dft_graph),
        )

        self.wait()

        self.play(
            sum_mob[0].animate.set_stroke(opacity=0.3),
            sum_mob[1].animate.set_stroke(opacity=0.3),
            sum_mob[2].animate.set_stroke(opacity=1),
            Transform(sum_dft_graph, cos_3_dft_graph),
        )

        self.wait()

        self.play(
            sum_mob[0].animate.set_stroke(opacity=1),
            sum_mob[1].animate.set_stroke(opacity=1),
            sum_mob[2].animate.set_stroke(opacity=1),
            Transform(sum_dft_graph, sum_dft_graph_og),
        )

        self.wait()

        self.play(
            sum_mob[0].animate.stretch_to_fit_height(0.3),
            sum_mob[1].animate.stretch_to_fit_height(0.8),
            sum_mob[2].animate.stretch_to_fit_height(0.1),
        )
        self.wait()

        self.play(Transform(sum_mob, sum_mob_og))

        self.wait()

        freq_repr_txt = (
            Text("Frequency Domain Representation", font=REDUCIBLE_FONT, weight=BOLD)
            .scale(0.9)
            .next_to(sum_mob_og, UP * 0.7, buff=2)
        )
        self.play(
            frame.animate.shift(UP * 0.6), FadeIn(freq_repr_txt, shift=DOWN * 0.3)
        )
