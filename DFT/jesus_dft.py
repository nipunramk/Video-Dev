import sys

### THIS IS A WORKAROUND FOR NOW
### IT REQUIRES RUNNING MANIM FROM INSIDE DIRECTORY VIDEO-DEV
sys.path.insert(1, "common/")

from manim import *

from dft_utils import *
from reducible_colors import *
from math import degrees
from classes import CustomLabel


class IntroSampling_002(MovingCameraScene):
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

        freq_1 = 2
        freq_2 = 5
        freq_3 = 10

        cos_1 = get_cosine_func(freq=freq_1, amplitude=0.3)
        cos_2 = get_cosine_func(freq=freq_2, amplitude=0.3)
        cos_3 = get_cosine_func(freq=freq_3, amplitude=0.3)
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

        _, cos_1_mob = plot_time_domain(
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
            sum_dft_graph[3]
            .animate.stretch_to_fit_height(sum_dft_graph[3].height - 0.3)
            .move_to(sum_dft_graph[3], aligned_edge=DOWN),
            sum_dft_graph[8]
            .animate.stretch_to_fit_height(sum_dft_graph[3].height - 0.1)
            .move_to(sum_dft_graph[8], aligned_edge=DOWN),
            sum_dft_graph[16]
            .animate.stretch_to_fit_height(sum_dft_graph[3].height - 0.6)
            .move_to(sum_dft_graph[16], aligned_edge=DOWN),
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


class IntroSimilarityConcept(MovingCameraScene):
    def construct(self):

        original_freq_mob = self.show_similarity_operation()
        self.show_point_sequence(original_freq_mob)
        self.show_signal_cosines()

    def show_similarity_operation(self):

        frame = self.camera.frame
        t_max = TAU * 2

        original_freq = 2

        cos_og = get_cosine_func(freq=original_freq, amplitude=0.3)

        _, original_freq_mob = plot_time_domain(cos_og, t_max=t_max)

        og_fourier = (
            get_fourier_bar_chart(cos_og, t_max=t_max, height_scale=14.8, n_samples=100)
            .scale_to_fit_width(original_freq_mob.width)
            .shift(DOWN * 2)
        )

        self.play(Write(original_freq_mob), run_time=1.5)
        self.wait()

        self.play(original_freq_mob.animate.shift(UP * 1.5))
        self.wait()

        self.play(LaggedStartMap(FadeIn, og_fourier))
        self.wait()

        self.play(frame.animate.scale(0.8).shift(UP * 0.2), FadeOut(og_fourier))
        self.wait()

        frequency_tracker = ValueTracker(3)

        def get_signal_of_frequency():
            analysis_freq_func = get_cosine_func(
                freq=frequency_tracker.get_value(), amplitude=0.3
            )
            _, new_signal = plot_time_domain(
                analysis_freq_func, t_max=t_max, color=FREQ_DOMAIN_COLOR
            )
            return new_signal.move_to(original_freq_mob)

        def get_dot_prod_bar():
            analysis_freq = get_cosine_func(freq=frequency_tracker.get_value())
            og_freq = get_cosine_func(freq=original_freq)
            dot_prod = abs(inner_prod(og_freq, analysis_freq))

            barchart = (
                BarChart(
                    values=[dot_prod],
                    y_range=[0, 4, 1],
                    x_length=1,
                    y_length=original_freq_mob.width,
                    bar_width=0.3,
                    bar_colors=[REDUCIBLE_GREEN_LIGHTER],
                )
                .rotate(PI / 2)
                .flip()
            )

            barchart[1:].set_opacity(0.0).next_to(
                original_freq_mob, DOWN, buff=2.4, aligned_edge=LEFT
            )

            barchart[0].next_to(original_freq_mob, DOWN, buff=2.4, aligned_edge=LEFT)

            return barchart

        changing_analysis_freq_signal = always_redraw(get_signal_of_frequency)
        changing_bar_dot_prod = always_redraw(get_dot_prod_bar)

        bg_rect = (
            Rectangle(height=0.3, width=original_freq_mob.width, color=REDUCIBLE_GREEN)
            .set_opacity(0.3)
            .set_stroke(opacity=0.3)
            .next_to(original_freq_mob, DOWN, buff=2.4)
        )
        v_similar_txt = (
            Text("Very Similar", font=REDUCIBLE_FONT)
            .scale(0.2)
            .next_to(bg_rect, DOWN, aligned_edge=RIGHT)
        )
        n_similar_txt = (
            Text("Very Different", font=REDUCIBLE_FONT)
            .scale(0.2)
            .next_to(bg_rect, DOWN, aligned_edge=LEFT)
        )

        self.play(Write(changing_analysis_freq_signal))
        self.play(FadeIn(bg_rect, changing_bar_dot_prod, shift=DOWN * 0.3))
        self.play(FadeIn(v_similar_txt, n_similar_txt, shift=DOWN * 0.3))
        self.wait()

        self.play(frequency_tracker.animate.set_value(3.5), run_time=4)
        self.play(frequency_tracker.animate.set_value(4), run_time=4, rate_func=linear)
        self.play(
            frequency_tracker.animate.set_value(3.6), run_time=4, rate_func=linear
        )
        self.play(
            frequency_tracker.animate.set_value(original_freq),
            run_time=6,
            rate_func=linear,
        )
        self.wait()
        self.play(
            FadeOut(changing_analysis_freq_signal),
            FadeOut(
                changing_bar_dot_prod,
                bg_rect,
                v_similar_txt,
                n_similar_txt,
                shift=DOWN * 0.3,
            ),
        )

        return original_freq_mob

    def show_point_sequence(self, original_freq_mob):

        self.play(original_freq_mob.animate.move_to(ORIGIN))

        dots = VGroup(
            *[
                Dot(color=WHITE).move_to(original_freq_mob.point_from_proportion(d))
                for d in np.linspace(0, 1, 3)
            ]
        )

        for i in range(6, 3 * 10, 3):
            dots_pos = np.linspace(0, 1, i)
            new_dots = VGroup(
                *[
                    Dot(color=WHITE).move_to(original_freq_mob.point_from_proportion(d))
                    for d in dots_pos
                ]
            )
            self.play(Transform(dots, new_dots), run_time=4 / i)
            self.wait(1 / config.frame_rate)

        self.wait()

        self.play(
            dots.animate.scale(0.7).move_to(ORIGIN).arrange(RIGHT),
            FadeOut(original_freq_mob),
        )

        left_bracket = (
            Tex("[")
            .scale_to_fit_height(dots[0].height * 3)
            .next_to(dots, LEFT, buff=0.2)
        )
        right_bracket = (
            Tex("]")
            .scale_to_fit_height(dots[0].height * 3)
            .next_to(dots, RIGHT, buff=0.2)
        )

        vector = VGroup(left_bracket, dots, right_bracket)

        self.play(FadeIn(left_bracket, right_bracket, shift=DOWN * 0.3))

        self.wait()
        vector_purple = vector.copy().set_color(REDUCIBLE_VIOLET).shift(DOWN * 1)

        times_symbol = MathTex(r"\times").scale(0.8)
        self.play(
            vector.animate.shift(UP * 1).set_color(REDUCIBLE_YELLOW),
            FadeIn(vector_purple, times_symbol),
        )

        self.wait()
        self.play(*[FadeOut(mob) for mob in self.mobjects])

    def show_signal_cosines(self):
        t_max = TAU * 2

        original_freq = 2

        cos_og = get_cosine_func(freq=original_freq, amplitude=1)

        _, original_freq_mob = plot_time_domain(cos_og, t_max=t_max)
        original_freq_mob.set_color(REDUCIBLE_YELLOW).set_stroke(width=12)

        analysis_freqs = VGroup()
        for i in range(1, 9):
            _, af = plot_time_domain(
                get_cosine_func(freq=i, amplitude=0.3), t_max=t_max
            )
            af.set_stroke(opacity=1 / i)
            analysis_freqs.add(af)

        analysis_freqs.arrange(DOWN).scale(0.6).set_color(REDUCIBLE_VIOLET)

        original_freq_mob.shift(LEFT * 6)
        analysis_freqs.shift(RIGHT * 5)

        times = MathTex(r"\times").scale(2)

        self.play(Write(original_freq_mob))
        self.play(FadeIn(times))
        self.play(LaggedStartMap(FadeIn, analysis_freqs), run_time=2)


class IntroducePhaseProblem(MovingCameraScene):
    def construct(self):

        frame = self.camera.frame.save_state()
        self.try_sine_wave()

        self.play(*[FadeOut(mob) for mob in self.mobjects])
        self.play(Restore(frame))

        self.test_cases_again()

    def try_sine_wave(self):
        frame = self.camera.frame
        t_min = 0
        t_max = TAU

        original_freq = 2

        # samples per second
        sample_frequency = 40

        n_samples = sample_frequency

        vt_frequency = ValueTracker(original_freq)
        vt_phase = ValueTracker(0)
        vt_amplitude = ValueTracker(1)
        vt_b = ValueTracker(0)

        def sine_cosine_redraw():
            phase_ch_cos = get_cosine_func(
                amplitude=vt_amplitude.get_value(),
                freq=vt_frequency.get_value(),
                phase=vt_phase.get_value(),
                b=vt_b.get_value(),
            )
            _, phase_ch_cos_mob = plot_time_domain(phase_ch_cos, t_max=t_max)

            return phase_ch_cos_mob.scale(0.6).shift(UP * 1.5)

        af_matrix = get_analysis_frequency_matrix(
            N=n_samples, sample_rate=sample_frequency, t_max=t_max
        )

        def updating_transform_redraw():
            signal_function = get_cosine_func(
                amplitude=vt_amplitude.get_value(),
                freq=vt_frequency.get_value(),
                phase=vt_phase.get_value(),
                b=vt_b.get_value(),
            )

            sampled_signal = np.array(
                [
                    signal_function(v)
                    for v in np.linspace(t_min, t_max, num=n_samples, endpoint=False)
                ]
            ).reshape(-1, 1)

            rects = get_rectangles_for_matrix_transform(sampled_signal, af_matrix)

            return rects

        changing_sine = always_redraw(sine_cosine_redraw)
        sampled_dots = VGroup(
            *[
                Dot()
                .move_to(changing_sine.point_from_proportion(p))
                .set_fill(REDUCIBLE_YELLOW, opacity=1)
                for p in np.linspace(0, 1, 10)
            ]
        )

        changing_rects = always_redraw(updating_transform_redraw)

        cos_tex = MathTex("cos(x)").scale(0.8).next_to(changing_sine, DOWN, buff=1)
        sin_tex = MathTex("sin(x)").scale(0.8).next_to(changing_sine, DOWN, buff=1)

        self.play(Write(changing_sine), FadeIn(cos_tex, changing_rects))

        self.wait()

        self.play(LaggedStartMap(FadeIn, sampled_dots))
        self.wait()
        self.play(FadeOut(sampled_dots))

        self.play(
            vt_phase.animate.set_value(PI / 2),
            FadeTransform(cos_tex, sin_tex),
            run_time=2,
        )

        cos_mob = changing_sine.copy()
        analysis_freq = get_cosine_func(freq=original_freq)
        _, analysis_freq_mob = plot_time_domain(analysis_freq, t_max=t_max)

        analysis_freq_mob.set_color(REDUCIBLE_VIOLET).scale_to_fit_width(
            changing_sine.width
        ).move_to(ORIGIN)

        self.play(FadeIn(cos_mob), FadeOut(changing_sine))
        self.play(
            FadeOut(changing_rects),
            FadeOut(sin_tex),
            cos_mob.animate.move_to(ORIGIN),
            Write(analysis_freq_mob),
            frame.animate.scale(0.8).shift(DOWN * 0.8),
        )
        self.wait()

        aux_analysis_freq_axis, _ = plot_time_domain(
            get_cosine_func(freq=original_freq), t_max=t_max
        )
        aux_analysis_freq_axis.scale_to_fit_width(analysis_freq_mob.width).move_to(
            analysis_freq_mob
        )

        aux_signal_axis, sine_wave = plot_time_domain(
            get_cosine_func(freq=original_freq, phase=vt_phase.get_value()), t_max=t_max
        )
        aux_signal_axis.scale_to_fit_width(cos_mob.width).move_to(cos_mob)

        dots_analysis_freq = get_sampled_dots(
            analysis_freq_mob, aux_analysis_freq_axis, num_points=10
        ).set_fill(REDUCIBLE_VIOLET)
        dots_cos_mob = get_sampled_dots(
            sine_wave, aux_signal_axis, num_points=10
        ).set_fill(REDUCIBLE_YELLOW)

        self.play(LaggedStartMap(FadeIn, [*dots_analysis_freq, *dots_cos_mob]))

        dots_line = dots_cos_mob.copy()
        [
            d.set_fill(REDUCIBLE_GREEN_LIGHTER, opacity=1).move_to(
                DOWN * 2.5, coor_mask=[0, 1, 0]
            )
            for d in dots_line
        ]

        self.play(
            Transform(dots_cos_mob, dots_line), Transform(dots_analysis_freq, dots_line)
        )
        self.wait()

        zero = (
            Text(str(0), font=REDUCIBLE_MONO, weight=SEMIBOLD)
            .scale(2)
            .move_to(dots_cos_mob)
        )

        self.play(Transform(dots_cos_mob, zero), FadeOut(dots_analysis_freq))
        self.wait()

    def test_cases_again(self):
        frame = self.camera.frame
        t_min = 0
        t_max = TAU * 2

        # samples per second
        sample_frequency = 40

        # total number of samples
        n_samples = sample_frequency

        analysis_frequencies = [
            sample_frequency * m / n_samples for m in range(n_samples // 2)
        ]

        # let's just take one AF as an example
        original_freq = analysis_frequencies[2]

        vt_frequency = ValueTracker(original_freq)
        # this tracker will move phase: from 0 to PI/2
        vt_phase = ValueTracker(0)
        vt_amplitude = ValueTracker(1)
        vt_b = ValueTracker(0)

        def change_text_redraw():
            v_freq = f"{vt_frequency.get_value():.2f}"
            v_amplitude = f"{vt_amplitude.get_value():.2f}"
            v_phase = f"{degrees(vt_phase.get_value()) % 360:.2f}"
            v_b = f"{vt_b.get_value():.2f}"

            tex_frequency = Text(
                "ƒ = " + v_freq + " Hz",
                font=REDUCIBLE_FONT,
                t2f={v_freq: REDUCIBLE_MONO},
            ).scale(0.8)

            phi_eq = MathTex(r"\phi = ")
            tex_phase_n = Text(
                v_phase + "º",
                font=REDUCIBLE_FONT,
                t2f={v_phase: REDUCIBLE_MONO},
            ).scale(0.8)
            tex_phase = VGroup(phi_eq, tex_phase_n).arrange(RIGHT)

            tex_amplitude = Text(
                "A = " + v_amplitude,
                font=REDUCIBLE_FONT,
                t2f={v_amplitude: REDUCIBLE_MONO},
            ).scale(0.8)

            tex_b = Text(
                "b = " + v_b,
                font=REDUCIBLE_FONT,
                t2f={v_b: REDUCIBLE_MONO},
            ).scale(0.8)

            text_group = (
                VGroup(tex_frequency, tex_phase, tex_amplitude, tex_b)
                .arrange(DOWN, aligned_edge=LEFT)
                .scale(0.6)
                .to_corner(UL)
            )
            return text_group

        def change_phase_redraw():
            phase_ch_cos = get_cosine_func(
                amplitude=vt_amplitude.get_value(),
                freq=vt_frequency.get_value(),
                phase=vt_phase.get_value(),
                b=vt_b.get_value(),
            )
            _, phase_ch_cos_mob = plot_time_domain(phase_ch_cos, t_max=t_max)
            return phase_ch_cos_mob.scale(0.6).shift(UP)

        af_matrix = get_analysis_frequency_matrix(
            N=n_samples, sample_rate=sample_frequency, t_max=t_max
        )

        rect_scale = 0.1

        def updating_transform_redraw():
            signal_function = get_cosine_func(
                amplitude=vt_amplitude.get_value(),
                freq=vt_frequency.get_value(),
                phase=vt_phase.get_value(),
                b=vt_b.get_value(),
            )

            sampled_signal = np.array(
                [
                    signal_function(v)
                    for v in np.linspace(t_min, t_max, num=n_samples, endpoint=False)
                ]
            ).reshape(-1, 1)

            rects = get_rectangles_for_matrix_transform(sampled_signal, af_matrix)

            return rects

        changing_signal_mob = always_redraw(change_phase_redraw)
        freq_analysis = always_redraw(updating_transform_redraw)
        changing_tex_group = always_redraw(change_text_redraw)

        line_ref = DashedVMobject(
            Line(freq_analysis.get_left(), freq_analysis.get_right())
            .set_stroke(WHITE, opacity=0.5)
            .move_to(changing_signal_mob)
        )

        self.play(Write(changing_signal_mob), FadeIn(freq_analysis), Write(line_ref))
        self.play(FadeIn(changing_tex_group))

        self.play(vt_amplitude.animate.set_value(0.5), run_time=0.8)
        self.wait()
        self.play(vt_amplitude.animate.set_value(0.1), run_time=0.8)
        self.wait()
        self.play(vt_amplitude.animate.set_value(0), run_time=0.8)
        self.wait()
        self.play(vt_amplitude.animate.set_value(1), run_time=0.8)
        self.wait()

        self.play(vt_frequency.animate.set_value(analysis_frequencies[3]))
        self.wait()
        self.play(vt_frequency.animate.set_value(analysis_frequencies[4]))
        self.wait()

        self.play(vt_b.animate.set_value(0.3))
        self.wait()
        self.play(vt_b.animate.set_value(0.5))
        self.wait()
        self.play(vt_b.animate.set_value(0))
        self.wait()

        for i in range(1, 5):
            self.play(vt_phase.animate.set_value(i * PI / 2))
            self.wait()

        self.play(
            vt_phase.animate.set_value(20 * PI / 2), run_time=10, rate_func=linear
        )


class SolvingPhaseProblem(MovingCameraScene):
    def construct(self):
        # frame = self.camera.frame.save_state()

        # self.hacky_sine_waves()

        # self.play(*[FadeOut(mob) for mob in self.mobjects])
        # self.play(Restore(frame))

        self.capture_sine_and_cosine_transforms()

    def hacky_sine_waves(self):
        original_frequency = 4
        t_max = PI

        # samples per second
        sample_frequency = 40

        # total number of samples
        n_samples = sample_frequency

        sin_f = get_sine_func(freq=original_frequency)
        _, sin_mob = plot_time_domain(sin_f, t_max=t_max, color=REDUCIBLE_YELLOW)

        sin_t = (
            Text("sin(x)", font=REDUCIBLE_FONT, weight=BOLD)
            .next_to(sin_mob, DOWN, buff=0.5)
            .set_color(REDUCIBLE_YELLOW)
        )

        vt_phase = ValueTracker(0)
        vt_amplitude = ValueTracker(1)
        vt_frequency = ValueTracker(original_frequency)
        vt_b = ValueTracker(0)

        def change_af_cos_sin():
            # cos analysis frequency
            cos_af = get_cosine_func(
                freq=original_frequency, phase=vt_phase.get_value()
            )
            _, cos_af_mob = plot_time_domain(
                cos_af, t_max=t_max, color=REDUCIBLE_VIOLET
            )

            return cos_af_mob

        af_matrix = get_analysis_frequency_matrix(
            n_samples, sample_rate=sample_frequency, func="sin", t_max=t_max
        )

        def show_transform():
            signal_function = get_sine_func(
                amplitude=vt_amplitude.get_value(),
                freq=vt_frequency.get_value(),
                phase=vt_phase.get_value(),
                b=vt_b.get_value(),
            )

            sampled_signal = np.array(
                [
                    signal_function(v)
                    for v in np.linspace(0, t_max, num=n_samples, endpoint=False)
                ]
            ).reshape(-1, 1)

            rects = get_rectangles_for_matrix_transform(sampled_signal, af_matrix)
            return rects

        af_mob = always_redraw(change_af_cos_sin)
        sine_transform = always_redraw(show_transform)

        self.play(Write(sin_mob), FadeIn(sin_t, shift=DOWN * 0.3), run_time=2)
        self.wait()
        self.play(Write(af_mob))

        self.wait()
        self.play(vt_phase.animate.set_value(-PI / 2))
        self.wait()
        self.play(
            FadeOut(af_mob),
            sin_t.animate.set_color(REDUCIBLE_VIOLET)
            .scale(0.5)
            .next_to(sine_transform, LEFT, aligned_edge=DOWN, buff=0.3),
            sin_mob.animate.scale(0.7).shift(UP * 2),
            Write(sine_transform),
        )

        vt_phase.set_value(0)

        def changing_og_signal():
            # cos analysis frequency
            og_signal = get_sine_func(
                freq=original_frequency, phase=vt_phase.get_value()
            )
            _, signal_mob = plot_time_domain(
                og_signal, t_max=t_max, color=REDUCIBLE_YELLOW
            )

            return signal_mob.scale(0.7).shift(UP * 2)

        changing_signal = always_redraw(changing_og_signal)

        self.play(FadeIn(changing_signal))
        self.play(FadeOut(sin_mob))
        self.wait()

        self.play(vt_phase.animate.set_value(PI / 2))
        self.wait()

    def capture_sine_and_cosine_transforms(self):
        frame = self.camera.frame
        t_max = PI

        # samples per second
        sample_frequency = 40

        # total number of samples
        n_samples = sample_frequency

        analysis_frequencies = [
            sample_frequency * m / n_samples for m in range(n_samples // 2)
        ]

        original_frequency = analysis_frequencies[4]

        # just an array of 5 dots to position the sine waves
        positions = (
            VGroup(*[Dot() for i in range(5)])
            .arrange(DOWN, buff=1.3)
            .to_edge(LEFT, buff=3)
        )

        vt_phase = ValueTracker(0)

        scale = 0.4
        dot_prod_scale = 1

        def get_dot_prod_cos_bar():
            cos_analysis_freq = get_cosine_func(freq=original_frequency)
            sin_analysis_freq = get_sine_func(freq=original_frequency)

            og_signal = get_cosine_func(
                freq=original_frequency, phase=vt_phase.get_value()
            )

            cos_dot_prod = (
                inner_prod(
                    og_signal,
                    cos_analysis_freq,
                    x_max=t_max,
                    num_points=sample_frequency,
                )
                * dot_prod_scale
            )
            sin_dot_prod = (
                inner_prod(
                    og_signal,
                    sin_analysis_freq,
                    x_max=t_max,
                    num_points=sample_frequency,
                )
                * dot_prod_scale
            )

            barchart = BarChart(
                values=[cos_dot_prod, sin_dot_prod],
                y_range=[-20, 20, 10],
                x_length=1,
                y_length=3,
                bar_width=0.3,
                bar_colors=[REDUCIBLE_GREEN_LIGHTER, REDUCIBLE_ORANGE],
                y_axis_config={
                    "label_constructor": CustomLabel,
                    "font_size": 8,
                },
            )

            barchart.scale(2).move_to(RIGHT * 3)

            return barchart

        # original signal that will keep changing
        def changing_original_signal():
            cos_signal = get_cosine_func(
                phase=vt_phase.get_value(), freq=original_frequency
            )

            signal_axis, signal_mob = plot_time_domain(
                cos_signal, t_max=t_max, color=REDUCIBLE_YELLOW
            )

            vg = VGroup(signal_axis, signal_mob)

            return vg.scale(scale).move_to(positions[0], aligned_edge=LEFT)

        def changing_sin_prod():
            og_signal = get_cosine_func(
                phase=vt_phase.get_value(), freq=original_frequency
            )

            sin_af = get_sine_func(freq=original_frequency)

            prod_f = get_prod_functions(og_signal, sin_af)

            axis, sine_prod = plot_time_domain(
                prod_f, t_max=t_max, color=REDUCIBLE_ORANGE
            )
            vg = VGroup(axis, sine_prod)

            return vg.scale(scale).move_to(positions[4], aligned_edge=LEFT)

        def changing_cos_prod():
            og_signal = get_cosine_func(
                phase=vt_phase.get_value(), freq=original_frequency
            )

            cos_af = get_cosine_func(freq=original_frequency)

            prod_f = get_prod_functions(og_signal, cos_af)

            axis, cos_prod = plot_time_domain(
                prod_f, t_max=t_max, color=REDUCIBLE_GREEN_LIGHTER
            )
            vg = VGroup(axis, cos_prod)
            return vg.scale(scale).move_to(positions[2], aligned_edge=LEFT)

        # cos based analysis frequency
        cos_axis, cos_af = plot_time_domain(
            get_cosine_func(freq=original_frequency),
            t_max=t_max,
            color=REDUCIBLE_VIOLET,
        )
        cos_af_vg = VGroup(cos_axis, cos_af)
        cos_af_vg.scale(scale).move_to(positions[1], aligned_edge=LEFT)

        # sin based analysis frequency
        sin_axis, sin_af = plot_time_domain(
            get_sine_func(freq=original_frequency),
            t_max=t_max,
            color=REDUCIBLE_CHARM,
        )
        sin_af_vg = VGroup(sin_axis, sin_af)
        sin_af_vg.scale(scale).move_to(positions[3], aligned_edge=LEFT)

        og_signal_mob = always_redraw(changing_original_signal)
        sin_prod_mob = always_redraw(changing_sin_prod)
        cos_prod_mob = always_redraw(changing_cos_prod)
        cos_dot_prod_mob = always_redraw(get_dot_prod_cos_bar)

        og_t = (
            Text(
                f"x[n]",
                font=REDUCIBLE_FONT,
                weight=BOLD,
            )
            .scale(0.4)
            .next_to(og_signal_mob, LEFT)
        )
        af_sine_t = (
            Text(f"sin(x)", font=REDUCIBLE_FONT, weight=BOLD)
            .scale(0.4)
            .next_to(sin_af_vg, LEFT)
        )
        af_cos_t = (
            Text(f"cos(x)", font=REDUCIBLE_FONT, weight=BOLD)
            .scale(0.4)
            .next_to(cos_af_vg, LEFT)
        )
        sin_prod_t = (
            Text(f"x[n] · sin(x)", font=REDUCIBLE_FONT, weight=BOLD)
            .scale(0.4)
            .next_to(sin_prod_mob, LEFT)
        )
        cos_prod_t = (
            Text(f"x[n] · cos(x)", font=REDUCIBLE_FONT, weight=BOLD)
            .scale(0.4)
            .next_to(cos_prod_mob, LEFT)
        )

        self.play(
            Write(og_signal_mob),
            Write(cos_af_vg),
            Write(cos_prod_mob),
            Write(sin_af_vg),
            Write(sin_prod_mob),
        )

        self.play(
            LaggedStartMap(
                FadeIn, VGroup(*[og_t, af_sine_t, af_cos_t, sin_prod_t, cos_prod_t])
            )
        )

        self.play(FadeIn(cos_dot_prod_mob))
        self.wait()

        self.play(
            vt_phase.animate.set_value(4 * 2 * PI),
            run_time=25,
            rate_func=rate_functions.ease_in_out_sine,
        )
        self.wait()

        # signals_vg = VGroup(
        #     og_signal_mob,
        #     cos_af_vg,
        #     cos_prod_mob,
        #     sin_af_vg,
        #     sin_prod_mob,
        #     og_t,
        #     af_sine_t,
        #     af_cos_t,
        #     sin_prod_t,
        #     cos_prod_t,
        # )

        cos_dot_prod_mob.clear_updaters()

        self.play(
            frame.animate.scale(1.1).shift(RIGHT * 1.3),
            cos_dot_prod_mob.animate.shift(LEFT * 2.3),
        )

        # we need to remake this function with the only change of the new position
        # and then swap the mobjects. a bit inconvenient but :(
        def get_dot_prod_cos_bar_new():
            cos_analysis_freq = get_cosine_func(freq=original_frequency)
            sin_analysis_freq = get_sine_func(freq=original_frequency)

            og_signal = get_cosine_func(
                freq=original_frequency, phase=vt_phase.get_value()
            )

            cos_dot_prod = (
                inner_prod(
                    og_signal,
                    cos_analysis_freq,
                    x_max=t_max,
                    num_points=sample_frequency,
                )
                * dot_prod_scale
            )
            sin_dot_prod = (
                inner_prod(
                    og_signal,
                    sin_analysis_freq,
                    x_max=t_max,
                    num_points=sample_frequency,
                )
                * dot_prod_scale
            )

            barchart = BarChart(
                values=[cos_dot_prod, sin_dot_prod],
                y_range=[-20, 20, 10],
                x_length=1,
                y_length=3,
                bar_width=0.3,
                bar_colors=[REDUCIBLE_GREEN_LIGHTER, REDUCIBLE_ORANGE],
                y_axis_config={
                    "label_constructor": CustomLabel,
                    "font_size": 8,
                },
            )

            barchart.scale(2).move_to(cos_dot_prod_mob)

            return barchart

        cos_dot_prod_new = always_redraw(get_dot_prod_cos_bar_new)

        self.play(FadeIn(cos_dot_prod_new))
        self.play(FadeOut(cos_dot_prod_mob))

        number_plane = (
            NumberPlane(
                x_length=5,
                y_length=5,
                x_range=[-2, 2],
                y_range=[-2, 2],
                background_line_style={"stroke_color": REDUCIBLE_VIOLET},
            )
            .set_opacity(0.7)
            .next_to(cos_dot_prod_new, RIGHT, buff=1)
        )

        def redraw_arc():
            radius = Line(number_plane.c2p(0, 0), number_plane.c2p(1, 0)).width
            np_center = number_plane.c2p(0, 0)
            arc_center = (np_center[0], np_center[1], 0)
            return (
                Arc(radius, angle=vt_phase.get_value())
                .move_arc_center_to(arc_center)
                .set_color(REDUCIBLE_YELLOW)
            )

        arc = always_redraw(redraw_arc)

        self.play(Write(number_plane), vt_phase.animate.set_value(0))
        self.play(FadeIn(arc))
        self.play(vt_phase.animate.set_value(2 * PI), run_time=10)
