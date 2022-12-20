import sys

### THIS IS A WORKAROUND FOR NOW
### IT REQUIRES RUNNING MANIM FROM INSIDE DIRECTORY VIDEO-DEV
sys.path.insert(1, "common/")

from manim import *


from dft_utils import *
from reducible_colors import *
from math import degrees
from classes import CustomLabel, LabeledDot


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

        # original_freq_mob = self.show_similarity_operation()
        # self.show_line_sequence(original_freq_mob)
        self.show_signal_cosines()

    def show_similarity_operation(self):

        frame = self.camera.frame
        t_max = 2 * PI

        original_freq = 2
        n_samples = 16

        cos_og = get_cosine_func(freq=original_freq)

        cos_vg = display_signal(cos_og, num_points=n_samples)
        original_freq_mob = VGroup(*cos_vg[2:]).set_color(REDUCIBLE_YELLOW)
        axis_and_lines = VGroup(*cos_vg[:2]).set_opacity(0.4)

        og_fourier = get_fourier_bar_chart(
            cos_og, t_max=t_max, n_samples=n_samples, height_scale=3, bar_width=0.4
        ).shift(DOWN * 2)

        freq_label = (
            Text(f"ƒ = {original_freq:.2f} Hz", font=REDUCIBLE_FONT, weight=BOLD)
            .scale(0.7)
            .add_updater(
                lambda mob: mob.next_to(axis_and_lines, UP, aligned_edge=LEFT, buff=0.5)
            )
        )
        self.play(Write(original_freq_mob), run_time=1.5)
        self.play(Write(axis_and_lines), FadeIn(freq_label, shift=UP * 0.3))
        self.wait()

        self.play(
            original_freq_mob.animate.scale(0.7).shift(UP * 1),
            axis_and_lines.animate.scale(0.7).shift(UP * 1),
            freq_label.animate.scale(0.7),
        )
        self.wait()

        self.play(LaggedStartMap(FadeIn, og_fourier))
        self.wait()

        self.play(frame.animate.scale(0.8).shift(UP * 0.2), FadeOut(og_fourier))
        self.wait()

        frequency_tracker = ValueTracker(3)

        def get_signal_of_frequency():
            analysis_freq_func = get_cosine_func(freq=frequency_tracker.get_value())
            new_signal = (
                display_signal(
                    analysis_freq_func, color=FREQ_DOMAIN_COLOR, num_points=n_samples
                )[2:]
                .set_color(FREQ_DOMAIN_COLOR)
                .scale_to_fit_width(original_freq_mob.width)
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

            barchart[1:].set_opacity(0).next_to(
                original_freq_mob, DOWN, buff=1.5, aligned_edge=LEFT
            )

            barchart[0].next_to(original_freq_mob, DOWN, buff=1.5, aligned_edge=LEFT)

            return barchart

        changing_analysis_freq_signal = always_redraw(get_signal_of_frequency)
        changing_bar_dot_prod = always_redraw(get_dot_prod_bar)

        bg_rect = (
            Rectangle(height=0.3, width=original_freq_mob.width, color=REDUCIBLE_GREEN)
            .set_opacity(0.3)
            .set_stroke(opacity=0.3)
            .next_to(original_freq_mob, DOWN, buff=1.5)
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

        self.play(
            frequency_tracker.animate.set_value(3.5),
            run_time=4,
            rate_func=rate_functions.ease_in_out_sine,
        )
        self.play(
            frequency_tracker.animate.set_value(4),
            run_time=4,
            rate_func=rate_functions.ease_in_out_sine,
        )
        self.play(
            frequency_tracker.animate.set_value(3.6),
            run_time=4,
            rate_func=rate_functions.ease_in_out_sine,
        )
        self.play(
            frequency_tracker.animate.set_value(original_freq),
            run_time=6,
            rate_func=rate_functions.ease_in_out_sine,
        )

        self.wait()
        self.play(
            FadeOut(changing_analysis_freq_signal),
            FadeOut(freq_label),
            FadeOut(
                changing_bar_dot_prod,
                bg_rect,
                v_similar_txt,
                n_similar_txt,
                shift=DOWN * 0.3,
            ),
        )

        return cos_vg

    def show_line_sequence(self, cos_vg):

        n_samples = 16
        original_freq = 2

        original_freq_mob = cos_vg[2]
        axis_and_lines = cos_vg[:2]
        dots = cos_vg[3]

        self.play(
            original_freq_mob.animate.move_to(ORIGIN),
            axis_and_lines.animate.set_opacity(0),
            FadeOut(dots),
        )
        axis_and_lines.move_to(ORIGIN)

        vert_samples = get_vertical_lines_as_samples(
            original_freq_mob, axis_and_lines[0], num_points=n_samples
        ).set_stroke(width=5)

        axes_small_amp, signal_small_amp = plot_time_domain(
            get_cosine_func(amplitude=0.7, freq=original_freq), t_max=2 * PI
        )
        axes_small_amp.scale(0.7)
        signal_small_amp.scale(0.7)

        vert_samples_smaller = get_vertical_lines_as_samples(
            signal_small_amp, axes_small_amp, num_points=n_samples
        ).set_stroke(width=5)
        vert_samples_smaller_analysis = vert_samples_smaller.copy().set_color(
            REDUCIBLE_VIOLET
        )

        self.play(LaggedStartMap(Write, vert_samples))
        self.wait()
        self.play(FadeOut(original_freq_mob))

        self.play(Transform(vert_samples, vert_samples_smaller))
        self.wait()

        self.wait()

        left_bracket = (
            Tex("[")
            .scale_to_fit_height(vert_samples[0].height * 2)
            .next_to(vert_samples, LEFT, buff=0.2)
        )
        right_bracket = (
            Tex("]")
            .scale_to_fit_height(vert_samples[0].height * 2)
            .next_to(vert_samples, RIGHT, buff=0.2)
        )

        vector = VGroup(left_bracket, vert_samples, right_bracket)

        self.play(FadeIn(left_bracket, right_bracket, shift=DOWN * 0.3))

        self.wait()
        vector_purple = vector.copy().set_color(REDUCIBLE_VIOLET).shift(DOWN * 1)

        times_symbol = MathTex(r"\times").scale(0.8).shift(UP * 0.4)
        self.play(
            vector.animate.shift(UP * 1.5).set_color(REDUCIBLE_YELLOW),
            FadeIn(vector_purple, times_symbol),
        )

        self.wait()
        self.play(*[FadeOut(mob) for mob in self.mobjects])

    def show_signal_cosines(self):
        t_max = 6 * PI

        original_freq = 2

        cos_og = get_cosine_func(freq=original_freq, amplitude=1)

        cos_vg = display_signal(cos_og)
        original_freq_mob = VGroup(*cos_vg[2:])
        original_freq_mob.set_color(REDUCIBLE_YELLOW).set_stroke(width=12)

        analysis_freqs = VGroup()
        for i in range(1, 9):
            af_vg = display_signal(get_cosine_func(freq=i, amplitude=0.3))
            af = VGroup(*af_vg[2:])
            af.set_stroke(opacity=1 / i)
            [d.set_opacity(1 / i) for d in af[1]]
            analysis_freqs.add(af)

        analysis_freqs.arrange(DOWN).scale(0.6).set_color(REDUCIBLE_VIOLET)

        times = MathTex(r"\times").scale(2)

        scene = (
            VGroup(original_freq_mob, times, analysis_freqs)
            .arrange(RIGHT, buff=1)
            .shift(LEFT * 0.5)
        )
        analysis_freqs.move_to(original_freq_mob, coor_mask=[0, 1, 0], aligned_edge=UP)

        main_signal_label = (
            Text("Main signal", font=REDUCIBLE_FONT, weight=BOLD)
            .scale(0.8)
            .next_to(original_freq_mob, UP, buff=1.0)
        )
        af_label = (
            Text("Comparison frequencies", font=REDUCIBLE_FONT, weight=BOLD)
            .scale(0.6)
            .next_to(main_signal_label, RIGHT)
            .move_to(analysis_freqs, coor_mask=[1, 0, 0])
        )

        self.play(Write(original_freq_mob))
        self.play(FadeIn(times))
        self.play(LaggedStartMap(FadeIn, analysis_freqs), run_time=2)
        self.play(FadeIn(main_signal_label, af_label, shift=UP * 0.3))


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
        t_max = 2 * PI

        original_freq = 2

        # samples per second
        sample_frequency = 16

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

            rects = get_fourier_rects_from_custom_matrix(
                signal_function, af_matrix, n_samples=n_samples, t_max=t_max
            )

            return rects.move_to(DOWN * 3.5, aligned_edge=DOWN)

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

        sampled_dots_af = [
            analysis_freq(v) for v in np.linspace(0, t_max, 10, endpoint=False)
        ]
        sampled_dots_signal = [
            get_cosine_func(freq=original_freq, phase=vt_phase.get_value())(v)
            for v in np.linspace(0, t_max, 10, endpoint=False)
        ]
        prod_per_sample = [
            af * s for af, s in zip(sampled_dots_af, sampled_dots_signal)
        ]

        self.play(LaggedStartMap(FadeIn, [*dots_analysis_freq, *dots_cos_mob]))
        self.wait()

        barchart = BarChart(
            prod_per_sample,
            bar_colors=[REDUCIBLE_PURPLE],
            bar_width=1,
            x_length=dots_analysis_freq.width,
        )
        barchart.remove(barchart[2])
        barchart.stretch_to_fit_width(dots_analysis_freq.width).stretch_to_fit_height(
            analysis_freq_mob.height * 1.3
        ).next_to(analysis_freq_mob, DOWN, buff=0.5, aligned_edge=LEFT)

        self.play(Write(barchart[1]))
        self.play(LaggedStartMap(FadeIn, barchart[0]))
        self.wait()

        self.play(
            *[
                bar.animate.set_color(REDUCIBLE_YELLOW)
                if prod > 0
                else bar.animate.set_color(REDUCIBLE_BLUE)
                for prod, bar in zip(prod_per_sample, barchart[0])
            ]
        )
        self.wait()

        self.play(
            barchart.animate.change_bar_values(
                [0 for i in range(len(prod_per_sample))], update_colors=False
            )
        )

    def test_cases_again(self):
        frame = self.camera.frame
        t_min = 0
        t_max = 2 * PI

        # samples per second
        n_samples = 16

        # total number of samples
        sample_frequency = n_samples

        analysis_frequencies = [
            sample_frequency * m / n_samples for m in range(n_samples)
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

        display_axes_lines = (
            display_signal(
                get_cosine_func(
                    amplitude=vt_amplitude.get_value(),
                    freq=vt_frequency.get_value(),
                    phase=vt_phase.get_value(),
                    b=vt_b.get_value(),
                )
            )
            .scale(0.6)
            .shift(UP)
        )
        axes_and_lines = VGroup(display_axes_lines[0], display_axes_lines[1])

        def change_phase_redraw():
            phase_ch_cos = get_cosine_func(
                amplitude=vt_amplitude.get_value(),
                freq=vt_frequency.get_value(),
                phase=vt_phase.get_value(),
                b=vt_b.get_value(),
            )
            changing_signal = display_signal(phase_ch_cos)

            # we only return the displayed signal
            changing_signal_and_points = VGroup(changing_signal[2], changing_signal[3])
            return changing_signal_and_points.scale(0.6).shift(UP)

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

            rects = get_fourier_rects_from_custom_matrix(
                signal_function, af_matrix, n_samples=n_samples, t_max=t_max
            )

            return rects.move_to(DOWN * 3.5, aligned_edge=DOWN)

        changing_signal_mob = always_redraw(change_phase_redraw)
        freq_analysis = always_redraw(updating_transform_redraw)
        changing_tex_group = always_redraw(change_text_redraw)

        self.play(Write(changing_signal_mob), FadeIn(freq_analysis, axes_and_lines))
        self.play(FadeIn(changing_tex_group))

        self.play(
            vt_amplitude.animate.set_value(0.5),
            run_time=0.8,
            rate_func=rate_functions.ease_in_out_sine,
        )
        self.wait()
        self.play(
            vt_amplitude.animate.set_value(0.1),
            run_time=0.8,
            rate_func=rate_functions.ease_in_out_sine,
        )
        self.wait()
        self.play(
            vt_amplitude.animate.set_value(0),
            run_time=0.8,
            rate_func=rate_functions.ease_in_out_sine,
        )
        self.wait()
        self.play(
            vt_amplitude.animate.set_value(1),
            run_time=0.8,
            rate_func=rate_functions.ease_in_out_sine,
        )
        self.wait()

        self.play(
            vt_frequency.animate.set_value(analysis_frequencies[3]),
            rate_func=rate_functions.ease_in_out_sine,
        )
        self.wait()
        self.play(
            vt_frequency.animate.set_value(analysis_frequencies[4]),
            rate_func=rate_functions.ease_in_out_sine,
        )
        self.wait()

        self.play(
            vt_b.animate.set_value(0.3), rate_func=rate_functions.ease_in_out_sine
        )
        self.wait()
        self.play(
            vt_b.animate.set_value(0.5), rate_func=rate_functions.ease_in_out_sine
        )
        self.wait()
        self.play(vt_b.animate.set_value(0), rate_func=rate_functions.ease_in_out_sine)
        self.wait()

        for i in range(1, 5):
            self.play(
                vt_phase.animate.set_value(i * PI / 2),
                rate_func=rate_functions.ease_in_out_sine,
            )
            self.wait()

        self.play(
            vt_phase.animate.set_value(20 * PI / 2),
            run_time=10,
            rate_func=rate_functions.ease_in_out_sine,
        )


class SolvingPhaseProblem(MovingCameraScene):
    def construct(self):
        reset_frame = self.camera.frame.save_state()

        self.hacky_sine_waves()

        self.play(*[FadeOut(mob) for mob in self.mobjects])
        self.play(Restore(reset_frame))

        self.capture_sine_and_cosine_transforms()
        self.play(*[FadeOut(mob) for mob in self.mobjects])
        self.play(Restore(reset_frame))

        self.sum_up_dft()
        self.play(*[FadeOut(mob) for mob in self.mobjects])
        self.play(Restore(reset_frame))

        self.final_tests_dft()

    def hacky_sine_waves(self):
        original_frequency = 4
        t_max = 2 * PI

        # samples per second
        sample_frequency = 16

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

        sin_af_matrix = get_sin_dft_matrix(n_samples)

        def show_transform():
            signal_function = get_sine_func(
                amplitude=vt_amplitude.get_value(),
                freq=vt_frequency.get_value(),
                phase=vt_phase.get_value(),
                b=vt_b.get_value(),
            )

            rects = get_fourier_rects_from_custom_matrix(
                signal_function, sin_af_matrix, n_samples=n_samples, t_max=t_max
            )
            return rects.move_to(DOWN * 3.5, aligned_edge=DOWN)

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
            # align the annotation to the bottom of the actual bar, not the labels
            sin_t.animate.set_color(REDUCIBLE_VIOLET)
            .scale(0.5)
            .next_to(sine_transform[0][0][0], LEFT, aligned_edge=DOWN, buff=0.3),
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

        np_center = number_plane.c2p(0, 0)
        vt_phase.set_value(0)

        def redraw_arc():
            radius = Line(np_center, number_plane.c2p(1, 0)).width
            return (
                Arc(radius, angle=-vt_phase.get_value())
                .move_arc_center_to(np_center)
                .set_color(REDUCIBLE_YELLOW)
            )

        arc = always_redraw(redraw_arc)
        arc.move_arc_center_to(np_center)

        def redraw_arc_coords():
            current_coord = arc.get_end()

            # this is the vertical line, from x axis to point
            x_line_start = (current_coord[0], np_center[1], 0)
            x_line_end = (current_coord[0], current_coord[1], 0)

            x_line = DashedLine(x_line_start, x_line_end).set_stroke(
                width=2, color=REDUCIBLE_YELLOW
            )

            # this is the horizontal line, from y axis to point
            y_line_start = (np_center[0], current_coord[1], 0)
            y_line_end = (current_coord[0], current_coord[1], 0)

            y_line = DashedLine(y_line_start, y_line_end).set_stroke(
                width=2, color=REDUCIBLE_YELLOW
            )

            return VGroup(x_line, y_line)

        def redraw_vector():
            current_coord = arc.get_end()
            vector = Arrow(
                np_center,
                current_coord,
                buff=0,
                max_tip_length_to_length_ratio=0.1,
                max_stroke_width_to_length_ratio=2,
            ).set_color(REDUCIBLE_YELLOW)
            return vector

        arc_lines = always_redraw(redraw_arc_coords)
        vector = always_redraw(redraw_vector)

        self.play(Write(number_plane))
        self.play(FadeIn(arc))
        self.play(FadeIn(arc_lines, vector))
        self.play(vt_phase.animate.set_value(2 * PI), run_time=10)

        brace = (
            Brace(vector, UP)
            .set_color(REDUCIBLE_YELLOW)
            .set_stroke(BLACK, 3, background=True)
        )
        xy_t = (
            MathTex("r = \sqrt{x^2 + y^2}")
            .scale(0.5)
            .next_to(brace, UP)
            .scale_to_fit_width(vector.width)
        )
        surr_rect = (
            SurroundingRectangle(xy_t, buff=SMALL_BUFF, color=BLACK, corner_radius=0.1)
            .set_stroke(width=0)
            .set_fill(BLACK, opacity=0.7)
        )

        self.play(focus_on(frame, vector, buff=7), run_time=3)
        self.play(FadeIn(surr_rect, xy_t, shift=DOWN * 0.3), Write(brace))

        self.wait()

    def sum_up_dft(self):
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

        main_signal = (
            plot_time_domain(
                get_cosine_func(freq=original_frequency, amplitude=0.4), t_max=t_max
            )[1]
            .scale(0.7)
            .shift(RIGHT * 0.6)
        )

        cos_matrix = self.get_analysis_matrix_mob(analysis_frequencies, "cos")
        sin_matrix = (
            self.get_analysis_matrix_mob(analysis_frequencies, "sin")
            .scale(0.4)
            .to_corner(DL)
        )

        self.play(Write(cos_matrix))
        self.wait()
        self.play(cos_matrix.animate.scale(0.4).to_corner(UL))

        cos_t = (
            Text("cos(x)", font=REDUCIBLE_FONT, weight=BOLD)
            .set_color(REDUCIBLE_VIOLET)
            .next_to(cos_matrix, DOWN, buff=0)
        ).scale(0.4)
        self.play(FadeIn(cos_t, shift=DOWN * 0.3))

        self.wait()
        self.play(Write(sin_matrix))

        sin_t = (
            Text("sin(x)", font=REDUCIBLE_FONT, weight=BOLD)
            .set_color(REDUCIBLE_CHARM)
            .next_to(sin_matrix, UP, buff=0)
        ).scale(0.4)
        self.play(FadeIn(sin_t, shift=UP * 0.3))

        x_t = Text("x", font=REDUCIBLE_FONT, weight=BOLD).next_to(
            main_signal, UP, buff=1.5
        )
        y_t = Text("y", font=REDUCIBLE_FONT, weight=BOLD).next_to(
            main_signal, DOWN, buff=1.5
        )
        self.play(Write(main_signal), Write(x_t), Write(y_t))

        coords = (
            VGroup(
                *[
                    Text(
                        "|(x, y)|",
                        font=REDUCIBLE_FONT,
                        weight=BOLD,
                        t2c={"x": REDUCIBLE_VIOLET, "y": REDUCIBLE_CHARM},
                    )
                    .scale(0.7)
                    .set_opacity(2 / (i + 1))
                    for i in range(8)
                ]
            )
            .arrange(DOWN, buff=0.2)
            .next_to(main_signal, RIGHT, buff=1.8)
        )
        brace = Brace(coords, LEFT, buff=0.7)

        self.play(FadeIn(brace, shift=LEFT * 0.3))
        self.play(FadeIn(coords, shift=DOWN * 0.3))

        self.wait()

    def final_tests_dft(self):
        frame = self.camera.frame
        t_min = 0
        t_max = 2 * PI

        # samples per second
        n_samples = 16

        # total number of samples
        sample_frequency = n_samples

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

        display_signal_axis_lines = (
            display_signal(
                get_cosine_func(
                    amplitude=vt_amplitude.get_value(),
                    freq=vt_frequency.get_value(),
                    phase=vt_phase.get_value(),
                    b=vt_b.get_value(),
                ),
                TIME_DOMAIN_COLOR,
            )
            .scale(0.6)
            .move_to(UP * 2)
        )
        axis_lines = VGroup(display_signal_axis_lines[0], display_signal_axis_lines[1])

        def changing_signal_redraw():
            # for viz purposes, we are going to make the signal's amplitude smaller
            # all the calculations will be done with the actual amplitude, though
            # this helps the redrawing function deal better with the signal
            changing_func = get_cosine_func(
                amplitude=vt_amplitude.get_value(),
                freq=vt_frequency.get_value(),
                phase=vt_phase.get_value(),
                b=vt_b.get_value(),
            )

            displayed_signal = display_signal(changing_func, TIME_DOMAIN_COLOR)
            signal_and_points = VGroup(displayed_signal[2], displayed_signal[3])

            return signal_and_points.scale(0.6).shift(UP * 2)

        def updating_transform_redraw():
            signal_function = get_cosine_func(
                amplitude=vt_amplitude.get_value(),
                freq=vt_frequency.get_value(),
                phase=vt_phase.get_value(),
                b=vt_b.get_value(),
            )

            rects = get_fourier_bar_chart(
                signal_function, t_max=t_max, n_samples=n_samples, height_scale=3
            )

            return (
                rects.move_to(DOWN * 2, aligned_edge=DOWN)
                .stretch_to_fit_width(changing_signal_mob.width)
                .set_color(REDUCIBLE_YELLOW)
            )

        sides_buff = 0.5

        sin_dft_matrix = get_sin_dft_matrix(n_samples)
        cos_dft_matrix = get_cosine_dft_matrix(n_samples)

        def updating_sine_transform_redraw():
            signal_function = get_cosine_func(
                amplitude=vt_amplitude.get_value(),
                freq=vt_frequency.get_value(),
                phase=vt_phase.get_value(),
                b=vt_b.get_value(),
            )

            rects = get_fourier_rects_from_custom_matrix(
                signal_function,
                sin_dft_matrix,
                n_samples=n_samples,
                t_max=t_max,
                rect_width=0.2,
            ).set_color(REDUCIBLE_CHARM)

            rects = VGroup(*[r[0] for r in rects[0]])

            return rects.next_to(
                freq_analysis, LEFT, aligned_edge=DOWN, buff=sides_buff
            )

        def updating_cosine_transform_redraw():
            signal_function = get_cosine_func(
                amplitude=vt_amplitude.get_value(),
                freq=vt_frequency.get_value(),
                phase=vt_phase.get_value(),
                b=vt_b.get_value(),
            )

            rects = get_fourier_rects_from_custom_matrix(
                signal_function,
                cos_dft_matrix,
                n_samples=n_samples,
                t_max=t_max,
                rect_width=0.2,
            )

            rects = VGroup(*[r[0] for r in rects[0]])

            return rects.next_to(
                freq_analysis, RIGHT, aligned_edge=DOWN, buff=sides_buff
            )

        changing_signal_mob = always_redraw(changing_signal_redraw)
        freq_analysis = always_redraw(updating_transform_redraw)
        sin_freq_analysis = always_redraw(updating_sine_transform_redraw)
        cos_freq_analysis = always_redraw(updating_cosine_transform_redraw)
        changing_tex_group = always_redraw(change_text_redraw)

        sin_t = (
            Text("sin(x)", font=REDUCIBLE_FONT, weight=BOLD)
            .set_color(REDUCIBLE_CHARM)
            .scale(0.4)
            .next_to(sin_freq_analysis, DOWN, buff=0.3)
        )
        cos_t = (
            Text("cos(x)", font=REDUCIBLE_FONT, weight=BOLD)
            .set_color(REDUCIBLE_VIOLET)
            .scale(0.4)
            .next_to(cos_freq_analysis, DOWN, buff=0.3)
        )
        complex_t = (
            Text("sin(x) + cos(x)", font=REDUCIBLE_FONT, weight=BOLD)
            .set_color(REDUCIBLE_YELLOW)
            .scale(0.4)
            .next_to(freq_analysis, DOWN, buff=0.3)
        )

        self.play(
            Write(changing_signal_mob),
            FadeIn(freq_analysis, sin_freq_analysis, cos_freq_analysis, axis_lines),
            FadeIn(sin_t, cos_t, complex_t, shift=UP * 0.3),
        )
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

        self.play(vt_phase.animate.set_value(20 * PI / 2), run_time=5, rate_func=linear)

        self.wait()

    # local utils
    def get_analysis_matrix_mob(self, analysis_frequencies, func="cos", t_max=2 * PI):
        if func != "cos" and func != "sin":
            raise ValueError('func can either be "cos" or "sin"')

        function = get_cosine_func if func == "cos" else get_sine_func
        color = REDUCIBLE_VIOLET if func == "cos" else REDUCIBLE_CHARM

        af = [function(freq=f, amplitude=0.1) for f in analysis_frequencies]
        mobs = (
            VGroup(*[plot_time_domain(f, t_max=t_max, color=color)[1] for f in af])
            .arrange(DOWN, buff=0.2)
            .rotate(PI / 2)
        )

        return VGroup(*mobs)


class InterpretDFT(MovingCameraScene):
    def construct(self):
        reset_frame = self.camera.frame.save_state()

        self.visualize_complex_and_frequencies()

    def visualize_complex_and_frequencies(self):

        frame = self.camera.frame
        t_max = 2 * PI

        # samples per second
        n_samples = 10

        # total number of samples
        sample_frequency = n_samples

        analysis_frequencies = [
            sample_frequency * m / n_samples for m in range(n_samples)
        ]
        original_frequency = analysis_frequencies[2]

        matrix_elements = []
        power = 0
        for i in range(n_samples):
            row = []
            for j in range(n_samples):
                if i == 0 or j == 0:
                    power = 0
                else:
                    power = i * j

                power_index = f"{power}"
                row.append(r"\omega^{" + power_index + "}")

            matrix_elements.append(row)

        dft_matrix_tex = Matrix(matrix_elements).shift(DOWN * 3)
        omega_definition = MathTex(
            r"\text{where} \ \omega = e ^{-\frac{2 \pi i}{N}}"
        ).next_to(dft_matrix_tex, UP)

        title = (
            Text("The DFT Matrix", font=REDUCIBLE_FONT, weight=BOLD)
            .scale(0.8)
            .to_edge(UP)
        )
        self.play(FadeIn(title, shift=UP * 0.3))
        self.play(FadeIn(dft_matrix_tex, shift=UP * 0.3))
        self.play(FadeIn(omega_definition, shift=UP * 0.3))
        self.wait()
        self.play(
            dft_matrix_tex.animate.scale(0.8).move_to(ORIGIN).shift(DOWN * 0.5),
            FadeOut(omega_definition, shift=UP * 0.3),
        )
        self.wait()

        cos_af_matrix = VGroup(
            *[
                VGroup(
                    *plot_time_domain(
                        get_cosine_func(freq=f, amplitude=0.13), t_max=t_max
                    )
                )
                .stretch_to_fit_width(dft_matrix_tex[0].width)
                .move_to(dft_matrix_tex[0][i * n_samples], aligned_edge=LEFT)
                for i, f in enumerate(range(n_samples))
            ]
        )

        sin_af_matrix = VGroup(
            *[
                VGroup(
                    *plot_time_domain(
                        get_sine_func(freq=f, amplitude=0.13), t_max=t_max
                    )
                )
                .stretch_to_fit_width(dft_matrix_tex[0].width)
                .move_to(dft_matrix_tex[0][i * n_samples], aligned_edge=LEFT)
                for i, f in enumerate(analysis_frequencies)
            ]
        )

        [af[0].set_opacity(0) for af in cos_af_matrix]
        [af[0].set_opacity(0) for af in sin_af_matrix]

        for sin_mob, cos_mob in zip(sin_af_matrix, cos_af_matrix):
            sin_mob[1].set_color(REDUCIBLE_CHARM)
            cos_mob[1].set_color(REDUCIBLE_VIOLET)

        self.play(LaggedStartMap(Write, cos_af_matrix))
        self.wait()
        self.play(FadeOut(dft_matrix_tex[0]), LaggedStartMap(Write, sin_af_matrix))
        self.wait()

        legend = (
            VGroup(
                VGroup(
                    Text("cos(x), \nreal part", font=REDUCIBLE_FONT).scale(0.2),
                    Line(LEFT * 0.3, ORIGIN)
                    .set_stroke(width=14)
                    .set_color(REDUCIBLE_VIOLET),
                ).arrange(DOWN, buff=0.1, aligned_edge=LEFT),
                VGroup(
                    Text("sin(x), \nimaginary part", font=REDUCIBLE_FONT).scale(0.2),
                    Line(LEFT * 0.3, ORIGIN)
                    .set_stroke(width=14)
                    .set_color(REDUCIBLE_CHARM),
                ).arrange(DOWN, buff=0.1, aligned_edge=LEFT),
            )
            .arrange(DOWN, buff=0.4, aligned_edge=LEFT)
            .next_to(dft_matrix_tex, LEFT, aligned_edge=UP)
        )

        self.play(
            FadeIn(legend, shift=UP * 0.3),
            FadeOut(title, shift=UP * 0.3),
            focus_on(frame, VGroup(dft_matrix_tex, legend)),
        )
        self.wait()

        sampled_points_cos_af = VGroup(
            *[
                get_sampled_dots(
                    signal,
                    axis,
                    x_max=t_max,
                    num_points=n_samples,
                    radius=DEFAULT_DOT_RADIUS,
                ).set_color(REDUCIBLE_VIOLET)
                for axis, signal in cos_af_matrix
            ]
        )
        sampled_points_sin_af = VGroup(
            *[
                get_sampled_dots(
                    signal,
                    axis,
                    x_max=t_max,
                    num_points=n_samples,
                    radius=DEFAULT_DOT_RADIUS,
                ).set_color(REDUCIBLE_CHARM)
                for axis, signal in sin_af_matrix
            ]
        )

        number_plane = (
            ComplexPlane(
                x_length=5,
                y_length=5,
                x_range=[-2, 2],
                y_range=[-2, 2],
                background_line_style={"stroke_color": REDUCIBLE_VIOLET},
            )
            .set_opacity(0.7)
            .next_to(dft_matrix_tex, RIGHT, buff=1)
        )
        np_radius = Line(number_plane.c2p(0, 0), number_plane.c2p(0, 1)).height

        complex_circle = (
            Circle(np_radius)
            .set_color(REDUCIBLE_YELLOW)
            .move_to(number_plane.c2p(0, 0))
        )

        indices = VGroup(
            *[
                Text(str(f), font=REDUCIBLE_MONO)
                .scale(0.7)
                .next_to(cos_af_matrix[f][0], LEFT)
                for f in range(n_samples)
            ]
        ).next_to(dft_matrix_tex, LEFT, buff=0.5)
        m_t = (
            Text("m", font=REDUCIBLE_FONT, weight=BOLD, slant=ITALIC)
            .scale(0.7)
            .next_to(indices, UP, buff=0.6)
        )

        self.play(
            FadeOut(legend),
            FadeIn(indices, m_t),
            focus_on(frame, [number_plane, indices, dft_matrix_tex]),
        )
        self.wait()
        self.play(
            Write(number_plane),
            Write(complex_circle),
        )
        self.wait()

        points_on_circle = (
            Dot()
            .set_color(REDUCIBLE_YELLOW)
            .move_to(complex_circle.point_from_proportion(0))
        )
        for i, sampled_points in enumerate(
            zip(sampled_points_cos_af, sampled_points_sin_af)
        ):
            if i == 0:
                self.play(FadeIn(*sampled_points))
                continue

            _points_on_circle = VGroup(
                *[
                    Dot(radius=DEFAULT_DOT_RADIUS * 1.4)
                    .move_to(complex_circle.point_from_proportion(n / i))
                    .set_color(REDUCIBLE_YELLOW)
                    for n in range(i)
                ]
            )
            self.play(
                # enable current index: number, sine and cosine graphs and sampled dots
                indices[i].animate.set_opacity(1),
                cos_af_matrix[i][1].animate.set_stroke(opacity=1),
                *[
                    cos_af_matrix[idx][1].animate.set_stroke(opacity=0.3)
                    for idx in range(n_samples)
                    if idx != i
                ],
                sin_af_matrix[i][1].animate.set_stroke(opacity=1),
                *[
                    sin_af_matrix[idx][1].animate.set_stroke(opacity=0.3)
                    for idx in range(n_samples)
                    if idx != i
                ],
                # "disable" every other index
                *[
                    indices[idx].animate.set_opacity(0.3)
                    for idx in range(n_samples)
                    if idx != i
                ],
                *[
                    sampled_points_cos_af[idx].animate.set_opacity(0.3)
                    for idx in range(n_samples)
                    if idx < i
                ],
                *[
                    sampled_points_sin_af[idx].animate.set_opacity(0.3)
                    for idx in range(n_samples)
                    if idx < i
                ],
                LaggedStartMap(Write, sampled_points[0]),
                LaggedStartMap(Write, sampled_points[1]),
                Transform(points_on_circle, _points_on_circle),
                run_time=1 / (i + 1),
            )
            self.wait(1 / (i + 1))

        self.wait()

        og_axis_and_signal = VGroup(
            *plot_time_domain(
                get_cosine_func(freq=original_frequency, amplitude=0.2), t_max=t_max
            )
        )
        og_axis_and_signal.rotate(-90 * DEGREES).stretch_to_fit_height(
            dft_matrix_tex.height - 0.3
        )
        og_axis_and_signal[0].set_opacity(0)

        sampled_dots_main_signal = get_sampled_dots(
            og_axis_and_signal[1],
            og_axis_and_signal[0],
            num_points=n_samples,
            x_max=t_max,
        ).set_color(REDUCIBLE_YELLOW)

        brackets = Matrix([[v] for v in range(n_samples)]).stretch_to_fit_height(
            dft_matrix_tex.height
        )
        brackets[0].set_opacity(0)

        main_signal_vector = VGroup(
            brackets.move_to(og_axis_and_signal),
            og_axis_and_signal,
            sampled_dots_main_signal,
        ).next_to(dft_matrix_tex, RIGHT, buff=1)

        dot_product = Dot().next_to(dft_matrix_tex, RIGHT, buff=1)
        equal_sign = Text("=", font=REDUCIBLE_FONT, weight=BOLD).next_to(
            main_signal_vector, RIGHT, buff=0.8
        )

        np_and_circle = VGroup(number_plane, complex_circle, points_on_circle)

        # aux mobject to allow for the panning animation to work properly
        np_and_circle_aux = np_and_circle.copy().next_to(
            main_signal_vector, RIGHT, buff=2
        )

        self.play(
            FadeIn(main_signal_vector),
            Write(dot_product),
            Write(equal_sign),
            np_and_circle.animate.move_to(np_and_circle_aux),
            focus_on(
                frame,
                (indices, dft_matrix_tex, main_signal_vector, np_and_circle_aux),
                buff=1.4,
            ),
            *[indices[idx].animate.set_opacity(1) for idx in range(n_samples)],
            *[
                sin_af_matrix[idx][1].animate.set_stroke(opacity=1)
                for idx in range(n_samples)
            ],
            *[
                cos_af_matrix[idx][1].animate.set_stroke(opacity=1)
                for idx in range(n_samples)
            ],
            *[
                sampled_points_cos_af[idx].animate.set_opacity(1)
                for idx in range(n_samples)
            ],
            *[
                sampled_points_sin_af[idx].animate.set_opacity(1)
                for idx in range(n_samples)
            ],
        )

        # had to make amplitude smaller in order for the actual points
        # to land on the screen. is there

        cos_func = get_cosine_func(freq=original_frequency, amplitude=0.3)
        sampled_signal = np.array(
            [cos_func(f) for f in np.linspace(0, t_max, num=n_samples, endpoint=False)]
        ).reshape(-1, 1)
        dft_on_signal = np.fft.fft2(sampled_signal)

        self.play(FadeOut(points_on_circle))

        current_dot = LabeledDot(0).move_to(number_plane.n2p(dft_on_signal[0]))
        for i in range(n_samples):
            point = dft_on_signal[i]

            _current_dot = LabeledDot(i, label_color=WHITE).move_to(
                number_plane.n2p(point)
            )

            self.play(
                Transform(current_dot, _current_dot),
                indices[i].animate.set_opacity(1),
                cos_af_matrix[i][1].animate.set_stroke(opacity=1),
                *[
                    sampled_points_cos_af[idx].animate.set_opacity(1)
                    for idx in range(n_samples)
                    if idx == i
                ],
                *[
                    sampled_points_sin_af[idx].animate.set_opacity(1)
                    for idx in range(n_samples)
                    if idx == i
                ],
                sin_af_matrix[i][1].animate.set_stroke(opacity=1),
                # "disable" every other index
                *[
                    cos_af_matrix[idx][1].animate.set_stroke(opacity=0.3)
                    for idx in range(n_samples)
                    if idx != i
                ],
                *[
                    sin_af_matrix[idx][1].animate.set_stroke(opacity=0.3)
                    for idx in range(n_samples)
                    if idx != i
                ],
                *[
                    indices[idx].animate.set_opacity(0.3)
                    for idx in range(n_samples)
                    if idx != i
                ],
                *[
                    sampled_points_cos_af[idx].animate.set_opacity(0.3)
                    for idx in range(n_samples)
                    if idx != i
                ],
                *[
                    sampled_points_sin_af[idx].animate.set_opacity(0.3)
                    for idx in range(n_samples)
                    if idx != i
                ],
                run_time=1 / (i + 1),
            )
            self.wait(1 / (i + 1))

        vt_frequency = ValueTracker(original_frequency)
        vt_phase = ValueTracker(0)
        vt_amplitude = ValueTracker(0.2)

        dft_matrix = get_dft_matrix(n_samples)

        def dft_barchart_redraw():
            cos_func = get_cosine_func(
                freq=vt_frequency.get_value(),
                amplitude=vt_amplitude.get_value(),
                phase=vt_phase.get_value(),
            )

            rects = get_fourier_rects_from_custom_matrix(
                cos_func,
                dft_matrix,
                n_samples=n_samples,
                t_max=t_max,
                rect_scale=0.6,
                full_spectrum=True,
            )
            rects = VGroup(*[r[0] for r in rects[0]])

            return (
                rects.rotate(-90 * DEGREES)
                .stretch_to_fit_height(dft_matrix_tex.height)
                .next_to(dft_matrix_tex, RIGHT, buff=0.4)
            )

        shift_mobs = RIGHT * 0.7
        _aux_complex_plane = number_plane.copy().shift(shift_mobs)
        self.play(
            focus_on(frame, (indices, _aux_complex_plane)),
            main_signal_vector.animate.shift(shift_mobs),
            dot_product.animate.shift(shift_mobs),
            equal_sign.animate.shift(shift_mobs),
            number_plane.animate.shift(shift_mobs),
            complex_circle.animate.shift(shift_mobs),
            FadeOut(current_dot),
        )

        def main_signal_redraw():
            cos_func = get_cosine_func(
                freq=vt_frequency.get_value(),
                amplitude=vt_amplitude.get_value(),
                phase=vt_phase.get_value(),
            )

            axis_and_signal = VGroup(*plot_time_domain(cos_func, t_max=t_max))
            axis_and_signal[0].set_opacity(0)
            axis_and_signal.rotate(-90 * DEGREES)
            axis_and_signal.stretch_to_fit_height(og_axis_and_signal[1].height).move_to(
                og_axis_and_signal[1]
            )

            sampled_dots_main_signal = get_sampled_dots(
                axis_and_signal[1],
                axis_and_signal[0],
                num_points=n_samples,
                x_max=t_max,
            ).set_color(REDUCIBLE_YELLOW)

            return VGroup(axis_and_signal, sampled_dots_main_signal)

        def tracking_text_redraw():

            v_freq = f"{vt_frequency.get_value():.2f}"
            v_amplitude = f"{vt_amplitude.get_value():.2f}"
            v_phase = f"{degrees(vt_phase.get_value()) % 360:.2f}"

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

            return (
                VGroup(tex_frequency, tex_phase, tex_amplitude)
                .arrange(DOWN, aligned_edge=LEFT)
                .next_to(
                    main_signal_vector,
                    UP,
                    buff=0.4,
                )
            ).scale(0.7)

        def complex_frequencies_on_plane():
            cos_func = get_cosine_func(
                freq=vt_frequency.get_value(),
                amplitude=vt_amplitude.get_value(),
                phase=vt_phase.get_value(),
            )

            sampled_signal = np.array(
                [
                    cos_func(f)
                    for f in np.linspace(0, t_max, num=n_samples, endpoint=False)
                ]
            ).reshape(-1, 1)

            dft_on_signal = np.fft.fft2(sampled_signal)

            complex_points = VGroup(
                *[
                    LabeledDot(str(i), label_color=WHITE).move_to(
                        number_plane.n2p(point)
                    )
                    for i, point in list(enumerate(dft_on_signal))[::-1]
                ]
            )

            return complex_points

        tracking_text_changing = always_redraw(tracking_text_redraw)
        main_signal_mob_changing = always_redraw(main_signal_redraw)
        dft_barchart_changing = always_redraw(dft_barchart_redraw)
        complex_points = always_redraw(complex_frequencies_on_plane)

        self.play(
            FadeOut(sampled_dots_main_signal),
            FadeOut(og_axis_and_signal[1]),
            FadeIn(complex_points),
            FadeIn(main_signal_mob_changing),
            FadeIn(dft_barchart_changing),
            FadeIn(tracking_text_changing),
            *[indices[idx].animate.set_opacity(1) for idx in range(n_samples)],
            *[
                sin_af_matrix[idx][1].animate.set_stroke(opacity=1)
                for idx in range(n_samples)
            ],
            *[
                cos_af_matrix[idx][1].animate.set_stroke(opacity=1)
                for idx in range(n_samples)
            ],
            *[
                sampled_points_cos_af[idx].animate.set_opacity(1)
                for idx in range(n_samples)
            ],
            *[
                sampled_points_sin_af[idx].animate.set_opacity(1)
                for idx in range(n_samples)
            ],
        )
        self.wait()
        self.play(
            vt_phase.animate.set_value(4 * 2 * PI),
            run_time=10,
            rate_func=rate_functions.ease_in_out_sine,
        )
        self.wait()
        self.play(
            vt_frequency.animate.set_value(analysis_frequencies[0]),
            run_time=5,
            rate_func=rate_functions.ease_in_out_sine,
        )
        self.wait()
        self.play(
            vt_frequency.animate.set_value(analysis_frequencies[1]),
            run_time=5,
            rate_func=rate_functions.ease_in_out_sine,
        )
        self.wait()
        self.play(
            vt_frequency.animate.set_value(analysis_frequencies[2]),
            run_time=5,
            rate_func=rate_functions.ease_in_out_sine,
        )
        self.wait()
        self.play(
            vt_frequency.animate.set_value(analysis_frequencies[3]),
            run_time=5,
            rate_func=rate_functions.ease_in_out_sine,
        )


class TransitionTemplate(Scene):
    def construct(self):
        bg = ImageMobject("assets/transition-bg.png").scale_to_fit_width(
            config.frame_width
        )
        self.add(bg)
        self.wait()

        transition_points = [
            # use a list if we want multiple lines
            ["The Discrete", "Cosine Transform"],
            "Sampling",
            ["Time and Frequency", " Domains"],
            ["Similarity between", "signals"],
            ["Analysis", "Frequencies"],
            "Phase Problems",
            ["Simplifying the", "Discrete Fourier Transform"],
            "Conclusion",
        ]
        for i in range(len(transition_points)):
            self.transition(
                transition_name=transition_points[i],
                index=i + 1,
                total=len(transition_points),
            )
            self.wait()

    def transition(self, transition_name, index, total):
        """
        Create transitions easily.

        - Transition name — string, self explanatory
        - Index correspond to the position of this transition on the video
        - Total corresponds to the total amount of transitions there will be

        Total will generate a number of nodes and index will highlight that specific
        node, showing the progress.
        """

        if isinstance(transition_name, list):
            subtitles = [
                Text(t, font=REDUCIBLE_FONT, weight=BOLD).set_stroke(
                    BLACK, width=9, background=True
                )
                for t in transition_name
            ]

            title = (
                VGroup(*subtitles)
                .arrange(DOWN)
                .scale_to_fit_width(config.frame_width - 3)
                .shift(UP)
            )
        else:
            title = (
                MarkupText(transition_name, font=REDUCIBLE_FONT, weight=BOLD)
                .set_stroke(BLACK, width=10, background=True)
                .scale_to_fit_width(config.frame_width - 3)
                .shift(UP)
            )

        nodes_and_lines = VGroup()
        for n in range(1, total + 1):
            if n == index:
                node = (
                    Circle()
                    .scale(0.2)
                    .set_stroke(REDUCIBLE_YELLOW)
                    .set_fill(REDUCIBLE_YELLOW_DARKER, opacity=1)
                )
                nodes_and_lines.add(node)
            else:
                nodes_and_lines.add(
                    Circle()
                    .scale(0.2)
                    .set_stroke(REDUCIBLE_PURPLE)
                    .set_fill(REDUCIBLE_PURPLE_DARK_FILL, opacity=1)
                )

            nodes_and_lines.add(Line().set_color(REDUCIBLE_PURPLE))

        nodes_and_lines.remove(nodes_and_lines[-1])

        nodes_and_lines.arrange(RIGHT, buff=0.5).scale_to_fit_width(
            config.frame_width - 5
        ).to_edge(DOWN, buff=1)

        self.play(
            FadeIn(title, shift=UP * 0.3), LaggedStartMap(FadeIn, nodes_and_lines)
        )

        self.wait()
        self.play(FadeOut(title), FadeOut(nodes_and_lines))
