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

        self.play(FadeIn(time_domain_graph))
        self.wait()

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
            r"a_k = \cos \left( \frac{2 \pi k \hat{f}}{N} \right)"
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
        sampled_points_dots = get_sampled_dots(graph, time_axis, color=color)
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
            bar_chart = (
                BarChart(
                    values=[dot_prod],
                    y_range=[-5, 5, 5],
                    x_length=3,
                    y_length=3,
                    bar_width=0.4,
                    bar_colors=[REDUCIBLE_GREEN_LIGHTER],
                )
                .rotate(PI / 2)
                .flip(UP)
                .shift(DOWN * 2)
            )
            bar_chart.y_axis.numbers.flip(RIGHT)
            bar_chart.y_axis.numbers[0].rotate(PI / 2)
            bar_chart.y_axis.numbers[1].rotate(PI / 2)
            shift_to_align = (
                bar_chart.y_axis.numbers[0].get_center()[1]
                - bar_chart.y_axis.numbers[1].get_center()[1]
            )
            bar_chart.y_axis.numbers[1].shift(UP * shift_to_align)
            return bar_chart

        self.remove(analysis_freq_signal)
        changing_analysis_freq_signal = always_redraw(get_signal_of_frequency)
        self.add(changing_analysis_freq_signal)

        bar_chart = always_redraw(get_dot_prod_bar_graph)
        label = MathTex(r"\vec{y} \cdot \vec{a}").scale(0.6).move_to(UP * changing_analysis_freq_signal.get_bottom()[1] - SMALL_BUFF + LEFT * (MED_SMALL_BUFF - 0.02))
        self.play(
            FadeIn(bar_chart),
            FadeIn(label)
        )
        self.wait()

        self.play(frequency_tracker.animate.set_value(0), run_time=14, rate_func=linear)
        self.wait()

        self.play(frequency_tracker.animate.set_value(1), run_time=4, rate_func=linear)
        self.wait()
        self.play(frequency_tracker.animate.set_value(2), run_time=4, rate_func=linear)
        self.wait()
        self.play(frequency_tracker.animate.set_value(3), run_time=4, rate_func=linear)
        self.wait()
        self.play(frequency_tracker.animate.set_value(4), run_time=4, rate_func=linear)
        self.wait()
        self.play(frequency_tracker.animate.set_value(5), run_time=4, rate_func=linear)
        self.wait()


class MatrixDefinition(Scene):
    def construct(self):
        time_signal_func = get_cosine_func(freq=DEFAULT_FREQ)
        self.define_transformation(time_signal_func)

    def define_transformation(self, time_signal_func):
        time_domain_graph = display_signal(time_signal_func)
        signal_samples_vec = MathTex(
            r"\vec{y} = \begin{bmatrix} y_{0} & y_{1} & \cdots & y_{N-1} \end{bmatrix}"
        ).scale(0.8)
        sample_value = MathTex(r"y_k = \cos \left( \frac{2 \pi k f}{N} \right)").scale(
            0.8
        )
        freq_annotation = MathTex("f = 2").scale(0.8)
        graph_notation = VGroup(
            freq_annotation, time_domain_graph, signal_samples_vec, sample_value
        ).arrange(DOWN)

        self.play(Write(time_domain_graph), Write(freq_annotation))
        self.wait()
        self.play(FadeIn(signal_samples_vec))
        self.wait()
        self.play(FadeIn(sample_value))
        self.wait()

        self.play(graph_notation.animate.scale(DEFAULT_SCALE).shift(LEFT * 3.5))

        time_signal_func = get_cosine_func(freq=2)
        rects = get_fourier_rects_from_custom_matrix(
            time_signal_func, get_cosine_dft_matrix(16)
        )
        rects.scale(1.2).move_to(RIGHT * 3.5)

        self.play(FadeIn(rects))
        self.wait()

        NUM_SAMPLES = 16
        cosine_dft_matrix = get_cosine_dft_matrix(NUM_SAMPLES)
        fourier_rects = (
            get_fourier_rects_from_custom_matrix(
                time_signal_func,
                cosine_dft_matrix,
                n_samples=NUM_SAMPLES,
                full_spectrum=True,
            )
            .scale(1.2)
            .move_to(DOWN * 2)
        )

        for increment in range(1, 4):
            new_time_signal_func = get_cosine_func(freq=DEFAULT_FREQ + increment)
            new_time_domain_graph = display_signal(new_time_signal_func)
            new_freq_annotation = MathTex(f"f = {DEFAULT_FREQ + increment}")
            new_time_domain_graph.move_to(
                time_domain_graph.get_center()
            ).scale_to_fit_height(time_domain_graph.height)
            new_freq_annotation.move_to(
                freq_annotation.get_center()
            ).scale_to_fit_height(freq_annotation.height)

            new_rects = (
                get_fourier_rects_from_custom_matrix(
                    new_time_signal_func,
                    cosine_dft_matrix,
                    n_samples=16,
                )
                .scale_to_fit_height(rects.height)
                .move_to(rects.get_center())
            )

            self.play(
                Transform(time_domain_graph, new_time_domain_graph),
                Transform(freq_annotation, new_freq_annotation),
                Transform(rects, new_rects),
            )
            self.wait()

        analysis_freq_func = get_cosine_func(freq=DEFAULT_FREQ)

        analysis_freq_graph = (
            display_signal(analysis_freq_func, color=FREQ_DOMAIN_COLOR)
            .scale_to_fit_height(time_domain_graph.height)
            .move_to(time_domain_graph.get_center())
            .shift(RIGHT * 7)
        )

        analysis_samples_vec = MathTex(
            r"\vec{a} = \begin{bmatrix} a_{0} & a_{1} & \cdots & a_{N - 1} \end{bmatrix}"
        ).scale(0.8)
        analysis_value = MathTex(
            r"a_k = \cos \left( \frac{2 \pi k \hat{f}}{N} \right)"
        ).scale(0.8)

        analysis_samples_vec.scale_to_fit_height(signal_samples_vec.height).next_to(
            analysis_freq_graph, DOWN
        )
        analysis_value.scale_to_fit_height(sample_value.height + SMALL_BUFF).next_to(
            analysis_samples_vec, DOWN
        )

        analysis_freq_group = VGroup(
            analysis_freq_graph, analysis_samples_vec, analysis_value
        )

        self.play(FadeOut(freq_annotation), FadeTransform(rects, analysis_freq_group))
        self.wait()

        dot_product_gen = (
            MathTex(r"\vec{a} \cdot \vec{y}")
            .scale(1)
            .move_to(analysis_freq_graph.get_center())
        )
        buff = SMALL_BUFF * 2
        dot_product_gen.next_to(analysis_freq_graph, UP, buff=buff)

        analysis_freq_with_dot_prod = VGroup(dot_product_gen, analysis_freq_graph)

        self.play(
            FadeIn(dot_product_gen)
        )
        self.wait()

        analysis_freq_with_dot_prod_0 = analysis_freq_with_dot_prod.copy().scale(0.5)

        dot_prod_0 = (
            MathTex(r"\vec{a}_0 \cdot \vec{y}")
            .scale(0.5)
            .move_to(dot_product_gen.get_center())
        )

        dot_prod_1 = (
            MathTex(r"\vec{a}_1 \cdot \vec{y}")
            .scale(0.5)
            .move_to(dot_product_gen.get_center())
        )

        analysis_freq_1 = display_signal(
            get_cosine_func(freq=3), color=FREQ_DOMAIN_COLOR
        ).scale_to_fit_height(analysis_freq_with_dot_prod_0[1].height)

        dot_prod_1.next_to(analysis_freq_1, UP, buff=buff)
        analysis_freq_with_dot_prod_1 = VGroup(dot_prod_1, analysis_freq_1)

        dot_prod_2 = (
            MathTex(r"\vec{a}_2 \cdot \vec{y}")
            .scale(0.5)
            .move_to(dot_product_gen.get_center())
        )
        analysis_freq_2 = display_signal(
            get_cosine_func(freq=4), color=FREQ_DOMAIN_COLOR
        ).scale_to_fit_height(analysis_freq_with_dot_prod_0[1].height)
        dot_prod_2.next_to(analysis_freq_2, UP, buff=buff)
        analysis_freq_with_dot_prod_2 = VGroup(dot_prod_2, analysis_freq_2)

        ellipses = MathTex(r"\vdots").scale(1)

        dot_prod_n = (
            MathTex(r"\vec{a}_{M - 1} \cdot \vec{y}")
            .scale(0.5)
            .move_to(dot_product_gen.get_center())
        )
        analysis_freq_n = display_signal(
            get_cosine_func(freq=7), color=FREQ_DOMAIN_COLOR
        ).scale_to_fit_height(analysis_freq_with_dot_prod_0[1].height)
        dot_prod_n.next_to(analysis_freq_n, UP, buff=buff)
        analysis_freq_with_dot_prod_n = VGroup(dot_prod_n, analysis_freq_n)

        analysis_frequencies_group = (
            VGroup(
                analysis_freq_with_dot_prod_0,
                analysis_freq_with_dot_prod_1,
                analysis_freq_with_dot_prod_2,
                ellipses,
                analysis_freq_with_dot_prod_n,
            )
            .scale(0.9)
            .arrange(DOWN)
            .move_to(RIGHT * 3.5)
        )

        self.play(
            FadeOut(analysis_samples_vec),
            FadeOut(analysis_value),
            ReplacementTransform(
                analysis_freq_with_dot_prod, analysis_freq_with_dot_prod_0
            ),
        )
        self.wait()
        dot_prod_0.move_to(analysis_freq_with_dot_prod_0[0].get_center())
        self.play(Transform(analysis_freq_with_dot_prod_0[0], dot_prod_0))
        self.wait()

        self.play(
            LaggedStartMap(FadeIn, analysis_frequencies_group[1:]),
        )
        self.wait()

        self.define_matrix(
            time_domain_graph,
            signal_samples_vec,
            sample_value,
            analysis_frequencies_group,
        )

    def define_matrix(
        self,
        time_domain_graph,
        signal_samples_vec,
        sample_value,
        analysis_frequencies_group,
    ):
        self.play(
            FadeOut(time_domain_graph),
            FadeOut(signal_samples_vec),
            FadeOut(sample_value),
            analysis_frequencies_group.animate.move_to(LEFT * 4.5),
        )
        self.wait()

        matrix = self.get_matrix()

        column_vec = make_column_vector(
            ["y_0", "y_1", "y_2", r"\vdots", "y_{N - 1}"], scale=1
        )
        result_vec = make_column_vector(
            ["F_0", "F_1", "F_2", r"\vdots", "F_{M - 1}"], scale=1
        )
        equals = MathTex("=").scale(1.2)
        column_vec.next_to(matrix, RIGHT)
        matrix_vector_product = (
            VGroup(matrix, column_vec, equals, result_vec)
            .arrange(RIGHT)
            .scale(0.8)
            .move_to(RIGHT * 2.5)
        )
        self.play(FadeIn(matrix_vector_product))
        self.wait()

        self.play(matrix_vector_product.animate.shift(UP * 2))
        self.wait()

        prop_1 = Tex(
            r"1. $\vec{a}_k = \vec{y}$ (frequencies match) $\rightarrow F_k > 0$"
        )
        prop_2 = Tex(r"2. $\vec{y} = a_j \neq \vec{a}_k  \rightarrow F_k = 0$")
        prop_3 = Tex(r"3. time $\longleftrightarrow$ frequency")

        properties = VGroup(prop_1, prop_2, prop_3).arrange(
            DOWN, aligned_edge=LEFT, buff=0.5
        )
        properties.scale(0.8).next_to(matrix_vector_product, DOWN, buff=1)

        for prop in properties:
            self.play(FadeIn(prop))
            self.wait()

        invertibility_rect = SurroundingRectangle(prop_3, color=REDUCIBLE_YELLOW)
        orthogonality_rect = SurroundingRectangle(
            VGroup(prop_1, prop_2), color=REDUCIBLE_GREEN_LIGHTER
        )

        self.play(Write(invertibility_rect))
        invertibility = (
            Text("Invertibility", font=REDUCIBLE_FONT)
            .scale(0.6)
            .next_to(invertibility_rect, DOWN)
        )
        orthogonality = (
            Text("Orthogonality", font=REDUCIBLE_FONT)
            .scale(0.6)
            .next_to(orthogonality_rect, UP)
        )

        self.play(FadeIn(invertibility))
        self.wait()

        self.play(Write(orthogonality_rect))
        self.play(FadeIn(orthogonality))
        self.wait()

        implication = (
            MathTex(r"\Rightarrow M = N").scale(0.8).next_to(invertibility_rect, RIGHT)
        )
        self.play(Write(implication))
        self.wait()

    def get_matrix(self):
        row0 = self.get_row_tex(0)
        row1 = self.get_row_tex(1)
        row2 = self.get_row_tex(2)
        vdots = MathTex(r"\vdots")
        row7 = self.get_row_tex("M - 1")

        rows = VGroup(row0, row1, row2, vdots, row7).arrange(DOWN).move_to(DOWN * 2)
        bracket_pair = MathTex("[", "]")
        bracket_pair.scale(2)
        bracket_v_buff = MED_SMALL_BUFF
        bracket_h_buff = MED_SMALL_BUFF
        bracket_pair.stretch_to_fit_height(rows.height + 2 * bracket_v_buff)
        l_bracket, r_bracket = bracket_pair.split()
        l_bracket.next_to(rows, LEFT, bracket_h_buff)
        r_bracket.next_to(rows, RIGHT, bracket_h_buff)
        brackets = VGroup(l_bracket, r_bracket)

        return VGroup(brackets, rows)

    def get_row_tex(self, index):
        latex_text = r"a_{" + str(index) + r"}^T"
        text = MathTex(latex_text).scale(0.8)
        left_arrow = (
            Arrow(
                RIGHT * 2,
                ORIGIN,
                stroke_width=3,
                max_tip_length_to_length_ratio=0.15,
            )
            .next_to(text, LEFT)
            .set_color(WHITE)
        )
        right_arrow = (
            Arrow(
                ORIGIN,
                RIGHT * 2,
                stroke_width=3,
                max_tip_length_to_length_ratio=0.15,
            )
            .next_to(text, RIGHT)
            .set_color(WHITE)
        )
        return VGroup(left_arrow, text, right_arrow)


class IntroduceOrthogonalityEnd2(Scene):
    def construct(self):
        left_analysis_freq = (
            self.get_analysis_freq(0).scale(DEFAULT_SCALE).move_to(LEFT * 3.5 + UP * 2)
        )
        right_analysis_freq = (
            self.get_analysis_freq(0).scale(DEFAULT_SCALE).move_to(RIGHT * 3.5 + UP * 2)
        )

        self.play(FadeIn(left_analysis_freq), FadeIn(right_analysis_freq))
        self.wait()

        analysis_freq_matrix = get_cosine_dft_matrix(8)
        dot_product_matrix = np.dot(analysis_freq_matrix.T, analysis_freq_matrix)

        ideal_matrix = 1 * np.eye(8)
        A_T_A_text = MathTex("A^TA").scale(0.8)
        ideal_heat_map = get_heat_map_from_matrix(ideal_matrix)
        ideal_case = Text("Ideal Case", font=REDUCIBLE_FONT).scale(0.6)
        ideal_group = (
            VGroup(ideal_case, ideal_heat_map, A_T_A_text)
            .arrange(DOWN)
            .scale(0.8)
            .move_to(LEFT * 3.5 + DOWN * 1.5)
        )
        ideal_case.next_to(ideal_heat_map[0], UP, buff=SMALL_BUFF)
        A_T_A_text.next_to(ideal_heat_map[0], DOWN, buff=SMALL_BUFF)
        self.play(FadeIn(ideal_group))
        self.wait()

        actual = Text("Actual", font=REDUCIBLE_FONT).scale(0.6)

        heat_map = get_heat_map_from_matrix(dot_product_matrix)

        right_A_T_A_text = MathTex("A^TA").scale(0.8)
        actual_group = (
            VGroup(actual, heat_map, right_A_T_A_text)
            .arrange(DOWN)
            .scale(0.8)
            .move_to(RIGHT * 3.5 + DOWN * 1.5)
        )
        actual.next_to(heat_map[0], UP, buff=SMALL_BUFF)

        right_A_T_A_text.next_to(heat_map[0], DOWN, buff=SMALL_BUFF)

        empty_grid = get_grid(rows=8, cols=8, height=3, width=3)
        empty_grid.scale_to_fit_height(heat_map[0].height).move_to(
            heat_map[0].get_center()
        )

        self.play(
            FadeIn(heat_map[1]),
            FadeIn(empty_grid),
            FadeIn(right_A_T_A_text),
            FadeIn(actual),
        )
        self.wait()

        dot_product_tex = self.get_dot_prod(0, 0, dot_product_matrix[0][0])
        dot_product_tex.move_to(DOWN * 1.5)
        self.play(FadeIn(dot_product_tex))
        self.wait()
        actual_grid = heat_map[0]
        self.play(Transform(empty_grid[0][0], actual_grid[0][0]))
        self.wait()
        count = 0
        for row in range(dot_product_matrix.shape[0]):
            for col in range(dot_product_matrix.shape[1]):
                if row == 0 and col == 0:
                    continue
                new_left_analysis_freq = (
                    self.get_analysis_freq(row)
                    .scale(DEFAULT_SCALE)
                    .move_to(LEFT * 3.5 + UP * 2)
                )
                new_right_analysis_freq = (
                    self.get_analysis_freq(col)
                    .scale(DEFAULT_SCALE)
                    .move_to(RIGHT * 3.5 + UP * 2)
                )
                new_dot_product_text = self.get_dot_prod(
                    row, col, dot_product_matrix[row][col]
                ).move_to(DOWN * 1.5)
                self.play(
                    Transform(left_analysis_freq, new_left_analysis_freq),
                    Transform(right_analysis_freq, new_right_analysis_freq),
                    Transform(dot_product_tex, new_dot_product_text),
                    Transform(empty_grid[row][col], actual_grid[row][col])
                )
                if count < 8:
                    self.wait()
                else:
                    self.wait(1 / self.camera.frame_rate)

                count += 1

        bad_indices = [(1, 7), (2, 6), (3, 5)]
        highlight_rect = SurroundingRectangle(
            empty_grid[1][7], color=REDUCIBLE_CHARM, buff=0
        )
        self.play(FadeIn(highlight_rect))
        self.wait()
        for row, col in bad_indices:
            new_left_analysis_freq = (
                self.get_analysis_freq(row)
                .scale(DEFAULT_SCALE)
                .move_to(LEFT * 3.5 + UP * 2)
            )
            new_right_analysis_freq = (
                self.get_analysis_freq(col)
                .scale(DEFAULT_SCALE)
                .move_to(RIGHT * 3.5 + UP * 2)
            )

            for dot in new_left_analysis_freq[1][-1].set_color(REDUCIBLE_CHARM):
                dot.scale(1.1)
            for dot in new_right_analysis_freq[1][-1].set_color(REDUCIBLE_CHARM):
                dot.scale(1.1)
            new_dot_product_text = self.get_dot_prod(
                row, col, dot_product_matrix[row][col]
            ).move_to(DOWN * 1.5)
            new_highlight_rect = SurroundingRectangle(
                empty_grid[row][col], color=REDUCIBLE_CHARM, buff=0
            )
            self.play(
                Transform(left_analysis_freq, new_left_analysis_freq),
                Transform(right_analysis_freq, new_right_analysis_freq),
                Transform(dot_product_tex, new_dot_product_text),
                Transform(highlight_rect, new_highlight_rect),
            )
            self.wait()

        self.play(FadeOut(highlight_rect))
        self.wait()

        good_subset = SurroundingRectangle(
            VGroup(empty_grid[0][0], empty_grid[3][3]),
            color=REDUCIBLE_GREEN_LIGHTER,
            buff=0,
        )
        fade_indices = []
        for i in range(8):
            for j in range(8):
                if i < 4 and j < 4:
                    continue
                fade_indices.append((i, j))
        self.play(
            FadeIn(good_subset),
            *[empty_grid[i][j].animate.set_fill(opacity=0.4) for i, j in fade_indices],
        )
        self.wait()

        self.clear()

        self.show_actual_transform_animation()

    def show_actual_transform_animation(self):
        NUM_SAMPLES = 8
        title = (
            Text("Current Transform Issues", font=REDUCIBLE_FONT)
            .scale(0.8)
            .move_to(UP * 3.5)
        )

        time_signal_func = get_cosine_func(freq=1)
        cosine_dft_matrix = get_cosine_dft_matrix(NUM_SAMPLES)

        time_domain_graph = (
            display_signal(time_signal_func, num_points=NUM_SAMPLES)
            .scale(0.8)
            .move_to(LEFT * 3.5 + UP * 0)
        )

        fourier_rects = (
            get_fourier_rects_from_custom_matrix(
                time_signal_func,
                cosine_dft_matrix,
                n_samples=NUM_SAMPLES,
                full_spectrum=True,
            )
            .scale(1.2)
            .move_to(RIGHT * 3.5 + UP * 0)
        )

        self.play(Write(title), FadeIn(time_domain_graph), FadeIn(fourier_rects))
        self.wait()
        highlight_rect = None
        for i, freq in enumerate(range(1, 8)):
            if i == 0:
                continue
            time_signal_func = get_cosine_func(freq=freq)
            new_time_domain_graph = (
                display_signal(time_signal_func, num_points=NUM_SAMPLES)
                .scale(0.8)
                .move_to(LEFT * 3.5 + UP * 0)
            )

            new_fourier_rects = (
                get_fourier_rects_from_custom_matrix(
                    time_signal_func,
                    cosine_dft_matrix,
                    n_samples=NUM_SAMPLES,
                    full_spectrum=True,
                )
                .scale(1.2)
                .move_to(RIGHT * 3.5 + UP * 0)
            )

            if i == 5:
                highlight_rect = SurroundingRectangle(
                    new_fourier_rects[0][:4], buff=SMALL_BUFF, color=REDUCIBLE_YELLOW
                )
                self.play(FadeIn(highlight_rect))
                self.wait()

            self.play(
                Transform(time_domain_graph, new_time_domain_graph),
                Transform(fourier_rects, new_fourier_rects),
            )
            self.wait()

        self.play(
            FadeOut(time_domain_graph),
            FadeOut(fourier_rects),
            FadeOut(highlight_rect)
        )
        self.wait()

        analysis_freq_matrix = get_cosine_dft_matrix(8)
        dot_product_matrix = np.dot(analysis_freq_matrix.T, analysis_freq_matrix)

        heat_map = get_heat_map_from_matrix(dot_product_matrix)

        matrix = self.get_matrix()
        A_T_A_text = MathTex("A^TA").scale(0.8)
        A_T_A_text.next_to(heat_map[0], DOWN)

        heat_map_with_label = VGroup(heat_map, A_T_A_text)
        matrix_and_heat_map = VGroup(matrix, heat_map_with_label).arrange(RIGHT, buff=2)

        self.play(FadeIn(matrix_and_heat_map))
        self.wait()

        equal_rows = VGroup(
            MathTex(r"\vec{a}_1 = \vec{a}_7").scale(0.8),
            MathTex(r"\vec{a}_2 = \vec{a}_6").scale(0.8),
            MathTex(r"\vec{a}_3 = \vec{a}_5").scale(0.8)
        ).arrange(DOWN).move_to(DOWN * 3)

        self.play(
            FadeIn(equal_rows)
        )
        self.wait()

        non_invertible_text = Text("Non-invertible", font=REDUCIBLE_FONT).scale(0.7).next_to(title, DOWN, buff=0.5)
        self.play(
            FadeIn(non_invertible_text)
        )
        self.wait()


    def get_matrix(self):
        row0 = self.get_row_tex(0)
        row1 = self.get_row_tex(1)
        row2 = self.get_row_tex(2)
        vdots = MathTex(r"\vdots")
        row7 = self.get_row_tex(7)

        rows = VGroup(row0, row1, row2, vdots, row7).arrange(DOWN).move_to(DOWN * 2)
        bracket_pair = MathTex("[", "]")
        bracket_pair.scale(2)
        bracket_v_buff = MED_SMALL_BUFF
        bracket_h_buff = MED_SMALL_BUFF
        bracket_pair.stretch_to_fit_height(rows.height + 2 * bracket_v_buff)
        l_bracket, r_bracket = bracket_pair.split()
        l_bracket.next_to(rows, LEFT, bracket_h_buff)
        r_bracket.next_to(rows, RIGHT, bracket_h_buff)
        brackets = VGroup(l_bracket, r_bracket)

        return VGroup(brackets, rows)

    def get_row_tex(self, index):
        latex_text = r"a_{" + str(index) + r"}^T"
        text = MathTex(latex_text).scale(0.8)
        left_arrow = (
            Arrow(
                RIGHT * 2,
                ORIGIN,
                stroke_width=3,
                max_tip_length_to_length_ratio=0.15,
            )
            .next_to(text, LEFT)
            .set_color(WHITE)
        )
        right_arrow = (
            Arrow(
                ORIGIN,
                RIGHT * 2,
                stroke_width=3,
                max_tip_length_to_length_ratio=0.15,
            )
            .next_to(text, RIGHT)
            .set_color(WHITE)
        )
        return VGroup(left_arrow, text, right_arrow)

    def get_analysis_freq(self, freq):
        analysis_freq_func = get_cosine_func(freq=freq)
        analysis_freq_graph = display_signal(
            analysis_freq_func, color=FREQ_DOMAIN_COLOR
        )

        sample_value = MathTex(r"\vec{a}_" + str(freq)).scale(0.8)
        graph_notation = VGroup(sample_value, analysis_freq_graph).arrange(DOWN)
        return graph_notation

    def get_dot_prod(self, row, col, value, threshold=True):
        if np.isclose(value, 0):
            value = 0.0
        value_string = f"{value:.1f}"
        dot_product_tex = Tex(
            r"$\vec{a}_" + str(row) + r"^T \vec{a}_" + str(col) + r" =$ " + value_string
        ).scale(0.8)
        return dot_product_tex


class TestCases(Scene):
    def construct(self):
        NUM_SAMPLES = 16
        time_signal_func = get_cosine_func(freq=DEFAULT_FREQ)
        time_domain_graph = display_signal(time_signal_func, num_points=NUM_SAMPLES)
        (
            time_axis,
            sampled_points_vert_lines,
            graph,
            sampled_points_dots,
        ) = time_domain_graph
        self.play(FadeIn(time_axis), Write(graph))
        self.wait()
        sample_value = MathTex(
            r"y_k = A \cos \left( \frac{2 \pi k f}{N} \right) + b"
        ).scale(0.7)

        signal_math = VGroup(sample_value).arrange(DOWN).next_to(time_domain_graph, UP)

        num_samples_text = MathTex("N = 16").scale(0.7)
        num_samples_text.next_to(time_domain_graph, DOWN)

        self.play(FadeIn(signal_math))
        self.wait()

        self.play(
            LaggedStartMap(Write, sampled_points_vert_lines),
            LaggedStartMap(GrowFromCenter, sampled_points_dots),
        )
        self.wait()
        time_domain_graph_with_math = VGroup(
            signal_math, time_domain_graph, num_samples_text
        )

        self.play(FadeIn(num_samples_text))
        self.wait()

        self.play(time_domain_graph_with_math.animate.scale(0.8).shift(UP * 1.5))
        self.wait()

        cosine_dft_matrix = get_cosine_dft_matrix(NUM_SAMPLES)
        fourier_rects = (
            get_fourier_rects_from_custom_matrix(
                time_signal_func,
                cosine_dft_matrix,
                n_samples=NUM_SAMPLES,
                full_spectrum=True,
            )
            .scale(1.2)
            .move_to(DOWN * 2)
        )

        self.play(FadeIn(fourier_rects))
        self.wait()

        surround_rect = SurroundingRectangle(
            fourier_rects[0][: NUM_SAMPLES // 2], color=REDUCIBLE_YELLOW
        )
        self.play(Write(surround_rect))
        self.wait()

        vt_amplitude = ValueTracker(1)
        vt_frequency = ValueTracker(DEFAULT_FREQ)
        vt_b = ValueTracker(0)

        half_fourier_rects, reference_label = (
            get_fourier_rects_from_custom_matrix(
                time_signal_func,
                cosine_dft_matrix,
                n_samples=NUM_SAMPLES,
                full_spectrum=False,
            )
            .scale(1.2)
            .move_to(fourier_rects.get_center())
        )

        half_fourier_rects_group = VGroup(half_fourier_rects, reference_label)

        def get_changing_fourier_rects():
            time_signal_func = get_cosine_func(
                amplitude=vt_amplitude.get_value(),
                freq=vt_frequency.get_value(),
                b=vt_b.get_value(),
            )
            changing_half_fourier_rects, frequency_label = (
                get_fourier_rects_from_custom_matrix(
                    time_signal_func,
                    cosine_dft_matrix,
                    n_samples=NUM_SAMPLES,
                    full_spectrum=False,
                )
                .scale(1.2)
                .move_to(half_fourier_rects, aligned_edge=DOWN)
                .shift(DOWN * 0.5)
            )
            return changing_half_fourier_rects

        self.play(
            FadeTransform(fourier_rects, half_fourier_rects_group),
            FadeOut(surround_rect),
        )
        self.wait()

        changing_half_fourier_rects = always_redraw(get_changing_fourier_rects)

        def get_changing_signal():
            # TODO might need to unpack the display signal function here
            # get the axes and vertical lines, move them to the center of reference axes and the vertical lines
            # then plot the signal with reference to the axes and vertical lines. Return that signal and dots.
            analysis_freq_func = get_cosine_func(
                amplitude=vt_amplitude.get_value(),
                freq=vt_frequency.get_value(),
                b=vt_b.get_value(),
            )
            new_signal = display_signal(analysis_freq_func, num_points=16)
            new_signal.scale(0.8).move_to(time_domain_graph.get_center())
            return VGroup(new_signal[2], new_signal[3])

        def change_text_redraw():
            v_freq = f"{vt_frequency.get_value():.2f}"
            v_amplitude = f"{vt_amplitude.get_value():.2f}"
            v_b = f"{vt_b.get_value():.2f}"

            freq_eq = MathTex(r"f = ")
            tex_frequency_val = Text(
                v_freq,
                font=REDUCIBLE_MONO,
            ).scale(0.8)
            tex_freq = VGroup(freq_eq, tex_frequency_val).arrange(RIGHT)

            amplitude_tex = MathTex("A = ")
            amplitude_val = Text(
                v_amplitude,
                font=REDUCIBLE_MONO,
            ).scale(0.8)
            tex_amplitude = VGroup(amplitude_tex, amplitude_val).arrange(RIGHT)

            b_eq = MathTex("b = ")
            b_val = Text(
                v_b,
                font=REDUCIBLE_MONO,
            ).scale(0.8)
            tex_b = VGroup(b_eq, b_val).arrange(RIGHT)

            text_group = (
                VGroup(tex_amplitude, tex_freq, tex_b)
                .arrange(DOWN, aligned_edge=LEFT)
                .scale(0.6)
                .to_corner(UL)
            )
            return text_group

        changing_time_domain_graph = always_redraw(get_changing_signal)

        text_group = always_redraw(change_text_redraw)
        self.remove(time_domain_graph[2], time_domain_graph[3])
        self.add(changing_time_domain_graph)
        self.remove(half_fourier_rects)
        self.add(changing_half_fourier_rects)

        self.play(
            FadeIn(text_group),
        )
        self.wait()

        new_text_group_with_math = (
            VGroup(
                text_group.copy(),
                signal_math.copy().scale(0.7 / 0.8),
                num_samples_text.copy().scale(0.7 / 0.8),
            )
            .arrange(DOWN, aligned_edge=LEFT)
            .to_corner(UL)
        )

        self.play(
            Transform(signal_math, new_text_group_with_math[1]),
            Transform(num_samples_text, new_text_group_with_math[2]),
        )
        self.wait()

        self.play(vt_amplitude.animate.set_value(0.5), run_time=2)
        self.wait()
        self.play(vt_amplitude.animate.set_value(1.5), run_time=4)
        self.wait()
        self.play(vt_amplitude.animate.set_value(1), run_time=2)
        self.wait()

        self.play(vt_frequency.animate.set_value(1), run_time=2)

        self.play(vt_frequency.animate.set_value(3), run_time=4)

        self.play(vt_frequency.animate.set_value(4), run_time=2)
        self.play(vt_frequency.animate.set_value(5), run_time=2)
        self.play(vt_frequency.animate.set_value(6), run_time=2)
        self.play(vt_frequency.animate.set_value(7), run_time=2)
        self.play(vt_frequency.animate.set_value(2), run_time=4)
        self.wait()

        self.play(vt_b.animate.set_value(0.5), run_time=2)
        self.wait()
        self.play(vt_b.animate.set_value(0), run_time=2)
        self.wait()

        self.play(
            FadeOut(text_group),
            FadeOut(
                changing_time_domain_graph, time_domain_graph[0], time_domain_graph[1]
            ),
            FadeOut(changing_half_fourier_rects),
            FadeOut(signal_math),
            FadeOut(num_samples_text),
            FadeOut(reference_label),
        )
        self.wait()
        f1 = get_cosine_func(freq=1, amplitude=0.4)
        freq_one = display_signal(f1, num_points=16)
        f2 = get_cosine_func(freq=4, amplitude=0.6)
        freq_four = display_signal(
            f2,
            num_points=16,
            color=REDUCIBLE_GREEN_LIGHTER,
        )
        f3 = get_cosine_func(freq=5, amplitude=0.1)
        freq_five = display_signal(
            f3,
            num_points=16,
            color=REDUCIBLE_ORANGE,
        )

        frequencies = (
            VGroup(freq_one, freq_four, freq_five)
            .scale(0.6)
            .arrange(RIGHT)
            .move_to(UP * 2)
        )

        self.play(FadeIn(frequencies))
        self.wait()

        a_f_label_1 = VGroup(MathTex("A = 0.4, f = 1")).scale(0.6).next_to(freq_one, UP)
        a_f_label_2 = (
            VGroup(MathTex("A = 0.6, f = 4")).scale(0.6).next_to(freq_four, UP)
        )
        a_f_label_3 = (
            VGroup(MathTex("A = 0.1, f = 5")).scale(0.6).next_to(freq_five, UP)
        )

        f_label_1 = MathTex("F_1").scale(0.7).next_to(freq_one, DOWN)
        f_label_2 = MathTex("F_2").scale(0.7).next_to(freq_four, DOWN)
        f_label_3 = MathTex("F_3").scale(0.7).next_to(freq_five, DOWN)

        self.play(
            FadeIn(a_f_label_1),
            FadeIn(a_f_label_2),
            FadeIn(a_f_label_3),
            FadeIn(f_label_1),
            FadeIn(f_label_2),
            FadeIn(f_label_3),
        )
        self.wait()

        separate_group = VGroup(
            a_f_label_1,
            a_f_label_2,
            a_f_label_3,
            f_label_1,
            f_label_2,
            f_label_3,
            freq_one,
            freq_four,
            freq_five,
        )

        combined_func = lambda x: f1(x) + f2(x) + f3(x)

        combined_func_graph = display_signal(
            combined_func, num_points=16, color=REDUCIBLE_CHARM
        )
        combined_func_graph.scale(0.7).next_to(f_label_2, DOWN)

        combined_func_label = (
            MathTex("F_1 + F_2 + F_3").scale(0.7).next_to(combined_func_graph, DOWN)
        )

        self.play(FadeIn(combined_func_graph), FadeIn(combined_func_label))
        self.wait()

        fourier_rects, _ = get_fourier_rects_from_custom_matrix(
            combined_func, get_cosine_dft_matrix(16)
        )
        fourier_rects.next_to(combined_func_label, DOWN)

        self.play(FadeIn(fourier_rects))
        self.wait()

        together_group = VGroup(combined_func_graph, combined_func_label, fourier_rects)
        self.play(
            separate_group.animate.scale(0.8).shift(UP * 0.5),
            together_group.animate.scale(0.8).shift(DOWN * 0.5),
        )
        self.wait()

        fourier_rects_copy = fourier_rects.copy()
        f_rects_1, _ = get_fourier_rects_from_custom_matrix(
            f1, get_cosine_dft_matrix(16)
        )
        f_rects_2, _ = get_fourier_rects_from_custom_matrix(
            f2, get_cosine_dft_matrix(16)
        )
        f_rects_3, _ = get_fourier_rects_from_custom_matrix(
            f3, get_cosine_dft_matrix(16)
        )
        f_rects_1.scale_to_fit_width(fourier_rects_copy.width)
        f_rects_2.scale_to_fit_width(fourier_rects_copy.width)
        f_rects_3.scale_to_fit_width(fourier_rects_copy.width)

        f_rects_1.next_to(f_label_1, DOWN).shift(DOWN * SMALL_BUFF * 1.3)
        f_rects_2.next_to(f_label_2, DOWN)
        f_rects_3.next_to(f_label_3, DOWN).shift(DOWN * SMALL_BUFF * 3.2)

        summed_transform = MathTex(
            "T(F_1 + F_2 + F_3) = T(F_1) + T(F_2) + T(F_3)"
        ).scale(0.8)
        summed_transform.next_to(f_rects_2, DOWN)

        self.play(FadeIn(summed_transform))
        self.wait()

        self.play(LaggedStartMap(FadeIn, [f_rects_1, f_rects_2, f_rects_3]))
        self.wait()


class TestCircleTheory(Scene):
    def construct(self):
        dots = VGroup()
        for phase in np.linspace(0, 2 * PI, 1000):
            cosine_func = get_cosine_func(freq=DEFAULT_FREQ, phase=phase)
            analysis_cosine_func = get_cosine_func(freq=DEFAULT_FREQ)
            analysis_sin_func = get_sine_func(freq=DEFAULT_FREQ)
            sampled_points = get_sampled_points(cosine_func)
            sampled_points_cosine_analysis = get_sampled_points(analysis_cosine_func)
            sampled_points_sin_analysis = get_sampled_points(analysis_sin_func)

            x_coord = (
                np.dot(sampled_points, sampled_points_cosine_analysis) * 0.5
            )  # 0.5 to scale down circle
            y_coord = (
                np.dot(sampled_points, sampled_points_sin_analysis) * 0.5
            )  # 0.5 to scale down circle
            dot = Dot().move_to(RIGHT * x_coord + UP * y_coord)
            dots.add(dot)
        self.play(FadeIn(dots))
        self.wait()

class BlankScreen(Scene):
    def construct(self):
        self.wait()
