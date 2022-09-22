import sys
import copy

from manim import *
from wavelet_utils import *

### THIS IS A WORKAROUND FOR NOW
### IT REQUIRES RUNNING MANIM FROM INSIDE DIRECTORY VIDEO-DEV
sys.path.insert(1, "common/")

from manim import *
from reducible_colors import *
from functions import *
import itertools
import pywt


class WaveletTester(Scene):
    def construct(self):
        T_MAX = 6
        time_axis, time_graph = self.plot_time_domain(get_sine_func(), t_max=T_MAX)
        time_axis_and_graph = VGroup(time_axis, time_graph)
        freq_axis, freq_graph = self.plot_fourier_transform(
            get_sine_func(), t_max=T_MAX
        )

        freq_axis_and_graph = VGroup(freq_axis, freq_graph)

        VGroup(time_axis_and_graph, freq_axis_and_graph).arrange(DOWN)

        self.play(FadeIn(time_axis))
        self.wait()

        self.play(FadeIn(time_graph))
        self.wait()

        self.play(
            FadeIn(freq_axis),
        )
        self.wait()

        self.play(FadeIn(freq_graph))
        self.wait()

        vertical_bars = get_fourier_vertical_lines(
            freq_axis,
            get_sine_func(),
            t_min=0,
            t_max=T_MAX,
            f_min=0,
            f_max=6,
        )

        self.play(FadeIn(vertical_bars))

        # self.plot_function(get_sine_func(), 600, 1 / 800)

    def plot_function(self, func, num_samples, time_between_samples):
        x_range = [0, num_samples * time_between_samples, time_between_samples]
        ax = Axes(
            x_range=x_range,
            y_range=[-4, 1, 4],
            y_length=3,
            x_length=6,
            tips=False,
            axis_config={"include_numbers": False, "include_ticks": False},
        ).move_to(UP * 2)
        graph = ax.plot(func).set_color(REDUCIBLE_VIOLET).set_stroke(width=2)

        self.add(ax)
        self.play(Write(graph))
        self.wait()
        sampled_x_points = np.linspace(
            0, num_samples * time_between_samples, num_samples
        )
        y_points = func(sampled_x_points)

        fft_values = np.fft.rfft(y_points)
        fft_freq_range = np.linspace(
            0, 1 / (2 * time_between_samples), num_samples // 2
        )
        fft_magnitudes_half_spectrum = (
            2 / num_samples * np.abs(fft_values[: num_samples // 2])
        )

        x_range = [0, float(max(fft_freq_range))]
        y_range = [0, float(max(fft_magnitudes_half_spectrum))]
        print(x_range, y_range)
        ax_down = Axes(
            x_range=x_range,
            y_range=y_range,
            x_length=8,
            y_length=3,
            tips=False,
            axis_config={"include_numbers": False, "include_ticks": False},
        ).move_to(DOWN * 2)
        graph = (
            VGroup()
            .set_points_smoothly(
                *[
                    [
                        ax_down.coords_to_point(x, y)
                        for x, y in zip(fft_freq_range, fft_magnitudes_half_spectrum)
                    ]
                ]
            )
            .set_color(REDUCIBLE_YELLOW)
        )

        self.play(FadeIn(ax_down))
        self.wait()

        self.play(FadeIn(graph))
        self.wait()

    def plot_fourier_transform(
        self, time_func, t_min=0, t_max=10, f_min=0, f_max=6, f_step=1
    ):
        freq_axes = get_freq_axes(x_range=[f_min, f_max, f_step])
        graph = get_fourier_graph(
            freq_axes, time_func, t_min=t_min, t_max=t_max, f_min=f_min, f_max=f_max
        )

        return freq_axes, graph

    def plot_time_domain(
        self, time_func, t_min=0, t_max=10, t_step=1, y_min=-3, y_max=3, y_step=1
    ):
        time_axis = get_time_axes(
            x_range=[t_min, t_max, t_step], y_range=[y_min, y_max, y_step]
        )
        graph = time_axis.plot(time_func).set_color(REDUCIBLE_VIOLET)

        return time_axis, graph

    def get_vertical_bars_of_samples(
        self, ax, freq_graph, num_samples, f_min=0, f_max=6, color=REDUCIBLE_VIOLET
    ):
        freq_bins = np.linspace(f_min, f_max, num_samples)
        x_axis_points = [ax.x_axis.n2p(p) for p in freq_bins]
        graph_points = [
            ax.coords_to_point(x, freq_graph.underlying_function(x)) for x in freq_bins
        ]
        vertical_lines = [
            Line(start_point, y_point).set_stroke(color=color, width=4)
            for start_point, y_point in zip(x_axis_points, graph_points)
        ]
        return VGroup(*vertical_lines)


class ChirpSignalTests(WaveletTester):
    def construct(self):
        T_MAX = 18
        time_axis = get_time_axes(x_range=[0, T_MAX, 0.5])
        graph = time_axis.plot(chirp_piecewise).set_color(REDUCIBLE_YELLOW)
        time_axis_and_graph = VGroup(time_axis, graph)
        # self.play(FadeIn(time_axis))
        # self.wait()

        # self.play(FadeIn(graph))
        # self.wait()

        freq_axis, freq_graph = self.plot_fourier_transform(
            chirp_piecewise, t_max=T_MAX
        )

        freq_axis_and_graph = VGroup(freq_axis, freq_graph)

        VGroup(time_axis_and_graph, freq_axis_and_graph).arrange(DOWN)

        self.play(FadeIn(time_axis_and_graph))
        self.wait()

        self.play(FadeIn(freq_axis_and_graph))
        self.wait()
        # num_samples = 1000
        # time_signal_x = np.linspace(0, 12, num_samples)
        # time_signal_y = np.vectorize(chirp_piecewise)(time_signal_x)
        # print(time_signal_y)
        # cwt = pywt.cwt(time_signal_y, [])


class WaveletGraph3D(ThreeDScene):
    def construct(self):
        # self.set_camera_orientation(theta=70 * DEGREES, phi=80 * DEGREES)

        # Define signal
        fs = 128
        sampling_period = 1 / fs
        t = np.linspace(0, 18, 2 * fs)
        signal = np.vectorize(chirp_piecewise)(t)
        scales = np.arange(1, 100)
        cwt_coefficients, frequencies = pywt.cwt(
            signal, scales, "morl", sampling_period=sampling_period
        )
        print(scales)
        print(frequencies)

        print(cwt_coefficients.shape)
        print(scales.shape, t.shape)

        axes, surface = self.get_cwt_surface(signal, scales, t, cwt_coefficients)
        scales = Text("Scales", font=REDUCIBLE_MONO).next_to(axes.x_axis, DOWN)
        translations = Text("T", font=REDUCIBLE_MONO).next_to(axes.y_axis, LEFT)
        self.add(axes, surface, scales, translations)
        self.wait()

        heat_map = self.get_heat_map(cwt_coefficients).shift(IN * 1 + RIGHT * 4)
        self.add(heat_map)
        self.wait()

        origin_heat_map = Text("(0, 0)", font=REDUCIBLE_MONO).scale(0.2)
        origin_heat_map.next_to(heat_map[0][0], UL, buff=SMALL_BUFF)

        upper_right_corner = (
            Text("(0, 99)", font=REDUCIBLE_MONO)
            .scale(0.2)
            .next_to(heat_map[0][98], UR, buff=SMALL_BUFF)
        )

        lower_left_corner = (
            Text("(256, 0)", font=REDUCIBLE_MONO)
            .scale(0.2)
            .next_to(heat_map[255][0], DL, buff=SMALL_BUFF)
        )

        lower_right_corner = (
            Text("(256, 99)", font=REDUCIBLE_MONO)
            .scale(0.2)
            .next_to(heat_map[255][98], DR, buff=SMALL_BUFF)
        )

        self.add(
            origin_heat_map, upper_right_corner, lower_left_corner, lower_right_corner
        )
        self.wait()

    def get_heat_map(self, wavelet_coefficients, height=5, width=5):
        cell_width = width / wavelet_coefficients.shape[0]
        cell_height = height / wavelet_coefficients.shape[1]
        cells = [
            [
                Rectangle(height=cell_height, width=cell_width)
                for i in range(wavelet_coefficients.shape[0])
            ]
            for j in range(wavelet_coefficients.shape[1])
        ]

        min_color = REDUCIBLE_PURPLE
        max_color = REDUCIBLE_YELLOW
        coefficient_range = np.max(wavelet_coefficients) - np.min(wavelet_coefficients)
        for j in range(len(cells)):
            for i in range(len(cells[0])):
                alpha = (
                    wavelet_coefficients[i][j] - np.min(wavelet_coefficients)
                ) / coefficient_range
                print(alpha)
                cells[j][i].set_color(interpolate_color(min_color, max_color, alpha))

        grid = VGroup(*[VGroup(*row).arrange(RIGHT, buff=0) for row in cells]).arrange(
            UP, buff=0
        )

        return grid

    def cwt(self, s, t, s_values, t_values, cwt_coefficients):
        def get_upper_lower(lst, x):
            upper_index = 0
            for i, item in enumerate(lst):
                if item >= x:
                    upper_index = i
                    break
            if upper_index == 0:
                return 0, 1
            return upper_index - 1, upper_index

        def bilinear_interpolation(x, y, points):
            """Interpolate (x,y) from values associated with four points.

            The four points are a list of four triplets:  (x, y, value).
            The four points can be in any order.  They should form a rectangle.

                >>> bilinear_interpolation(12, 5.5,
                ...                        [(10, 4, 100),
                ...                         (20, 4, 200),
                ...                         (10, 6, 150),
                ...                         (20, 6, 300)])
                165.0

            """
            # See formula at:  http://en.wikipedia.org/wiki/Bilinear_interpolation

            points = sorted(points)  # order points by x, then by y
            (x1, y1, q11), (_x1, y2, q12), (x2, _y1, q21), (_x2, _y2, q22) = points
            if x1 != _x1 or x2 != _x2 or y1 != _y1 or y2 != _y2:
                raise ValueError("points do not form a rectangle")
            if not x1 <= x <= x2 or not y1 <= y <= y2:
                raise ValueError("(x, y) not within the rectangle")

            return (
                q11 * (x2 - x) * (y2 - y)
                + q21 * (x - x1) * (y2 - y)
                + q12 * (x2 - x) * (y - y1)
                + q22 * (x - x1) * (y - y1)
            ) / ((x2 - x1) * (y2 - y1))

        # print(s_values, s)
        # print(t_values, t)
        s_lower, s_upper = get_upper_lower(s_values, s)
        t_lower, t_upper = get_upper_lower(t_values, t)
        # print(s_lower, s_upper, t_lower, t_upper)

        points = [
            (s_values[s_lower], t_values[t_lower], cwt_coefficients[s_lower][t_lower]),
            (s_values[s_lower], t_values[t_upper], cwt_coefficients[s_lower][t_upper]),
            (s_values[s_upper], t_values[t_lower], cwt_coefficients[s_upper][t_lower]),
            (s_values[s_upper], t_values[t_upper], cwt_coefficients[s_upper][t_upper]),
        ]

        return [s, t, bilinear_interpolation(s, t, points)]

    def get_cwt_surface(
        self,
        signal,
        scales,
        translations,
        coefficients,
    ):
        axes = ThreeDAxes(
            x_range=[min(scales), max(scales)],
            y_range=[min(translations), max(translations)],
            z_range=[0, np.max(coefficients)],
            x_length=5,
            y_length=5,
            z_length=5,
            tips=False,
            axis_config={"include_ticks": False},
        ).shift(IN * 1 + LEFT * 2)

        surface = Surface(
            lambda s, t: axes.c2p(*self.cwt(s, t, scales, translations, coefficients)),
            u_range=[min(scales), max(scales)],
            v_range=[min(translations), max(translations)],
            checkerboard_colors=[REDUCIBLE_PURPLE],
            fill_opacity=0.5,
            resolution=32,
            stroke_color=REDUCIBLE_YELLOW,
            stroke_width=2,
        )

        return axes, surface
