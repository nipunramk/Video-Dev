import sys

### THIS IS A WORKAROUND FOR NOW
### IT REQUIRES RUNNING MANIM FROM INSIDE DIRECTORY VIDEO-DEV
sys.path.insert(1, "common/")


import matplotlib.pyplot as plt
import numpy as np
from manim import *
from reducible_colors import *
from functions import *

NUM_SAMPLES_FOR_FFT = 10000
DEFAULT_AXIS_CONFIG = {"include_numbers": False, "include_ticks": False}
TIME_DOMAIN_COLOR = REDUCIBLE_YELLOW
FREQ_DOMAIN_COLOR = REDUCIBLE_VIOLET


def get_dft_matrix(N):
    """
    Returns N x N DFT Matrix
    """
    return np.fft.fft(np.eye(N))


def get_cosine_dft_matrix(N):
    """
    Returns real component of N x N DFT Matrix
    Essentially only the cosine components
    """
    return np.real(get_dft_matrix(N))


def get_sin_dft_matrix(N):
    """
    Returns imaginary component of N x N DFT Matrix
    Essentially only the sin components
    TODO: need to account for standard DFT sign convention
    """
    return np.imag(get_dft_matrix(N))


def apply_matrix_transform(signal, matrix):
    return np.dot(matrix, signal)


def get_time_axis(
    x_range=[0, 6, 0.5],
    y_range=[-4, 4, 1],
    x_length=7,
    y_length=3,
    tips=False,
    axis_config=DEFAULT_AXIS_CONFIG,
):

    return Axes(
        x_range=x_range,
        y_range=y_range,
        x_length=x_length,
        y_length=y_length,
        tips=tips,
        axis_config=axis_config,
    )


def get_freq_axes(
    x_range=[0, 10, 1],
    y_range=[-1, 1, 0.5],
    x_length=7,
    y_length=3,
    tips=False,
    axis_config=DEFAULT_AXIS_CONFIG,
):
    return Axes(
        x_range=x_range,
        y_range=y_range,
        x_length=x_length,
        y_length=y_length,
        tips=tips,
        axis_config=axis_config,
    )


def plot_time_domain(
    time_func,
    t_min=0,
    t_max=10,
    t_step=1,
    y_min=-1,
    y_max=1,
    y_step=0.1,
    color=TIME_DOMAIN_COLOR,
):
    time_axis = get_time_axis(
        x_range=[t_min, t_max, t_step],
        y_range=[y_min, y_max, y_step],
    )
    graph = time_axis.plot(time_func).set_color(color)

    return time_axis, graph


def get_sum_functions(*time_functions):
    """
    Returns a time function corresponding to the sum of
    all the time functions passed in
    """
    return lambda x: sum([f(x) for f in time_functions])


def get_prod_functions(*time_functions):
    return lambda x: np.prod([f(x) for f in time_functions])


def get_cosine_func(amplitude=1, freq=1, phase=0, b=0):
    return lambda x: amplitude * np.cos(freq * x + phase) + b


def get_sine_func(amplitude=1, freq=1, phase=0, b=0):
    return lambda x: amplitude * np.sin(freq * x + phase) + b


def get_sampled_coords(graph, x_min=0, x_max=2 * PI, num_points=8):
    func = graph.underlying_function
    x_coords = np.linspace(x_min, x_max, num=num_points, endpoint=False)
    y_coords = func(x_coords)
    return x_coords, y_coords


def get_sampled_points(func, x_min=0, x_max=2 * PI, num_points=8):
    x_coords = np.linspace(x_min, x_max, num=num_points, endpoint=False)
    return func(x_coords)


def inner_prod(time_domain_func, freq_func, x_min=0, x_max=2 * PI, num_points=8):
    x_coords = np.linspace(x_min, x_max, num=num_points, endpoint=False)
    y_coords_time = time_domain_func(x_coords)
    y_coords_freq = freq_func(x_coords)
    return np.dot(y_coords_time, y_coords_freq)


def get_sampled_dots(
    graph, axes, x_min=0, x_max=2 * PI, num_points=8, radius=DEFAULT_DOT_RADIUS, color=REDUCIBLE_YELLOW
):
    x_coords, y_coords = get_sampled_coords(
        graph, x_min=x_min, x_max=x_max, num_points=num_points
    )
    coord_dots = [
        Dot(radius=radius).move_to(axes.coords_to_point(x_coord, y_coord))
        for x_coord, y_coord in zip(x_coords, y_coords)
    ]
    return VGroup(*coord_dots).set_color(color)


def get_vertical_bars_for_samples(
    graph, axes, x_min=0, x_max=2 * PI, num_points=8, color=REDUCIBLE_YELLOW
):
    x_coords, y_coords = get_sampled_coords(
        graph, x_min=x_min, x_max=x_max, num_points=num_points
    )
    x_axis_points = [axes.x_axis.n2p(p) for p in x_coords]
    graph_points = [axes.coords_to_point(x, y) for x, y in zip(x_coords, y_coords)]
    vertical_lines = [
        Line(start_point, y_point).set_stroke(color=color, width=4)
        for start_point, y_point in zip(x_axis_points, graph_points)
    ]
    return VGroup(*vertical_lines)


def display_signal(time_signal_func, color=TIME_DOMAIN_COLOR, num_points=8):
    time_axis, graph = plot_time_domain(time_signal_func, t_max=2 * PI, color=color)
    sampled_points_dots = get_sampled_dots(graph, time_axis, num_points=num_points)
    sampled_points_vert_lines = get_vertical_bars_for_samples(
        graph, time_axis, num_points=num_points
    )
    return VGroup(time_axis, graph, sampled_points_dots, sampled_points_vert_lines)


def get_fourier_line_chart(
    axes,
    time_func,
    t_min=0,
    t_max=10,
    f_min=0,
    f_max=10,
    n_samples=NUM_SAMPLES_FOR_FFT,
    complex_to_real_func=lambda z: z.real,
    color=FREQ_DOMAIN_COLOR,
):
    time_range = float(t_max - t_min)
    time_step_size = time_range / n_samples
    time_samples = np.vectorize(time_func)(np.linspace(t_min, t_max, n_samples))
    fft_output = np.fft.fft(time_samples)
    frequencies = np.linspace(0.0, n_samples / (2.0 * time_range), n_samples // 2)

    print(time_range, n_samples)
    print(min(frequencies), max(frequencies))

    graph = VMobject()
    graph.set_points_smoothly(
        [
            axes.coords_to_point(
                x,
                np.abs(y) / n_samples,
            )
            for x, y in zip(frequencies, fft_output[: n_samples // 2])
            if x <= f_max + 0.1
        ]
    )
    graph.set_color(color)
    graph.underlying_function = lambda f: axes.y_axis.point_to_number(
        graph.point_from_proportion((f - f_min) / (f_max - f_min))
    )
    return graph


def get_fourier_bar_chart(
    time_func,
    t_min=0,
    t_max=10,
    f_min=0,
    f_max=10,
    n_samples=NUM_SAMPLES_FOR_FFT,
    bar_width=0.2,
    height_scale=1,
    color=FREQ_DOMAIN_COLOR,
    full_spectrum=False,
):

    spectrum_selection = n_samples if full_spectrum else n_samples // 2

    time_range = float(t_max - t_min)
    time_samples = np.vectorize(time_func)(np.linspace(t_min, t_max, n_samples))
    fft_output = np.fft.fft(time_samples)
    frequencies = np.linspace(0.0, n_samples / (2.0 * time_range), spectrum_selection)

    graph = VGroup()

    for x, y in zip(frequencies, fft_output[:spectrum_selection]):
        if x <= f_max + 0.1:
            rect = (
                Rectangle(height=height_scale * np.abs(y) / n_samples, width=bar_width)
                .set_color(color)
                .set_fill(color, opacity=1)
                .set_stroke(width=1)
            )
            graph.add(rect)

    graph.arrange(RIGHT, buff=0.1, aligned_edge=DOWN)
    return graph


def get_vertical_dashed_lines_for_samples(
    graph, axes, x_min=0, x_max=2 * PI, num_points=8, color=REDUCIBLE_YELLOW
):
    y_min, y_max, _ = axes.y_range
    x_coords, _ = get_sampled_coords(
        graph, x_min=x_min, x_max=x_max, num_points=num_points
    )
    line_height = (axes.y_axis.n2p(y_max) - axes.y_axis.n2p(y_min))[1]
    x_axis_points = [axes.x_axis.n2p(p) for p in x_coords]

    vertical_lines = [
        DashedLine(ORIGIN, UP * line_height)
        .set_stroke(color=color, width=4)
        .move_to(x_point)
        for x_point in x_axis_points
    ]
    return VGroup(*vertical_lines)


def make_column_vector(values, v_buff=0.6, scale=0.6):
    integer_values = []
    for value in values:
        if isinstance(value, str):
            integer_values.append(value)
        else:
            integer_values.append(int(value))
    vector = Matrix(
        [[value] for value in integer_values],
        v_buff=v_buff,
        element_alignment_corner=DOWN,
    )
    return vector.scale(scale)


def make_row_vector(values, h_buff=0.6, scale=0.6):
    integer_values = []
    for value in values:
        if isinstance(value, str):
            integer_values.append(value)
        else:
            integer_values.append(int(value))
    vector = Matrix([[value for value in integer_values]], h_buff=h_buff)
    return vector.scale(scale)


def display_signal(time_signal_func, color=TIME_DOMAIN_COLOR, num_points=8):
    time_axis, graph = plot_time_domain(time_signal_func, t_max=2 * PI, color=color)
    sampled_points_dots = get_sampled_dots(graph, time_axis, num_points=num_points, color=color)
    sampled_points_vert_lines = get_vertical_dashed_lines_for_samples(
        graph, time_axis, color=color, num_points=num_points
    )
    return VGroup(time_axis, sampled_points_vert_lines, graph, sampled_points_dots)


def get_fourier_line_chart(
    time_func,
    t_min=0,
    t_max=10,
    f_min=0,
    f_max=10,
    n_samples=NUM_SAMPLES_FOR_FFT,
    color=FREQ_DOMAIN_COLOR,
    x_length=7,
    y_length=3,
):
    axes = get_freq_axes(
        x_range=[f_min, f_max, 1], x_length=x_length, y_length=y_length
    )
    time_samples = np.vectorize(time_func)(
        np.arange(t_min, t_max, (t_max - t_min) / n_samples)
    )
    fft_output = np.fft.fft(time_samples)
    frequencies = np.linspace(f_min, f_max, n_samples // 2)
    graph = VMobject()
    graph.set_points_smoothly(
        [
            axes.coords_to_point(
                x,
                np.abs(y) / n_samples,
            )
            for x, y in zip(frequencies, fft_output[: n_samples // 2])
        ]
    )
    graph.set_color(color)
    graph.underlying_function = lambda f: axes.y_axis.point_to_number(
        graph.point_from_proportion((f - f_min) / (f_max - f_min))
    )
    return VGroup(graph, axes)


def get_fourier_with_sample_points_and_vert_lines(
    time_func,
    t_min=0,
    t_max=10,
    f_min=0,
    f_max=10,
    n_samples=NUM_SAMPLES_FOR_FFT,
    color=FREQ_DOMAIN_COLOR,
    x_length=7,
    y_length=3,
):

    graph, axes = get_fourier_line_chart(
        time_func,
        t_min=t_min,
        t_max=t_max,
        f_min=f_min,
        f_max=f_max,
        n_samples=n_samples,
        color=color,
        x_length=x_length,
        y_length=y_length,
    )
    time_samples = np.vectorize(time_func)(
        np.arange(t_min, t_max, (t_max - t_min) / n_samples)
    )
    fft_output = np.fft.fft(time_samples)
    frequencies = np.linspace(f_min, f_max, n_samples // 2)

    x_coords = frequencies
    y_coords = [np.abs(y) / n_samples for y in fft_output[: n_samples // 2]]

    x_axis_points = [axes.x_axis.n2p(p) for p in x_coords]
    graph_points = [axes.coords_to_point(x, y) for x, y in zip(x_coords, y_coords)]
    vertical_lines = [
        Line(start_point, y_point).set_stroke(color=color, width=4)
        for start_point, y_point in zip(x_axis_points, graph_points)
    ]
    vertical_lines = VGroup(*vertical_lines)
    coord_dots = [
        Dot().move_to(axes.coords_to_point(x_coord, y_coord))
        for x_coord, y_coord in zip(x_coords, y_coords)
    ]
    sampled_points = VGroup(*coord_dots)
    return VGroup(graph, axes, sampled_points, vertical_lines)


def get_fourier_bar_chart(
    time_func,
    t_min=0,
    t_max=10,
    f_min=0,
    f_max=10,
    n_samples=NUM_SAMPLES_FOR_FFT,
    bar_width=0.2,
    height_scale=1,
    color=FREQ_DOMAIN_COLOR,
):

    time_range = float(t_max - t_min)
    time_samples = np.vectorize(time_func)(np.linspace(t_min, t_max, n_samples))
    fft_output = np.fft.fft(time_samples)
    frequencies = np.linspace(0.0, n_samples / (2.0 * time_range), n_samples // 2)

    graph = VGroup()

    for x, y in zip(frequencies, fft_output[: n_samples // 2]):
        if x <= f_max + 0.1:
            rect = (
                Rectangle(height=height_scale * np.abs(y) / n_samples, width=bar_width)
                .set_color(color)
                .set_fill(color, opacity=1)
                .set_stroke(width=1)
            )
            graph.add(rect)

    graph.arrange(RIGHT, buff=0.1, aligned_edge=DOWN)
    return graph


def get_fourier_rects_n(
    signal_func,
    n_samples=16,
    sample_rate=16,
    t_min=0,
    t_max=2 * PI,
    rect_scale=0.1,
    rect_width=0.3,
    font_scale=0.4,
    full_spectrum=False,
):
    spectrum_selection = n_samples if full_spectrum else n_samples // 2

    af_matrix = get_analysis_frequency_matrix(
        N=n_samples, sample_rate=sample_rate, t_max=t_max
    )
    sampled_signal = np.array(
        [
            signal_func(v)
            for v in np.linspace(t_min, t_max, num=n_samples, endpoint=False)
        ]
    ).reshape(-1, 1)

    mt = apply_matrix_transform(sampled_signal, af_matrix)
    rects = VGroup(
        *[
            VGroup(
                Rectangle(
                    color=REDUCIBLE_VIOLET, width=rect_width, height=f * rect_scale
                ).set_fill(REDUCIBLE_VIOLET, opacity=1),
                Text(str(i), font=REDUCIBLE_MONO).scale(font_scale),
            ).arrange(DOWN)
            for i, f in enumerate(mt.flatten()[:spectrum_selection])
        ]
    ).arrange(RIGHT, aligned_edge=DOWN)

    frequency_label = Text("Frequency", font=REDUCIBLE_MONO).scale(font_scale / 1.2)
    frequency_label.next_to(rects, DOWN)

    return VGroup(rects, frequency_label)


def get_fourier_rects_from_custom_matrix(
    signal_func,
    af_matrix,
    n_samples=16,
    t_min=0,
    t_max=2 * PI,
    rect_scale=0.1,
    rect_width=0.3,
    font_scale=0.4,
    full_spectrum=False,
):
    spectrum_selection = n_samples if full_spectrum else n_samples // 2

    sampled_signal = np.array(
        [
            signal_func(v)
            for v in np.linspace(t_min, t_max, num=n_samples, endpoint=False)
        ]
    ).reshape(-1, 1)

    mt = apply_matrix_transform(sampled_signal, af_matrix)
    mt = np.abs(mt)
    if True in np.iscomplex(mt):
        mt = np.abs(mt)

    width_heights = [
        (rect_width, f * rect_scale) for f in mt.flatten()[:spectrum_selection]
    ]
    # print("(width, height)", width_heights)
    rects = VGroup(
        *[
            VGroup(
                Rectangle(
                    color=REDUCIBLE_VIOLET,
                    width=rect_width,
                    height=f * rect_scale if f > 0.001 else 0.001,
                ).set_fill(REDUCIBLE_VIOLET, opacity=1),
                Text(str(i), font=REDUCIBLE_MONO).scale(font_scale),
            ).arrange(DOWN)
            for i, f in enumerate(mt.flatten()[:spectrum_selection])
        ]
    ).arrange(RIGHT, aligned_edge=DOWN)

    frequency_label = Text("Frequency", font=REDUCIBLE_FONT, weight=BOLD).scale(
        font_scale / 1.2
    )
    frequency_label.next_to(rects, DOWN)

    return VGroup(rects, frequency_label)


def get_analysis_frequency_matrix(N, sample_rate, t_min=0, t_max=2 * PI):
    """
    Constructs a N x N matrix of N analysis frequencies
    sampled at N points.
    The matrix holds the values in columns,
    as in the N values of the Nth analysis frequency are stored in
    column number N.
    """

    # analysis frequencies
    af = [get_cosine_func(freq=sample_rate * m / N) for m in range(N)]

    # for each analysis frequency, sample that function along N points
    # this returns the frequencies per rows, so .T transposes and
    # have the signal values in cols
    return np.array(
        [[f(s) for s in np.linspace(t_min, t_max, num=N, endpoint=False)] for f in af]
    )


def get_vertical_lines_as_samples(
    graph, axes, x_min=0, x_max=2 * PI, num_points=8, color=REDUCIBLE_YELLOW
):

    return VGroup(
        *axes.get_vertical_lines_to_graph(graph, [x_min, x_max], num_lines=num_points)[
            :-1
        ].set_color(color)
    )


def get_heat_map_from_matrix(
    matrix,
    height=3,
    width=3,
    min_color=REDUCIBLE_PURPLE,
    max_color=REDUCIBLE_YELLOW,
    integer_scale=0.3,
    grid_cell_stroke_width=1,
    color=WHITE,
):
    min_value, max_value = np.min(matrix), np.max(matrix)
    rows, cols = matrix.shape

    grid = get_grid(
        rows, cols, height, width, stroke_width=grid_cell_stroke_width, color=color
    )

    for i in range(rows):
        for j in range(cols):
            alpha = (matrix[i][j] - min_value) / (max_value - min_value)
            grid[i][j].set_fill(
                interpolate_color(min_color, max_color, alpha), opacity=1
            )

    scale = Line(grid.get_top(), grid.get_bottom())
    scale.set_stroke(width=10).set_color(color=[min_color, max_color])

    top_value = Text(str(int(max_value)), font="SF Mono").scale(integer_scale)
    top_value.next_to(scale, RIGHT, aligned_edge=UP)
    bottom_value = Text(str(int(min_value)), font="SF Mono").scale(integer_scale)
    bottom_value.next_to(scale, RIGHT, aligned_edge=DOWN)

    heat_map_scale = VGroup(scale, top_value, bottom_value)
    heat_map_scale.next_to(grid, LEFT)

    return VGroup(grid, heat_map_scale)


def get_grid(rows, cols, height, width, color=WHITE, stroke_width=1):
    cell_height = height / rows
    cell_width = width / cols
    grid = VGroup(
        *[
            VGroup(
                *[
                    Rectangle(height=cell_height, width=cell_width).set_stroke(
                        color=color, width=stroke_width
                    )
                    for j in range(cols)
                ]
            ).arrange(RIGHT, buff=0)
            for i in range(rows)
        ]
    ).arrange(DOWN, buff=0)
    return grid
