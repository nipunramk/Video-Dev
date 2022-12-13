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


def inner_prod(time_domain_func, freq_func, x_min=0, x_max=2 * PI, num_points=8):
    x_coords = np.linspace(x_min, x_max, num=num_points, endpoint=False)
    y_coords_time = time_domain_func(x_coords)
    y_coords_freq = freq_func(x_coords)
    return np.dot(y_coords_time, y_coords_freq)


def get_sampled_dots(
    graph, axes, x_min=0, x_max=2 * PI, num_points=8, radius=DEFAULT_DOT_RADIUS
):
    x_coords, y_coords = get_sampled_coords(
        graph, x_min=x_min, x_max=x_max, num_points=num_points
    )
    coord_dots = [
        Dot(radius=radius).move_to(axes.coords_to_point(x_coord, y_coord))
        for x_coord, y_coord in zip(x_coords, y_coords)
    ]
    return VGroup(*coord_dots)


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


def display_signal(time_signal_func, color=TIME_DOMAIN_COLOR):
    time_axis, graph = plot_time_domain(time_signal_func, t_max=2 * PI, color=color)
    sampled_points_dots = get_sampled_dots(graph, time_axis)
    sampled_points_vert_lines = get_vertical_bars_for_samples(graph, time_axis)
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


def get_analysis_frequency_matrix(N, sample_rate, func="cos", t_min=0, t_max=2 * PI):
    """
    Constructs a N x N matrix of N analysis frequencies
    sampled at N points.

    The matrix holds the values in columns,
    as in the N values of the Nth analysis frequency are stored in
    column number N.

    We can choose how the transform is calculated, if using sine or cosine
    functions. Just pass 'sin' to the "func" argument. By default, this value
    is set to 'cos'.
    """

    if func != "cos" and func != "sin":
        raise ValueError('func can either be "cos" or "sin"')

    if func == "cos":
        signal_func = get_cosine_func
    else:
        signal_func = get_sine_func

    # analysis frequencies
    af = [signal_func(freq=sample_rate * m / N) for m in range(N)]

    # for each analysis frequency, sample that function along N points
    # this returns the frequencies per rows, so .T transposes and
    # have the signal values in cols
    return np.array(
        [[f(s) for s in np.linspace(t_min, t_max, num=N, endpoint=False)] for f in af]
    )


def get_rectangles_for_matrix_transform(
    sampled_signal, analysis_frequency_matrix, rect_scale=0.1
):
    """
    Create an array of rectangles with annotations to represent the matrix transform input.

    INPUTS
    ------
    This function accepts a sampled signal and an analysis frequency matrix to transform the signal.
    The sampled signal is an array of values and the analysis frequency matrix is a 2D square array of
    N analysis frequencies sampled at N points.

    The rectangle scaling argument is there to control the height of the rectangles, since
    the values can get pretty wild if not controlled properly.

    RETURN
    ------
    The function returns a VGroup of rectangles and text, and this VGroup is already aligned down,
    in order for it to work properly with redrawing animations.
    """

    matrix_transform = apply_matrix_transform(sampled_signal, analysis_frequency_matrix)

    rects = (
        VGroup(
            *[
                VGroup(
                    Rectangle(
                        color=REDUCIBLE_VIOLET, width=0.3, height=f * rect_scale
                    ).set_fill(REDUCIBLE_VIOLET, opacity=1),
                    Text(str(i), font=REDUCIBLE_MONO).scale(0.4),
                ).arrange(DOWN)
                for i, f in enumerate(
                    matrix_transform.flatten()[: matrix_transform.shape[0]]
                )
            ]
        )
        .arrange(RIGHT, aligned_edge=DOWN)
        .scale(0.6)
        .move_to(DOWN * 3.4, aligned_edge=DOWN)
    )
    return rects


def display_signal(time_signal_func, color=TIME_DOMAIN_COLOR):
    time_axis, graph = plot_time_domain(time_signal_func, t_max=2 * PI, color=color)
    sampled_points_dots = get_sampled_dots(graph, time_axis)
    sampled_points_vert_lines = get_vertical_dashed_lines_for_samples(
        graph, time_axis, color=color
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


def get_fourier_rects(
    signal_func,
    n_samples=32,
    sample_rate=32,
    t_min=0,
    t_max=2 * PI,
    rect_scale=0.1,
    rect_width=0.3,
    font_scale=0.4,
):
    # idea: sample points from get_fourier_line_chart
    # and create num_bars dynamically to scale with the axes size
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
            for i, f in enumerate(mt.flatten()[: mt.shape[0]])
        ]
    ).arrange(RIGHT, aligned_edge=DOWN)

    frequency_label = Text("Frequency", font=REDUCIBLE_MONO).scale(font_scale / 1.2)
    frequency_label.next_to(rects, DOWN)

    return VGroup(rects, frequency_label)
