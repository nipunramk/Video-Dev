import sys

### THIS IS A WORKAROUND FOR NOW
### IT REQUIRES RUNNING MANIM FROM INSIDE DIRECTORY VIDEO-DEV
sys.path.insert(1, "common/")


import matplotlib.pyplot as plt
import numpy as np
from manim import *
from reducible_colors import *
from functions import *

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
        x_range=[t_min, t_max, t_step], y_range=[y_min, y_max, y_step]
    )
    graph = time_axis.plot(time_func).set_color(color)

    return time_axis, graph


def get_cosine_func(amplitude=1, freq=1, phase=0, b=0):
    return lambda x: amplitude * np.cos(freq * x + phase) + b


def get_sine_func(amplitude=1, freq=1, phase=0, b=0):
    return lambda x: amplitude * np.sin(freq * x + phase) + b


def get_sampled_coords(graph, x_min=0, x_max=2 * PI, num_points=8):
    func = graph.underlying_function
    x_coords = np.linspace(x_min, x_max, num=num_points, endpoint=False)
    y_coords = func(x_coords)
    return x_coords, y_coords


def get_sampled_dots(graph, axes, x_min=0, x_max=2 * PI, num_points=8):
    x_coords, y_coords = get_sampled_coords(
        graph, x_min=x_min, x_max=x_max, num_points=num_points
    )
    coord_dots = [
        Dot().move_to(axes.coords_to_point(x_coord, y_coord))
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
