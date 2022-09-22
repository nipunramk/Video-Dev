import sys

### THIS IS A WORKAROUND FOR NOW
### IT REQUIRES RUNNING MANIM FROM INSIDE DIRECTORY VIDEO-DEV
sys.path.insert(1, "common/")


import matplotlib.pyplot as plt
import numpy as np
from reducible_colors import *
from manim import *
from scipy.signal import chirp
import pywt

NUM_SAMPLES_FOR_FFT = 10000
DEFAULT_AXIS_CONFIG = {"include_numbers": False, "include_ticks": False}


def get_summed_sin_functions(amplitudes, freqs):
    pass


def get_sine_func():
    f1 = 1
    a1 = 1
    f2 = 3
    a2 = 2
    f3 = 5
    a3 = 1
    return (
        lambda x: a1 * np.sin(x * f1 * 2 * np.pi)
        + a2 * np.sin(x * f2 * 2 * np.pi)
        + a3 * np.sin(x * f3 * 2 * np.pi)
    )


def plot_function(f, N):
    x = np.arange(0, N, 0.001)
    y = f(x)

    plt.plot(x, y)
    plt.show()


def get_fourier_graph(
    axes,
    time_func,
    t_min=0,
    t_max=10,
    f_min=0,
    f_max=10,
    n_samples=NUM_SAMPLES_FOR_FFT,
    complex_to_real_func=lambda z: z.real,
    color=REDUCIBLE_YELLOW,
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


def get_fourier_vertical_lines(
    axes,
    time_func,
    t_min=0,
    t_max=10,
    f_min=0,
    f_max=10,
    n_samples=NUM_SAMPLES_FOR_FFT,
    complex_to_real_func=lambda z: z.real,
    color=REDUCIBLE_VIOLET,
):
    time_range = float(t_max - t_min)
    time_step_size = time_range / n_samples
    time_samples = np.vectorize(time_func)(np.linspace(t_min, t_max, n_samples))
    fft_output = np.fft.fft(time_samples)
    freq_bins = np.linspace(0.0, n_samples / (2.0 * time_range), n_samples // 2)
    x_axis_points = [axes.x_axis.n2p(p) for p in freq_bins]
    freq_points = [
        axes.coords_to_point(x, np.abs(y) / n_samples)
        for x, y in zip(freq_bins, fft_output[: n_samples // 2])
        if x <= f_max
    ]

    vertical_lines = [
        Line(start_point, y_point).set_stroke(color=color, width=4)
        for start_point, y_point in zip(x_axis_points, freq_points)
    ]

    print(len(freq_points), (len(vertical_lines)))
    return VGroup(*vertical_lines)


def get_time_axes(
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


def chirp_piecewise(t):
    f1 = 1
    a1 = 10
    f2 = 2
    a2 = 10
    f3 = 3
    a3 = 10
    if t < 6:
        return a1 * np.sin(t * f1 * 2 * np.pi)
    elif t < 12:
        return a2 * np.sin(t * f2 * 2 * np.pi)
    return a3 * np.sin(t * f3 * 2 * np.pi)


def wavelet_plot_exp():
    # Define signal
    fs = 128
    sampling_period = 1 / fs
    t = np.linspace(0, 18, 2 * fs)
    x = np.vectorize(chirp_piecewise)(t)

    # Calculate continuous wavelet transform
    coef, freqs = pywt.cwt(
        x, np.arange(1, 100), "morl", sampling_period=sampling_period
    )
    # Show w.r.t. time and frequency
    plt.figure(figsize=(5, 2))
    plt.pcolor(t, freqs, coef)
    print(t.shape, x.shape, freqs.shape, coef.shape)

    # Set yscale, ylim and labels
    plt.ylim([1, 100])
    plt.ylabel("Frequency (Hz)")
    plt.xlabel("Time (sec)")
    plt.show()

    coefficient, freq = pywt.cwt(x, [1], "morl", sampling_period=sampling_period)
    print(coefficient, freq)


# wavelet_plot_exp()

# def
