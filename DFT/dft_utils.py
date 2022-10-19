import sys

### THIS IS A WORKAROUND FOR NOW
### IT REQUIRES RUNNING MANIM FROM INSIDE DIRECTORY VIDEO-DEV
sys.path.insert(1, "common/")


import matplotlib.pyplot as plt
import numpy as np

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

