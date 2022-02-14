"""
File for various PNG and QOI Helper Functions
"""
from manim import SurroundingRectangle, interpolate, VGroup
from reducible_colors import *
import numpy as np

def get_1d_index(i, j, pixel_array):
	return pixel_array.shape[1] * i + j

def is_last_pixel(channel, row, col):
	return row == channel.shape[0] - 1 and col == channel.shape[1] - 1

def qoi_hash(rgb_val):
	"""
	Implemented as per https://qoiformat.org/qoi-specification.pdf
	"""
	r, g, b = rgb_val
	return r * 3 + g * 5 + b * 7

def is_diff_small(dr, dg, db):
	return dr > -3 and dr < 2 and dg > -3 and dg < 2 and db > -3 and db < 2

def is_diff_med(dg, dr_dg, db_dg):
	return dr_dg > -9 and dr_dg < 8 and dg > -33 and dg < 32 and db_dg > -9 and db_dg < 8

def get_glowing_surround_rect(pixel, buff_min=0, buff_max=0.15, color=REDUCIBLE_YELLOW, n=40, opacity_multiplier=1):
	glowing_rect = VGroup(*[SurroundingRectangle(pixel, buff=interpolate(buff_min, buff_max, b)) for b in np.linspace(0, 1, n)])
	for i, rect in enumerate(glowing_rect):
		rect.set_stroke(color, width=0.5, opacity=1 - i/n)
	return glowing_rect

