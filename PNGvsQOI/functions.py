"""
File for various PNG and QOI Helper Functions
"""

def get_1d_index(i, j, pixel_array):
	return pixel_array.shape[1] * i + j