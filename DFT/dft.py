import sys
### THIS IS A WORKAROUND FOR NOW
### IT REQUIRES RUNNING MANIM FROM INSIDE DIRECTORY VIDEO-DEV
sys.path.insert(1, "common/")

from dft_utils import *

class AnalysisFrequencies(Scene):
	def construct(self):
		time_signal_func = get_cosine_func(freq=3)

		time_axis, graph = plot_time_domain(time_signal_func, t_max=2*PI)
		self.add(time_axis, graph)
		self.wait()