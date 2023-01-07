from manim import *
sys.path.insert(1, "common/")

from reducible_colors import *

class Title(Scene):
	def construct(self):
		title = Text("ALUM YE MO YEH", font=REDUCIBLE_FONT, t2c={"ALUM": RED})
		self.play(
			FadeIn(title)
		)
		self.wait()