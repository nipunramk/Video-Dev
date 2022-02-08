from manim import *
from functions import *
from classes import *
from reducible_colors import *

config["assets_dir"] = "assets"

class QOIDemo(Scene):
	def construct(self):
		"""
		Plan for animation
		1. Show RGB Image pixel representation
		2. Split RGB Image into R, G, and B channels
		3. Show first option of keeping the pixel in the RGB format (show byte order)
		4. Show option of encoding run length of pixels
		5. Show option of encoding the diff of a pixel
		6. Show option of indexing to a previously seen pixel value
		7. Make sure to show how the byte order works for all this (should clarify how it works)
		"""
		self.show_image()

	def show_image(self):
		image = ImageMobject("r.png")
		pixel_array = image.get_pixel_array()
		pixel_array_mob = PixelArray(pixel_array).scale(0.4).shift(UP * 2)
		self.play(
			FadeIn(pixel_array_mob)
		)
		self.wait()

		self.show_rgb_split(pixel_array, pixel_array_mob)

	def show_rgb_split(self, pixel_array, pixel_array_mob):
		r_channel = pixel_array[:, :, 0]
		g_channel = pixel_array[:, :, 1]
		b_channel = pixel_array[:, :, 2]

		r_channel_padded = self.get_channel_image(r_channel)
		g_channel_padded = self.get_channel_image(g_channel, mode='G')
		b_channel_padded = self.get_channel_image(b_channel, mode='B')

		pixel_array_mob_r = PixelArray(r_channel_padded).scale(0.4).shift(LEFT * 4 + DOWN * 1.5)
		pixel_array_mob_g = PixelArray(g_channel_padded).scale(0.4).shift(DOWN * 1.5)
		pixel_array_mob_b = PixelArray(b_channel_padded).scale(0.4).shift(RIGHT * 4 + DOWN * 1.5)

		self.play(
			TransformFromCopy(pixel_array_mob, pixel_array_mob_r),
			TransformFromCopy(pixel_array_mob, pixel_array_mob_b),
			TransformFromCopy(pixel_array_mob, pixel_array_mob_g)
		)
		self.wait()
		r_channel_pixel_text = self.get_pixel_values(r_channel, pixel_array_mob_r)
		g_channel_pixel_text = self.get_pixel_values(g_channel, pixel_array_mob_g, mode='G')
		b_channel_pixel_text = self.get_pixel_values(b_channel, pixel_array_mob_b, mode='B')
		self.play(
			FadeIn(r_channel_pixel_text),
			FadeIn(g_channel_pixel_text),
			FadeIn(b_channel_pixel_text)
		)
		self.wait()

		self.play(
			FadeOut(pixel_array_mob),
			pixel_array_mob_r.animate.shift(UP * 3),
			pixel_array_mob_b.animate.shift(UP * 3),
			pixel_array_mob_g.animate.shift(UP * 3),
			r_channel_pixel_text.animate.shift(UP * 3),
			g_channel_pixel_text.animate.shift(UP * 3),
			b_channel_pixel_text.animate.shift(UP * 3),
		)
		self.wait()

	def get_pixel_values(self, channel, channel_mob, mode='R'):
		pixel_values_text = VGroup()
		for p_val, mob in zip(channel.flatten(), channel_mob):
			text = Text(str(p_val), font="SF Mono", weight=MEDIUM).scale(0.25).move_to(mob.get_center())
			if mode == 'G' and p_val > 200:
				text.set_color(BLACK)
			pixel_values_text.add(text)

		return pixel_values_text


	def get_channel_image(self, channel, mode='R'):
		new_channel = np.zeros((channel.shape[0], channel.shape[1], 3))
		for i in range(channel.shape[0]):
			for j in range(channel.shape[1]):
				if mode == 'R':	
					new_channel[i][j] = np.array([channel[i][j], 0, 0])
				elif mode == 'G':
					new_channel[i][j] = np.array([0, channel[i][j], 0])
				else:
					new_channel[i][j] = np.array([0, 0, channel[i][j]])

		return new_channel



		