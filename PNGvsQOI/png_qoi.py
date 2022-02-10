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

		qoi_rgb_bytes = self.get_rbg_bytes().move_to(DOWN * 2)
		self.add(qoi_rgb_bytes)
		self.wait()
		self.remove(qoi_rgb_bytes)

		qoi_index_bytes = self.get_index_bytes().move_to(DOWN * 2)
		self.add(qoi_index_bytes)
		self.wait()

		self.remove(qoi_index_bytes)

		qoi_run_bytes = self.get_run_bytes().move_to(DOWN * 2)
		self.add(qoi_run_bytes)
		self.wait()

		self.remove(qoi_run_bytes)

		qoi_large_diff_bytes = self.get_large_diff_bytes().move_to(DOWN * 2)
		self.add(qoi_large_diff_bytes)
		self.wait()

		self.remove(qoi_large_diff_bytes)
		qoi_small_diff_bytes = self.get_small_diff_bytes().move_to(DOWN * 2)
		self.add(qoi_small_diff_bytes)
		self.wait()

	def get_rbg_bytes(self):
		rgb_tag_byte = Byte(
			["Byte[0]",
			"7,6,5,4,3,2,1,0"]
		).scale(0.5).move_to(DOWN * 2)

		red_first_byte = Byte(
			["Byte[1]",
			"7,...,0"],
			width=2,
			height=rgb_tag_byte.height
		)

		red_first_byte.text.scale_to_fit_height(rgb_tag_byte.text.height)

		red_first_byte.next_to(rgb_tag_byte, RIGHT, buff=0)

		green_second_byte = Byte(
			["Byte[2]",
			"7,...,0"],
			width=red_first_byte.width,
			height=rgb_tag_byte.height
		)

		green_second_byte.text.scale_to_fit_height(red_first_byte.text.height)

		green_second_byte.next_to(red_first_byte, RIGHT, buff=0)

		blue_third_byte = Byte(
			["Byte[3]",
			"7,...,0"],
			width=red_first_byte.width,
			height=rgb_tag_byte.height
		)

		blue_third_byte.text.scale_to_fit_height(red_first_byte.text.height)

		blue_third_byte.next_to(green_second_byte, RIGHT, buff=0)

		tag_value = Byte(
			"8 bit RGB_TAG",
			width=rgb_tag_byte.width,
			height=rgb_tag_byte.height,
			text_scale=0.25
		)
		
		tag_value.next_to(rgb_tag_byte, DOWN, buff=0)

		red_value = Byte(
			"Red",
			width=red_first_byte.width,
			height=red_first_byte.height,
			text_scale=0.25
		).next_to(red_first_byte, DOWN, buff=0)
		red_value.text.scale_to_fit_height(tag_value.text.height)

		green_value = Byte(
			"Green",
			width=green_second_byte.width,
			height=green_second_byte.height,
			text_scale=0.25
		).next_to(green_second_byte, DOWN, buff=0)
		green_value.text.scale_to_fit_height(tag_value.text.height)

		blue_value = Byte(
			"Blue",
			width=blue_third_byte.width,
			height=blue_third_byte.height,
			text_scale=0.25
		).next_to(blue_third_byte, DOWN, buff=0)
		blue_value.text.scale_to_fit_height(tag_value.text.height)

		qoi_rgb_bytes = VGroup(
			rgb_tag_byte, red_first_byte, green_second_byte, blue_third_byte,
			tag_value, red_value, green_value, blue_value,
		).move_to(ORIGIN)

		return qoi_rgb_bytes

	def get_index_bytes(self):
		index_tag_byte = Byte(
			["Byte[0]",
			"7,6,5,4,3,2,1,0"]
		).scale(0.5).move_to(DOWN * 2)

		target_text = VGroup(
			index_tag_byte.text[1][1],
			index_tag_byte.text[1][2]
		)
		tag_value = Byte(
			"0,0",
			width=target_text.get_center()[0] + SMALL_BUFF - index_tag_byte.get_left()[0],
			height=index_tag_byte.height
		)
		tag_value.text.scale_to_fit_height(index_tag_byte.text[1][1].height)

		tag_value.next_to(index_tag_byte, DOWN, aligned_edge=LEFT, buff=0)

		index_value = Byte(
			"index",
			width=index_tag_byte.get_right()[0] - tag_value.get_right()[0],
			height=tag_value.height
		)

		index_value.text.scale_to_fit_height(tag_value.text.height).scale(1.3)

		index_value.next_to(index_tag_byte, DOWN, aligned_edge=RIGHT, buff=0)

		qoi_index_bytes = VGroup(
			index_tag_byte,
			tag_value, index_value
		).move_to(ORIGIN)

		return qoi_index_bytes

	def get_run_bytes(self):
		run_tag_byte = Byte(
			["Byte[0]",
			"7,6,5,4,3,2,1,0"]
		).scale(0.5).move_to(DOWN * 2)

		target_text = VGroup(
			run_tag_byte.text[1][1],
			run_tag_byte.text[1][2]
		)
		tag_value = Byte(
			"1,1",
			text_scale=1,
			width=target_text.get_center()[0] + SMALL_BUFF - run_tag_byte.get_left()[0],
			height=run_tag_byte.height
		)
		tag_value.text.scale_to_fit_height(run_tag_byte.text[1][1].height)

		tag_value.next_to(run_tag_byte, DOWN, aligned_edge=LEFT, buff=0)
		tag_value.text.rotate(PI) # Not sure why this has to be done? Some issue with VGroup arrangement
		tag_value.text[0].shift(LEFT * SMALL_BUFF * 0.5)
		tag_value.text[1].shift(RIGHT * SMALL_BUFF * 0.5)

		run_value = Byte(
			"run",
			width=run_tag_byte.get_right()[0] - tag_value.get_right()[0],
			height=tag_value.height
		)

		run_value.text.scale_to_fit_height(tag_value.text.height).scale(1.1)


		run_value.next_to(run_tag_byte, DOWN, aligned_edge=RIGHT, buff=0)

		qoi_index_bytes = VGroup(
			run_tag_byte,
			tag_value, run_value
		).move_to(ORIGIN)

		return qoi_index_bytes

	def get_large_diff_bytes(self):
		diff_tag_byte = Byte(
			["Byte[0]",
			"7,6,5,4,3,2,1,0"]
		).scale(0.5).move_to(DOWN * 2)

		target_text = VGroup(
			diff_tag_byte.text[1][1],
			diff_tag_byte.text[1][2]
		)
		tag_value = Byte(
			"1,0",
			width=target_text.get_center()[0] + SMALL_BUFF - diff_tag_byte.get_left()[0],
			height=diff_tag_byte.height
		)
		tag_value.text.scale_to_fit_height(diff_tag_byte.text[1][1].height)
		tag_value.text.rotate(PI)

		tag_value.next_to(diff_tag_byte, DOWN, aligned_edge=LEFT, buff=0)

		dg_value = Byte(
			"diff green (dg)",
			width=diff_tag_byte.get_right()[0] - tag_value.get_right()[0],
			height=tag_value.height
		)

		dg_value.text.scale_to_fit_height(tag_value.text.height).scale(1.1)
		

		dg_value.next_to(diff_tag_byte, DOWN, aligned_edge=RIGHT, buff=0)

		second_byte = Byte(
			["Byte[1]",
			"7,6,5,4,3,2,1,0"]
		).scale(0.5).next_to(diff_tag_byte, RIGHT, buff=0)

		second_target_text = VGroup(
			diff_tag_byte.text[1][3],
			diff_tag_byte.text[1][4]
		)

		dr_dg_value = Byte(
			"dr - dg",
			width=second_target_text.get_center()[0] - dg_value.get_right()[0],
			height=dg_value.height
		).next_to(second_byte, DOWN, aligned_edge=LEFT, buff=0)
		dr_dg_value.text.scale_to_fit_height(dg_value.text.height)

		db_dg_value = Byte(
			"db - dg",
			width=dr_dg_value.width,
			height=dg_value.height
		).next_to(second_byte, DOWN, aligned_edge=RIGHT, buff=0)
		db_dg_value.text.scale_to_fit_height(dr_dg_value.text.height)

		qoi_diff_bytes = VGroup(
			diff_tag_byte, second_byte,
			tag_value, dg_value, dr_dg_value, db_dg_value,
		).move_to(ORIGIN)

		return qoi_diff_bytes

	def get_small_diff_bytes(self):
		diff_tag_byte = Byte(
			["Byte[0]",
			"7,6,5,4,3,2,1,0"]
		).scale(0.5).move_to(DOWN * 2)

		target_text = VGroup(
			diff_tag_byte.text[1][1],
			diff_tag_byte.text[1][2]
		)
		tag_value = Byte(
			"1,0",
			width=target_text.get_center()[0] + SMALL_BUFF - diff_tag_byte.get_left()[0],
			height=diff_tag_byte.height
		)
		tag_value.text.scale_to_fit_height(diff_tag_byte.text[1][1].height)
		tag_value.text.rotate(PI)

		tag_value.next_to(diff_tag_byte, DOWN, aligned_edge=LEFT, buff=0)

		second_target_text = VGroup(
			diff_tag_byte.text[1][3],
			diff_tag_byte.text[1][4]
		)

		dr_value = Byte(
			"dr",
			width=second_target_text.get_center()[0] - tag_value.get_right()[0],
			height=tag_value.height
		).next_to(tag_value, RIGHT, buff=0)
		dr_value.text.scale_to_fit_height(tag_value.text.height)
		dr_value.text.rotate(PI)

		third_target_text = VGroup(
			diff_tag_byte.text[1][5],
			diff_tag_byte.text[1][6]
		)

		dg_value = Byte(
			"dg",
			width=third_target_text.get_center()[0] - dr_value.get_right()[0],
			height=tag_value.height
		).next_to(dr_value, RIGHT, buff=0)
		dg_value.text.scale_to_fit_height(dr_value.text.height).scale(1.2)
		dg_value.text.rotate(PI)

		db_value = Byte(
			"db",
			width=diff_tag_byte.get_right()[0] - third_target_text.get_center()[0],
			height=tag_value.height
		).next_to(dg_value, RIGHT, buff=0)
		db_value.text.scale_to_fit_height(dr_value.text.height)
		db_value.text.rotate(PI)

		qoi_diff_bytes = VGroup(
			diff_tag_byte, 
			tag_value, dr_value, dg_value, db_value
		)

		return qoi_diff_bytes

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



		