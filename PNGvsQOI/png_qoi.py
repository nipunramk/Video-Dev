from manim import *
from functions import *
from classes import *
from reducible_colors import *

config["assets_dir"] = "assets"

QOI_INDEX = 0
QOI_DIFF_SMALL = 1
QOI_DIFF_MED = 2
QOI_RUN = 3
QOI_RGB = 4

class QOIDemo(Scene):

	def construct(self):
		"""
		Plan for animation
		1. Show RGB Image pixel representation
		2. Split RGB Image into R, G, and B channels
		3. Show 5 different compression options (keep RGB, run, index, diff small, diff large)
		4. Make a function that takes image and generates QOI tags -- TEST ON DIFFS (seems to work)
		5. Make function to take a list of encoding options (with necessary data) 
		and demo QOI
		In demo, make functional example of a full encoding with list, show the entire bitsteam (can shift info out of the window)
		
		TODO: Tuesday
		show_encode_run
		show_encode_diff_small
		show_encode_diff_med
		show_encode_index
		show_encode_rgb
		
		TODO: Wednesday
		Further complexities we need to deal with to make full animation of QOI
		1. Moving prev and current pixels when current pixel is a RLE
		2. Keeping track of index in a visually pleasing manner
		3. Showing encoding into bytes
		4. Bring bytes and pixels into focus
		5. Dealing with initial prev value

		update_index -- handles index tracking

		get_next_animations(curr_encoding, next_encoding)
		CASES:
		1. curr_encoding - RGB, next_encoding - anything -> use update_prev_and_current
		2. curr_encoding - diff, next_encoding - anything -> use update_prev_and_current
		3. curr_encoding - index, next_encoding - anything -> use unindicate on index and then update_prev_and_current
		4. curr_encoding - RLE, next_encoding - anything -> new logic from Wednesday (1)

		TODO: Thursday
		Make fully functioning QOI animation
		"""
		self.show_image()

	def show_image(self):
		image = ImageMobject("r.png")
		pixel_array = image.get_pixel_array().astype(int)
		pixel_array_mob = PixelArray(pixel_array).scale(0.4).shift(UP * 2)
		self.play(
			FadeIn(pixel_array_mob)
		)
		self.wait()

		flattened_pixels = self.show_rgb_split(pixel_array, pixel_array_mob)
		self.introduce_qoi_tags(flattened_pixels)
		qoi_data = self.get_qoi_encoding(pixel_array)
		print('QOI_ENCODING:\n', qoi_data)

	def get_qoi_encoding(self, pixel_array):
		"""
		@param: pixel_array - np.array[r, g, b] representing pixels of image
		@return: list[] containing QOI encodings for image
		each QOI encoding is a tuple of 
		"""
		INDEX_SIZE = 64
		r_channel = pixel_array[:, :, 0]
		g_channel = pixel_array[:, :, 1]
		b_channel = pixel_array[:, :, 2]

		prev_rgb = [0, 0, 0]

		encodings = []
		indices = [[0, 0, 0]] * INDEX_SIZE
		run = 0
		for row in range(r_channel.shape[0]):
			for col in range(r_channel.shape[1]):
				current_rgb = [r_channel[row][col], g_channel[row][col], b_channel[row][col]]
				if prev_rgb == current_rgb:
					run += 1
					if run == 62 or is_last_pixel(r_channel, row, col):
						encodings.append(
							(QOI_RUN, run)
						)
						run = 0
				else:
					index_pos = 0
					if run > 0:
						encodings.append(
							(QOI_RUN, run)
						)
						run = 0

					index_pos = qoi_hash(current_rgb) % INDEX_SIZE
					if indices[index_pos] == current_rgb:
						encodings.append(
							(QOI_INDEX, index_pos)
						)
					else:
						indices[index_pos] = current_rgb

						dr = current_rgb[0] - prev_rgb[0]
						dg = current_rgb[1] - prev_rgb[1]
						db = current_rgb[2] - prev_rgb[2]

						dr_dg = dr - dg
						db_dg = db - dg

						if is_diff_small(dr, dg, db):
							encodings.append(
								(QOI_DIFF_SMALL, dr, dg, db)
							)
						elif is_diff_med(dg, dr_dg, db_dg):
							encodings.append(
								(QOI_DIFF_MED, dg, dr_dg, db_dg)
							)
						else:
							encodings.append(
								(QOI_RGB, current_rgb[0], current_rgb[1], current_rgb[2])
							)

				prev_rgb = current_rgb
		
		print('INDEX:\n', indices)
		return encodings

	def animate_encoding_of_qoi(pixel_array):
		pass

	def introduce_qoi_tags(self, flattened_pixels):
		r_channel, g_channel, b_channel = flattened_pixels
		rgb_pixels = self.get_rgb_pixels(r_channel, g_channel, b_channel)
		indices = [1]
		self.bring_pixel_to_screen(indices[0], flattened_pixels)
		transforms = self.get_indication_transforms(indices, rgb_pixels)
		
		prev_transforms = self.get_indication_transforms([0], rgb_pixels, shift=SMALL_BUFF, direction=np.array([-2.5, 1, 0]), color=REDUCIBLE_VIOLET)

		self.play(
			*transforms
		)

		self.play(
			*prev_transforms
		)
		self.wait()


		update_animations = self.update_prev_and_current(0, 1, rgb_pixels)
		self.play(
			*update_animations
		)
		self.wait()


		transforms = self.get_indication_transforms([3], rgb_pixels, extend=True)
		self.play(
			*transforms,
		)
		self.wait()

		transforms = self.get_indication_transforms([4], rgb_pixels, extend=True)
		self.play(
			*transforms,
		)
		self.wait()

		transforms = self.get_indication_transforms([5], rgb_pixels, extend=True)
		self.play(
			*transforms,
		)
		self.wait()

		# transforms = self.get_indication_transforms([6], rgb_pixels, extend=True)
		# self.play(
		# 	*transforms,
		# )
		# self.wait()

		reset_transforms = self.reset_indications(rgb_pixels)
		self.play(
			*reset_transforms
		)
		self.wait()

		# transforms = self.get_indication_transforms([54], rgb_pixels, extend=True)
		# self.play(
		# 	*transforms,
		# )
		# self.wait()

		# transforms = self.get_indication_transforms([55], rgb_pixels, extend=True)
		# self.play(
		# 	*transforms,
		# )
		# self.wait()

		# transforms = self.get_indication_transforms([49], rgb_pixels, shift=SMALL_BUFF, direction=np.array([-2.5, 1, 0]), color=REDUCIBLE_VIOLET)
		# self.play(
		# 	*transforms,
		# )
		# self.wait()

		

	def get_indication_transforms(self, indices, rgb_pixels, 
		opacity=0.2, extend=False, shift=SMALL_BUFF, direction=UP, color=REDUCIBLE_YELLOW):
		indication_transforms = []
		all_other_indices = [index for index in range(len(rgb_pixels)) if index not in indices]
		for index in all_other_indices:
			animations = []
			pixel = rgb_pixels[index]
			if pixel.indicated:
				continue
			faded_pixels = self.get_faded_pixels(pixel, opacity=opacity)
			animations.extend([Transform(pixel.r, faded_pixels[0]), Transform(pixel.g, faded_pixels[1]), Transform(pixel.b, faded_pixels[2])])
			indication_transforms.extend(animations)
		
		animations = []
		if extend:
			last_pixel_index = indices[0] - 1
			while rgb_pixels[last_pixel_index].surrounded is None:
				last_pixel_index -= 1
			original_rect = rgb_pixels[last_pixel_index].surrounded
			indicated_pixels = self.get_indicated_pixels([rgb_pixels[index] for index in range(last_pixel_index, indices[-1] + 1)], shift=shift, direction=direction)
			surrounded_rects = self.get_surrounded_rects(indicated_pixels, color=color)
			animations.append(Transform(original_rect, VGroup(*surrounded_rects)))

		pixels = [rgb_pixels[index] for index in indices]
		indicated_pixels = self.get_indicated_pixels(pixels, shift=shift, direction=direction)
		for pixel in pixels:
			pixel.indicated = True
		surrounded_rects = self.get_surrounded_rects(indicated_pixels, color=color)
		if not extend:
			pixels[0].surrounded = VGroup(*surrounded_rects)
		animations.extend(self.get_scale_transforms(pixels, indicated_pixels))
		if not extend:
			animations.extend(
				[
				FadeIn(surrounded_rects[0]), FadeIn(surrounded_rects[1]), FadeIn(surrounded_rects[2])
				]
			)
		indication_transforms.extend(animations)

		return indication_transforms

	def update_prev_and_current(self, prev_index, current_index, rgb_pixels):
		current_direction_shift = LEFT * 2.5 * SMALL_BUFF
		current_direction_scale = 1
		prev_direction_shift = RIGHT * 2.5 * SMALL_BUFF
		prev_direction_scale = 1 / 1.2
		next_direction_shift = UP * SMALL_BUFF
		next_direction_scale = 1.2

		prev_pixel = rgb_pixels[prev_index]
		current_pixel = rgb_pixels[current_index]
		next_pixel = rgb_pixels[current_index + 1]
		
		animations = []
		unindicate_prev = self.unindicate_pixels(prev_pixel)
		indicate_next, next_pixels = self.indicate_next_pixel(rgb_pixels[current_index + 1])
		transform_curr_to_prev, new_prev_pixels = self.current_to_prev(current_pixel, current_direction_shift)
		animations.extend(unindicate_prev + indicate_next + transform_curr_to_prev)
		animations.append(ApplyMethod(prev_pixel.surrounded.move_to, VGroup(*new_prev_pixels).get_center()))
		animations.append(ApplyMethod(current_pixel.surrounded.move_to, VGroup(*next_pixels).get_center()))
		prev_pixel.surrounded, current_pixel.surrounded, next_pixel.surrounded = None, prev_pixel.surrounded, current_pixel.surrounded
		return animations

	def current_to_prev(self, rgb_pixel, shift):
		rgb_pixel.shift = np.array([shift[0], rgb_pixel.shift[1], 0])
		animations = []
		new_pixel = [
		self.current_to_prev_channel(rgb_pixel.r, shift),
		self.current_to_prev_channel(rgb_pixel.g, shift),
		self.current_to_prev_channel(rgb_pixel.b, shift)
		]
		animations.append(Transform(rgb_pixel.r, new_pixel[0]))
		animations.append(Transform(rgb_pixel.g, new_pixel[1]))
		animations.append(Transform(rgb_pixel.b, new_pixel[2]))
		return animations, new_pixel

	def current_to_prev_channel(self, channel, shift):
		return channel.copy().shift(shift)

	def unindicate_pixels(self, rgb_pixel):
		animations = []
		if rgb_pixel.indicated:
			animations.append(Transform(rgb_pixel.r, self.unindicate_pixel(rgb_pixel, rgb_pixel.r)))
			animations.append(Transform(rgb_pixel.g, self.unindicate_pixel(rgb_pixel, rgb_pixel.g)))
			animations.append(Transform(rgb_pixel.b, self.unindicate_pixel(rgb_pixel, rgb_pixel.b)))
			rgb_pixel.indicated = False
			rgb_pixel.scaled = 1
			rgb_pixel.shift = ORIGIN
		return animations

	def indicate_next_pixel(self, next_pixel):
		animations = []
		indicated_pixel = self.get_indicated_pixels([next_pixel])
		next_pixels = [indicated_pixel[0][0], indicated_pixel[1][0], indicated_pixel[2][0]]
		if not next_pixel.indicated:
			animations.append(Transform(next_pixel.r, next_pixels[0]))
			animations.append(Transform(next_pixel.g, next_pixels[1]))
			animations.append(Transform(next_pixel.b, next_pixels[2]))
			next_pixel.indicated = True
		return animations, next_pixels

	def unindicate_pixel(self, original_pixel, channel, opacity=0.2):
		pixel = channel.copy()
		pixel.scale(1/original_pixel.scaled).shift(-original_pixel.shift)
		pixel[0].set_fill(opacity=opacity).set_stroke(opacity=opacity)
		pixel[1].set_fill(opacity=opacity)
		return pixel

	def get_scale_transforms(self, pixels, indicated_pixels):
		transforms = []
		for i, pixel in enumerate(pixels):
			transforms.append(Transform(pixel.r, indicated_pixels[0][i]))
			transforms.append(Transform(pixel.g, indicated_pixels[1][i]))
			transforms.append(Transform(pixel.b, indicated_pixels[2][i]))

		return transforms

	def get_faded_pixels(self, pixel, opacity=0.2):
		r_pixel = self.get_faded_pixel(pixel.r, opacity=opacity)
		g_pixel = self.get_faded_pixel(pixel.g, opacity=opacity)
		b_pixel = self.get_faded_pixel(pixel.b, opacity=opacity)
		return [r_pixel, g_pixel, b_pixel]

	def get_indicated_pixels(self, pixels, scale=1.2, shift=SMALL_BUFF, direction=UP, reset=False):
		r_pixel = VGroup(*[self.get_indicated_pixel(pixel, pixel.r, scale=scale, shift=shift, direction=direction, reset=reset) for pixel in pixels])
		g_pixel = VGroup(*[self.get_indicated_pixel(pixel, pixel.g, scale=scale, shift=shift, direction=direction, reset=reset) for pixel in pixels])
		b_pixel = VGroup(*[self.get_indicated_pixel(pixel, pixel.b, scale=scale, shift=shift, direction=direction, reset=reset) for pixel in pixels])
		return [r_pixel, g_pixel, b_pixel]

	def get_surrounded_rects(self, indicated_pixels, color=REDUCIBLE_YELLOW):
		return [get_glowing_surround_rect(pixel, color=color) for pixel in indicated_pixels]

	def get_indicated_pixel(self, original_pixel, channel, scale=1.2, shift=SMALL_BUFF, direction=UP, indicated=False, reset=False):
		pixel = channel.copy()
		pixel = self.get_faded_pixel(pixel, opacity=1)
		if not original_pixel.indicated:
			original_pixel.scaled = scale
			original_pixel.shift = direction * shift
			pixel.scale(scale).shift(direction * shift)
		elif reset:
			pixel.scale(1/original_pixel.scaled).shift(-original_pixel.shift)

		return pixel

	def get_faded_pixel(self, channel, opacity=0.2):
		pixel = channel.copy()
		pixel[0].set_fill(opacity=opacity).set_stroke(opacity=opacity)
		pixel[1].set_fill(opacity=opacity)
		return pixel

	def get_rgb_pixels(self, r_channel, b_channel, g_channel):
		pixels = []
		for i in range(len(r_channel[1])):
			r_mob = VGroup(r_channel[0][0, i], r_channel[1][i])
			g_mob = VGroup(g_channel[0][0, i], g_channel[1][i])
			b_mob = VGroup(b_channel[0][0, i], b_channel[1][i])
			pixels.append(RGBMob(r_mob, g_mob, b_mob))

		return pixels

	def reset_indications(self, rgb_pixels):
		animations = []
		for i, pixel in enumerate(rgb_pixels):
			if pixel.indicated:
				if pixel.surrounded:
					animations.append(FadeOut(pixel.surrounded))
					pixel.surrounded = None
				original_pixel = self.get_indicated_pixels([pixel], reset=True)
				animations.append(Transform(pixel.r, original_pixel[0][0]))
				animations.append(Transform(pixel.g, original_pixel[1][0]))
				animations.append(Transform(pixel.b, original_pixel[2][0]))
				pixel.indicated = False
				pixel.scale = 1
				pixel.shift = ORIGIN
			else:
				original_pixel = self.get_faded_pixels(pixel, opacity=1)
				animations.append(Transform(pixel.r, original_pixel[0]))
				animations.append(Transform(pixel.g, original_pixel[1]))
				animations.append(Transform(pixel.b, original_pixel[2]))

		return animations

		

	def bring_pixel_to_screen(self, index, flattened_pixels, target_x_pos=0):
		r_channel = flattened_pixels[0]
		target_pixel = r_channel[0][index]
		shift_amount = self.get_shift_amount(target_pixel.get_center(), target_x_pos=target_x_pos)
		if not np.array_equal(shift_amount, ORIGIN):
			self.play(
				flattened_pixels.animate.shift(shift_amount)
			)
			self.wait()


	def get_shift_amount(self, position, target_x_pos=0):
		x_pos = position[0]
		print(x_pos)
		BUFF = 1
		if x_pos > -config.frame_x_radius + BUFF and x_pos < config.frame_x_radius - BUFF:
			# NO Shift needed
			return ORIGIN

		return (target_x_pos - x_pos) * RIGHT

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

		r_channel_flattened = self.reshape_channel(r_channel_padded)
		g_channel_flattened = self.reshape_channel(g_channel_padded)
		b_channel_flattened = self.reshape_channel(b_channel_padded)

		r_channel_f_mob = PixelArray(r_channel_flattened, buff=MED_SMALL_BUFF, outline=False).scale(0.6).to_edge(LEFT)
		g_channel_f_mob = PixelArray(g_channel_flattened, buff=MED_SMALL_BUFF, outline=False).scale(0.6).to_edge(LEFT)
		b_channel_f_mob = PixelArray(b_channel_flattened, buff=MED_SMALL_BUFF, outline=False).scale(0.6).to_edge(LEFT)

		r_channel_f_mob.to_edge(LEFT * 3).shift(DOWN * 1.1)
		g_channel_f_mob.next_to(r_channel_f_mob, DOWN * 2, aligned_edge=LEFT)
		b_channel_f_mob.next_to(g_channel_f_mob, DOWN * 2, aligned_edge=LEFT)
		
		r_channel_f_mob_text = self.get_pixel_values(r_channel_flattened[:, :, 0], r_channel_f_mob, mode='R')
		g_channel_f_mob_text = self.get_pixel_values(g_channel_flattened[:, :, 1], g_channel_f_mob, mode='G')
		b_channel_f_mob_text = self.get_pixel_values(b_channel_flattened[:, :, 2], b_channel_f_mob, mode='B')


		r_transforms = self.get_flatten_transform(pixel_array_mob_r, r_channel_f_mob, r_channel_pixel_text, r_channel_f_mob_text)

		g_transforms = self.get_flatten_transform(pixel_array_mob_g, g_channel_f_mob, g_channel_pixel_text, g_channel_f_mob_text)

		b_transforms = self.get_flatten_transform(pixel_array_mob_b, b_channel_f_mob, b_channel_pixel_text, b_channel_f_mob_text)
		
		self.play(
			*r_transforms,
			run_time=3
		)
		self.wait()

		self.play(
			*g_transforms,
			run_time=3
		)
		self.wait()

		self.play(
			*b_transforms,
			run_time=3
		)
		self.wait()

		self.play(
			FadeOut(pixel_array_mob_r),
			FadeOut(pixel_array_mob_g),
			FadeOut(pixel_array_mob_b),
			FadeOut(r_channel_pixel_text),
			FadeOut(g_channel_pixel_text),
			FadeOut(b_channel_pixel_text),
			r_channel_f_mob.animate.shift(UP * 2.5),
			g_channel_f_mob.animate.shift(UP * 2.5),
			b_channel_f_mob.animate.shift(UP * 2.5),
			r_channel_f_mob_text.animate.shift(UP * 2.5),
			g_channel_f_mob_text.animate.shift(UP * 2.5),
			b_channel_f_mob_text.animate.shift(UP * 2.5),
		)
		self.wait()

		r_channel = VGroup(r_channel_f_mob, r_channel_f_mob_text)
		g_channel = VGroup(g_channel_f_mob, g_channel_f_mob_text)
		b_channel = VGroup(b_channel_f_mob, b_channel_f_mob_text)

		return VGroup(r_channel, g_channel, b_channel)

		# qoi_rgb_bytes = self.get_rbg_bytes().move_to(DOWN * 2)
		# self.add(qoi_rgb_bytes)
		# self.wait()
		# self.remove(qoi_rgb_bytes)

		# qoi_index_bytes = self.get_index_bytes().move_to(DOWN * 2)
		# self.add(qoi_index_bytes)
		# self.wait()

		# self.remove(qoi_index_bytes)

		# qoi_run_bytes = self.get_run_bytes().move_to(DOWN * 2)
		# self.add(qoi_run_bytes)
		# self.wait()

		# self.remove(qoi_run_bytes)

		# qoi_large_diff_bytes = self.get_large_diff_bytes().move_to(DOWN * 2)
		# self.add(qoi_large_diff_bytes)
		# self.wait()

		# self.remove(qoi_large_diff_bytes)
		# qoi_small_diff_bytes = self.get_small_diff_bytes().move_to(DOWN * 2)
		# self.add(qoi_small_diff_bytes)
		# self.wait()

		# self.remove(qoi_small_diff_bytes)
		# self.wait()

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
			text = Text(str(int(p_val)), font="SF Mono", weight=MEDIUM).scale(0.25).move_to(mob.get_center())
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

	def reshape_channel(self, channel):
		return np.reshape(channel, (1, channel.shape[0] * channel.shape[1], channel.shape[2]))

	def get_flatten_transform(self, original_mob, flattened_mob, original_text, flattened_text):
		transforms = []
		for i in range(original_mob.shape[0]):
			for j in range(original_mob.shape[1]):
				one_d_index = i * original_mob.shape[1] + j
				transforms.append(TransformFromCopy(original_mob[i, j], flattened_mob[0, one_d_index]))
				transforms.append(TransformFromCopy(original_text[one_d_index], flattened_text[one_d_index]))

		return transforms





		