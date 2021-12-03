import enum
from manim import *
import cv2
from scipy import fftpack


config["assets_dir"] = "assets"

"""
Make sure you run manim CE with --disable_caching flag
If you run with caching, since there are some scenes that change pixel arrays,
there might be some unexpected behavior
E.g manim -pql JPEGImageCompression/image_compression.py --disable_caching
"""

REDUCIBLE_BLUE = "#650FFA"
REDUCIBLE_PURPLE = "#8c4dfb"
REDUCIBLE_VIOLET = "#d7b5fe"
REDUCIBLE_YELLOW = "#ffff5c"
REDUCIBLE_GREEN_DARKER = "#00cc70"
REDUCIBLE_GREEN_DARKER = "#008f4f"


class MotivateAndExplainYCbCr(ThreeDScene):
    def construct(self):
        self.set_camera_orientation(phi=75 * DEGREES, theta=-45 * DEGREES)
        self.move_camera(zoom=0.2)

        cubes_vg = self.create_color_space_cube(
            coords2rgbcolor, color_res=4, cube_side_length=1
        )
        self.wait(2)
        self.add(
            cubes_vg,
        )
        self.wait(2)

        for index, cube in enumerate(cubes_vg):
            coords = index2coords(index, base=4)
            print(coords)
            self.remove(cube)
            self.wait()

    def create_color_space_cube(
        self,
        color_space_func,
        color_res=8,
        cube_side_length=0.1,
        buff=0.05,
    ):
        """
        Creates a YCbCr cube composed of many smaller cubes. The `color_res` argument defines
        how many cubes there will be to represent the full spectrum, particularly color_res ^ 3.
        Works exactly like `create_rgb_cube` but for YCrCb colors.

        @param: color_space_func - A function that defines what color space we are going to use.

        @param: color_res - defines the number of cubes in every dimension. Higher values yield a finer
        representation of the space, but are very slow to deal with. It is recommended that color_res is a power of two.

        @param: cube_side_length - defines the side length of each individual cube

        @param: buff - defines how much space there will be between each cube

        @return: Group - returns a group of 3D cubes colored to form the color space
        """

        MAX_COLOR_RES = 256
        discrete_ratio = MAX_COLOR_RES // color_res

        side_length = cube_side_length
        offset = side_length + buff
        cubes = []

        for i in range(color_res):
            for j in range(color_res):
                for k in range(color_res):

                    i_discrete = i * discrete_ratio
                    j_discrete = j * discrete_ratio
                    k_discrete = k * discrete_ratio

                    color = color_space_func(i_discrete, j_discrete, k_discrete)

                    curr_cube = Cube(
                        side_length=side_length, fill_color=color, fill_opacity=1
                    ).shift((LEFT * i + UP * j + OUT * k) * offset)

                    cubes.append(curr_cube)

        cubes_vg = Group(*cubes)

        return cubes_vg


class ImageUtils(Scene):
    def construct(self):
        NUM_PIXELS = 32
        HEIGHT = 4
        new_image = self.down_sample_image(
            "duck", NUM_PIXELS, NUM_PIXELS, image_height=HEIGHT
        )
        print("New down-sampled image shape:", new_image.get_pixel_array().shape)
        self.wait()
        self.add(new_image)
        self.wait()
        pixel_grid = self.get_pixel_grid(new_image, NUM_PIXELS)
        self.add(pixel_grid)
        self.wait()
        pixel_array = new_image.get_pixel_array().copy()

        pixel_array[-1, 6, :] = [0, 0, 255, 255]
        pixel_array[-2, 5, :] = [0, 0, 255, 255]
        pixel_array[-2, 6, :] = [0, 0, 255, 255]
        pixel_array[-1, 5, :] = [0, 0, 255, 255]

        adjusted_image = self.get_image_mob(pixel_array, height=HEIGHT)
        # self.remove(pixel_grid)
        # self.wait()
        self.play(
            new_image.animate.become(adjusted_image),
            # pixel_grid.animate.become(pixel_grid.copy())
        )
        self.wait()

    def get_image_mob(self, pixel_array, height=4):
        """
        @param pixel_array: multi-dimensional np.array[uint8]
        @return: ImageMobject of pixel array with given height
        """
        image = ImageMobject(pixel_array)
        # height of value None will just return original image mob size
        if height:
            image.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
            image.height = height
        return image

    def down_sample_image(
        self, filepath, num_horiz_pixels, num_vert_pixels, image_height=4
    ):
        """
        @param: filepath - file name of image to down sample
        @param: num_horiz_pixels - number of horizontal pixels in down sampled image
        @param: num_vert_pixels - number of vertical pixels in down sampled image
        """
        assert (
            num_horiz_pixels == num_vert_pixels
        ), "Non-square downsampling not supported"
        original_image = ImageMobject(filepath)
        original_image_pixel_array = original_image.get_pixel_array()
        width, height, num_channels = original_image_pixel_array.shape
        horizontal_slice = self.get_indices(width, num_horiz_pixels)
        vertical_slice = self.get_indices(height, num_vert_pixels)
        new_pixel_array = self.sample_pixel_array_from_slices(
            original_image_pixel_array, horizontal_slice, vertical_slice
        )
        new_image = ImageMobject(new_pixel_array)
        new_image.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
        new_image.height = image_height
        assert (
            new_pixel_array.shape[0] == num_horiz_pixels
            and new_pixel_array.shape[1] == num_vert_pixels
        ), self.get_assert_error_message(
            new_pixel_array, num_horiz_pixels, num_vert_pixels
        )
        return new_image

    def sample_pixel_array_from_slices(
        self, original_pixel_array, horizontal_slice, vertical_slice
    ):
        return original_pixel_array[horizontal_slice][:, vertical_slice]

    def get_indices(self, size, num_pixels):
        """
        @param: size of row or column
        @param: down-sampled number of pixels
        @return: an array of indices with len(array) = num_pixels
        """
        array = []
        index = 0
        float_index = 0
        while float_index < size:
            index = round(float_index)
            array.append(index)
            float_index += size / num_pixels
        return np.array(array)

    def get_assert_error_message(
        self, new_pixel_array, num_horiz_pixels, num_vert_pixels
    ):
        return f"Resizing performed incorrectly: expected {num_horiz_pixels} x {num_vert_pixels} but got {new_pixel_array.shape[0]} x {new_pixel_array.shape[1]}"

    def get_pixel_grid(self, image, num_pixels_in_dimension, color=WHITE):
        side_length_single_cell = image.height / num_pixels_in_dimension
        pixel_grid = VGroup(
            *[
                Square(side_length=side_length_single_cell).set_stroke(
                    color=color, width=1, opacity=0.5
                )
                for _ in range(num_pixels_in_dimension ** 2)
            ]
        )
        pixel_grid.arrange_in_grid(rows=num_pixels_in_dimension, buff=0)
        return pixel_grid


    def get_yuv_image_from_rgb(self, pixel_array, mapped=True):
        """
        Extracts the Y, U and V channels from a given image.

        @param: pixel_array - the image to be processed
        @param: mapped - boolean. if true, return the YUV data mapped back to RGB for presentation purposes. Otherwise,
        return the y, u, and v channels directly for further processing, such as chroma subsampling.
        """
        # discard alpha channel
        rgb_img = pixel_array[:, :, :3]
        # channels need to be flipped to BGR for openCV processing
        rgb_img = rgb_img[:, :, [2, 1, 0]]
        img_yuv = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2YUV)
        y, u, v = cv2.split(img_yuv)

        if not mapped:
            return y, u, v
        else:
            lut_u, lut_v = make_lut_u(), make_lut_v()

            # Convert back to BGR so we display the images
            y = cv2.cvtColor(y, cv2.COLOR_GRAY2BGR)
            u = cv2.cvtColor(u, cv2.COLOR_GRAY2BGR)
            v = cv2.cvtColor(v, cv2.COLOR_GRAY2BGR)

            u_mapped = cv2.LUT(u, lut_u)
            v_mapped = cv2.LUT(v, lut_v)

            # Flip channels to RGB
            y_rgb = y[:, :, [2, 1, 0]]
            u_mapped_rgb = u_mapped[:, :, [2, 1, 0]]
            v_mapped_rgb = v_mapped[:, :, [2, 1, 0]]

            return y_rgb, u_mapped_rgb, v_mapped_rgb

    def chroma_subsample_image(self, pixel_array, mode="4:2:2"):
        """
        Applies chroma subsampling to the image. Modes supported are the most common ones: 4:2:2 and 4:2:0.

        @param: pixel_array - the image to be processed
        @param: mode - a string, either `4:2:2` or `4:2:0`, corresponding to 4:2:2 and 4:2:0 subsampling respectively.
        @param: image - returns back the image in RGB format with subsampling applied
        """
        assert mode in (
            "4:2:2",
            "4:2:0",
        ), "Please choose one of the following {'4:2:2', '4:2:0'}"

        y, u, v = self.get_yuv_image_from_rgb(pixel_array, mapped=False)

        out_u = u.copy()
        out_v = v.copy()
        # Downsample with a window of 2 in the horizontal direction
        if mode == "4:2:2":
            # first the u channel
            for i in range(0, u.shape[0], 2):
                out_u[i : i + 2] = np.mean(u[i : i + 2], axis=0)

            # then the v channel
            for i in range(0, v.shape[0], 2):
                out_v[i : i + 2] = np.mean(v[i : i + 2], axis=0)

        # Downsample with a window of 2 in both directions
        elif mode == "4:2:0":
            for i in range(0, u.shape[0], 2):
                for j in range(0, u.shape[1], 2):
                    out_u[i : i + 2, j : j + 2] = np.mean(u[i : i + 2, j : j + 2])

            for i in range(0, v.shape[0], 2):
                for j in range(0, v.shape[1], 2):
                    out_v[i : i + 2, j : j + 2] = np.mean(v[i : i + 2, j : j + 2])

        ycbcr_sub = np.stack(
            (y, np.round(out_u).astype("uint8"), np.round(out_v).astype("uint8")),
            axis=2,
        )

        return cv2.cvtColor(ycbcr_sub, cv2.COLOR_YUV2RGB)


class IntroChromaSubsampling(ImageUtils):
    def construct(self):
        shed_raw = ImageMobject("shed")
        chroma_subsampled = self.chroma_subsample_image(
            shed_raw.get_pixel_array(), mode="4:2:2"
        )

        chroma_subsampled_mobj = ImageMobject(chroma_subsampled)

        diff_image = shed_raw.get_pixel_array()[:, :, :3] - chroma_subsampled
        diff_image = cv2.cvtColor(diff_image, cv2.COLOR_RGB2GRAY)
        diff_image_mobj = ImageMobject(diff_image)

        img_group = Group(shed_raw, chroma_subsampled_mobj, diff_image_mobj).arrange(
            RIGHT
        )

        self.play(
            FadeIn(img_group.scale(2)),
            run_time=3,
        )
        self.wait(2)

class TestGrayScaleImages(ImageUtils):
    def construct(self):
        pixel_array = np.uint8(
            [[63, 0, 0, 0], [0, 127, 0, 0], [0, 0, 191, 0], [0, 0, 0, 255]]
        )
        image = self.get_image_mob(pixel_array, height=2)

        self.add(image)
        self.wait()

        new_pixel_array = image.get_pixel_array()
        print(new_pixel_array.shape)
        new_pixel_array[3][1] = 255
        new_image = self.get_image_mob(new_pixel_array, height=2)
        self.remove(image)
        self.wait()
        self.add(new_image)
        self.wait()
        next_image_pixel_array = new_image.get_pixel_array()
        next_image_pixel_array[1, 3, :] = [255, 255, 255, 255]
        next_image = self.get_image_mob(next_image_pixel_array, height=2)
        self.remove(new_image)
        self.wait()
        self.add(next_image)
        self.wait()


class TestColorImage(ImageUtils):
    def construct(self):
        NUM_PIXELS = 32
        HEIGHT = 4
        new_image = self.down_sample_image(
            "duck", NUM_PIXELS, NUM_PIXELS, image_height=HEIGHT
        )

        next_image_pixel_array = new_image.get_pixel_array()
        next_image_pixel_array[-1, 0, :] = [255, 255, 255, 255]
        next_image = self.get_image_mob(next_image_pixel_array, height=HEIGHT)
        self.add(next_image)
        self.wait()


class TestYCbCrImages(ImageUtils):
    def construct(self):
        original_image = ImageMobject("shed")
        y, u, v = self.get_yuv_image_from_rgb(original_image.get_pixel_array())
        y_channel = ImageMobject(y)
        u_channel = ImageMobject(u)
        v_channel = ImageMobject(v)

        original_image.move_to(LEFT * 2)
        y_channel.move_to(RIGHT * 2 + UP * 2)
        u_channel.move_to(RIGHT * 2 + UP * 0)
        v_channel.move_to(RIGHT * 2 + DOWN * 2)

        self.add(original_image)
        self.wait()

        self.add(y_channel, u_channel, v_channel)
        self.wait()


class TestYCbCrImagesDuck(ImageUtils):
    def construct(self):
        NUM_PIXELS = 32
        HEIGHT = 2
        new_image = self.down_sample_image(
            "duck", NUM_PIXELS, NUM_PIXELS, image_height=HEIGHT
        )
        y, u, v = self.get_yuv_image_from_rgb(new_image.get_pixel_array())
        y_channel = self.get_image_mob(y, height=HEIGHT)
        u_channel = self.get_image_mob(u, height=HEIGHT)
        v_channel = self.get_image_mob(v, height=HEIGHT)

        new_image.move_to(LEFT * 2)
        y_channel.move_to(RIGHT * 2 + UP * 2)
        u_channel.move_to(RIGHT * 2 + UP * 0)
        v_channel.move_to(RIGHT * 2 + DOWN * 2)

        self.add(new_image)
        self.wait()

        self.add(y_channel, u_channel, v_channel)
        self.wait()


# Animation for representing duck image as signal
class ImageToSignal(ImageUtils):
    NUM_PIXELS = 32
    HEIGHT = 3

    def construct(self):
        image_mob = self.down_sample_image(
            "duck",
            ImageToSignal.NUM_PIXELS,
            ImageToSignal.NUM_PIXELS,
            image_height=ImageToSignal.HEIGHT,
        )
        gray_scale_image_mob = self.get_gray_scale_image(
            image_mob, height=ImageToSignal.HEIGHT
        )
        self.play(FadeIn(gray_scale_image_mob))
        self.wait()

        pixel_grid = self.add_grid(gray_scale_image_mob)

        axes = self.get_axis()

        pixel_row_mob, row_values = self.pick_out_row_of_image(
            gray_scale_image_mob, pixel_grid, 16
        )

        self.play(Write(axes))
        self.wait()

        self.plot_row_values(axes, row_values)

    def get_gray_scale_image(self, image_mob, height=4):
        """
        @param: image_mob -- Mobject.ImageMobject representation of image
        @return: Mobject.ImageMobject of Y (brightness) channel from YCbCr representation
        (equivalent to gray scale)
        """
        y, u, v = self.get_yuv_image_from_rgb(image_mob.get_pixel_array())
        y_channel = self.get_image_mob(y, height=height)
        y_channel.move_to(UP * 2)
        return y_channel

    def add_grid(self, image_mob):
        pixel_grid = self.get_pixel_grid(image_mob, ImageToSignal.NUM_PIXELS)
        pixel_grid.move_to(image_mob.get_center())
        self.play(FadeIn(pixel_grid))
        self.wait()

        return pixel_grid

    def get_axis(self):
        ax = Axes(
            x_range=[0, ImageToSignal.NUM_PIXELS, 1],
            y_range=[0, 255, 1],
            y_length=2.7,
            x_length=10,
            tips=False,
            axis_config={"include_numbers": True, "include_ticks": False},
            x_axis_config={
                "numbers_to_exclude": list(range(1, ImageToSignal.NUM_PIXELS))
            },
            y_axis_config={"numbers_to_exclude": list(range(1, 255))},
        ).move_to(DOWN * 2.2)
        return ax

    def pick_out_row_of_image(self, image_mob, pixel_grid, row):
        pixel_array = image_mob.get_pixel_array()
        pixel_row_mob, row_values = self.get_pixel_row_mob(pixel_array, row)
        pixel_row_mob.next_to(image_mob, DOWN)
        surround_rect = SurroundingRectangle(
            pixel_grid[
                row * ImageToSignal.NUM_PIXELS : row * ImageToSignal.NUM_PIXELS
                + ImageToSignal.NUM_PIXELS
            ],
            buff=0,
        ).set_stroke(width=2, color=PURE_GREEN)
        self.play(Create(surround_rect))
        self.wait()

        self.play(FadeIn(pixel_row_mob))
        self.wait()
        return pixel_row_mob, row_values

    def get_pixel_row_mob(self, pixel_array, row, height=SMALL_BUFF * 3):
        row_values = [pixel_array[row][i][0] for i in range(ImageToSignal.NUM_PIXELS)]
        pixel_row_mob = VGroup(
            *[
                Square(side_length=height)
                .set_stroke(width=1, color=PURE_GREEN)
                .set_fill(color=gray_scale_value_to_hex(value), opacity=1)
                for value in row_values
            ]
        ).arrange(RIGHT, buff=0)
        return pixel_row_mob, row_values

    def plot_row_values(self, axes, pixel_row_values):
        pixel_coordinates = list(enumerate(pixel_row_values))
        axes.add_coordinates()
        path = VGroup()
        path_points = [axes.coords_to_point(x, y) for x, y in pixel_coordinates]
        path.set_points_smoothly(*[path_points]).set_color(YELLOW)
        dots = VGroup(
            *[
                Dot(axes.coords_to_point(x, y), radius=SMALL_BUFF / 2, color=YELLOW)
                for x, y in pixel_coordinates
            ]
        )
        self.play(LaggedStartMap(GrowFromCenter, dots), run_time=3)
        self.wait()
        self.play(Create(path), run_time=4)
        self.wait()


# This class handles animating any row of pixels from an image into a signal
# Handling this differently since in general, pixel counts per row will be high
class GeneralImageToSignal(ImageToSignal):
    def construct(self):
        image_mob = ImageMobject("dog").move_to(UP * 2)
        pixel_array = image_mob.get_pixel_array()
        ImageToSignal.NUM_PIXELS = pixel_array.shape[1]

        ROW = 140
        self.play(FadeIn(image_mob))
        self.wait()
        highlight_row = self.highlight_row(ROW, image_mob)

        row_mob, row_values = self.get_pixel_row_mob(pixel_array, ROW)

        row_mob.next_to(image_mob, DOWN)

        surround_rect = self.show_highlight_to_surround_rect(highlight_row, row_mob)

        axes = self.get_axis()
        self.play(Write(axes))
        self.wait()

        self.show_signal(axes, row_values, row_mob)

    def get_pixel_row_mob(self, pixel_array, row, height=SMALL_BUFF * 3, row_length=13):
        row_values = [pixel_array[row][i][0] for i in range(ImageToSignal.NUM_PIXELS)]
        pixel_row_mob = VGroup(
            *[
                Rectangle(height=height, width=row_length / ImageToSignal.NUM_PIXELS)
                .set_stroke(color=gray_scale_value_to_hex(value))
                .set_fill(color=gray_scale_value_to_hex(value), opacity=1)
                for value in row_values
            ]
        ).arrange(RIGHT, buff=0)
        return pixel_row_mob, row_values

    # draws a line indicating row of image_mob we are highlight
    def highlight_row(self, row, image_mob):
        vertical_pos = (
            image_mob.get_top()
            + DOWN * row / image_mob.get_pixel_array().shape[0] * image_mob.height
        )
        left_bound = vertical_pos + LEFT * image_mob.width / 2
        right_bound = vertical_pos + RIGHT * image_mob.width / 2
        line = Line(left_bound, right_bound).set_color(PURE_GREEN).set_stroke(width=1)
        self.play(Create(line))
        self.wait()
        return line

    def show_highlight_to_surround_rect(self, highlight_row, row_mob):
        surround_rect = SurroundingRectangle(row_mob, buff=0).set_color(
            highlight_row.get_color()
        )
        self.play(
            FadeIn(row_mob), TransformFromCopy(highlight_row, surround_rect), run_time=2
        )
        self.wait()
        return surround_rect

    def show_signal(self, axes, pixel_row_values, pixel_row_mob):
        pixel_coordinates = list(enumerate(pixel_row_values))
        axes.add_coordinates()
        path = VGroup()
        path_points = [axes.coords_to_point(x, y) for x, y in pixel_coordinates]
        path.set_points_smoothly(*[path_points]).set_color(YELLOW)

        arrow = (
            Arrow(DOWN * 0.5, UP * 0.5)
            .set_color(YELLOW)
            .next_to(pixel_row_mob, DOWN, aligned_edge=LEFT, buff=SMALL_BUFF)
        )
        self.play(Write(arrow))
        self.wait()
        new_arrow = arrow.copy().next_to(
            pixel_row_mob, DOWN, aligned_edge=RIGHT, buff=SMALL_BUFF
        )
        self.play(
            Transform(arrow, new_arrow),
            Create(path),
            run_time=5,
            rate_func=linear,
        )
        self.wait()

class DCTExperiments(ImageUtils):
    def construct(self):
        image_mob = ImageMobject("dog").move_to(UP * 2)
        self.play(
            FadeIn(image_mob)
        )
        self.wait()

        print('Image size:', image_mob.get_pixel_array().shape)
        
        self.perform_JPEG(image_mob)

    def perform_JPEG(self, image_mob):
        # Performs encoding/decoding steps on a gray scale block
        block_image, pixel_grid, block = self.highlight_pixel_block(image_mob, 125, 125)
        print('Before\n', block[:, :, 1])
        block_centered = format_block(block)
        print('After centering\n', block_centered)

        dct_block = dct_2d(block_centered)
        np.set_printoptions(suppress=True)
        print('DCT block (rounded)\n', np.round(dct_block, decimals=1))

        heat_map = self.get_heat_map(block_image, dct_block)
        heat_map.move_to(block_image.get_center() + RIGHT * 6)
        self.play(
            FadeIn(heat_map)
        )
        self.wait()

        self.play(
            FadeOut(heat_map)
        )
        self.wait()
        
        quantized_block = quantize(dct_block)
        print('After quantization\n', quantized_block)

        dequantized_block = dequantize(quantized_block)
        print('After dequantize\n', dequantized_block)

        invert_dct_block = idct_2d(dequantized_block)
        print('Invert DCT block\n', invert_dct_block)

        compressed_block = invert_format_block(invert_dct_block)
        print('After reformat\n', compressed_block)

        print('MSE\n', np.mean((compressed_block - block[:, :, 1]) ** 2))

        final_image = self.get_image_mob(compressed_block, height=2)
        final_image.move_to(block_image.get_center() + RIGHT * 6)

        final_image_grid = self.get_pixel_grid(final_image, 8).move_to(final_image.get_center())
        self.play(
            FadeIn(final_image),
            FadeIn(final_image_grid)
        )
        self.wait()

        # self.get_dct_component(0, 0)

    def highlight_pixel_block(self, image_mob, start_row, start_col, block_size=8):
        pixel_array = image_mob.get_pixel_array()
        block = pixel_array[start_row:start_row+block_size, start_col:start_col+block_size]
        center_row = start_row + block_size // 2
        center_col = start_col + block_size // 2
        vertical_pos = (
            image_mob.get_top()
            + DOWN * center_row / pixel_array.shape[0] * image_mob.height
        )
        horizontal_pos = (
            image_mob.get_left()
            + RIGHT * center_col / pixel_array.shape[1] * image_mob.width
        )
        tiny_square_highlight = Square(side_length=SMALL_BUFF * 0.8)
        highlight_position = np.array([horizontal_pos[0], vertical_pos[1], 0])
        tiny_square_highlight.set_color(REDUCIBLE_YELLOW).move_to(highlight_position)

        self.play(
            Create(tiny_square_highlight)
        )
        self.wait()

        block_position = DOWN * 2
        block_image = self.get_image_mob(block, height=2).move_to(block_position)
        pixel_grid = self.get_pixel_grid(block_image, block_size).move_to(block_position)
        surround_rect = SurroundingRectangle(pixel_grid, buff=0).set_color(REDUCIBLE_YELLOW)
        self.play(
            FadeIn(block_image),
            FadeIn(pixel_grid),
            TransformFromCopy(tiny_square_highlight, surround_rect),
        )
        self.wait()

        self.play(
            FadeOut(surround_rect),
            FadeOut(tiny_square_highlight),
            block_image.animate.shift(LEFT * 3),
            pixel_grid.animate.shift(LEFT * 3),
        )
        self.wait()

        return block_image, pixel_grid, block

    def get_heat_map(self, block_image, dct_block):
        block_size = dct_block.shape[0]
        pixel_grid_dct = self.get_pixel_grid(block_image, block_size)
        dct_block_abs = np.abs(dct_block)
        max_dct_coeff = np.amax(dct_block_abs)
        max_color = REDUCIBLE_YELLOW
        min_color = REDUCIBLE_PURPLE
        for i, square in enumerate(pixel_grid_dct):
            row, col = i // block_size, i % block_size
            alpha = dct_block_abs[row][col] / max_dct_coeff
            square.set_fill(color=interpolate_color(min_color, max_color, alpha), opacity=1)
        
        scale = Line(pixel_grid_dct.get_top(), pixel_grid_dct.get_bottom())
        scale.set_stroke(width=10).set_color(color=[min_color, max_color])
        integer_scale = 0.5
        top_value = Integer(round(max_dct_coeff)).scale(integer_scale)
        top_value.next_to(scale, RIGHT, aligned_edge=UP)
        bottom_value = Integer(0).scale(integer_scale)
        bottom_value.next_to(scale, RIGHT, aligned_edge=DOWN)

        heat_map_scale = VGroup(scale, top_value, bottom_value)

        return VGroup(pixel_grid_dct, heat_map_scale).arrange(RIGHT)

class DCTComponents(ImageUtils):
    def construct(self):
        image_mob = ImageMobject("dog").move_to(UP * 2)
        block_image, pixel_grid, block = self.get_pixel_block(image_mob, 125, 125)
        print('Before\n', block[:, :, 1])
        block_image.move_to(LEFT * 2 + DOWN * 2)
        pixel_grid.move_to(block_image.get_center())
        block_centered = format_block(block)
        print('After centering\n', block_centered)

        dct_block = dct_2d(block_centered)
        np.set_printoptions(suppress=True)
        print('DCT block (rounded)\n', np.round(dct_block, decimals=1))


        self.play(
            FadeIn(image_mob)
        )
        self.wait()

        self.play(
            FadeIn(block_image),
            FadeIn(pixel_grid)
        )
        self.wait()
        num_components = 40
        partial_block = self.get_partial_block(dct_block, num_components)
        print(f'Partial block - {num_components} components\n', partial_block)

        partial_block_image = self.get_image_mob(partial_block, height=2)
        partial_pixel_grid = self.get_pixel_grid(partial_block_image, partial_block.shape[0])

        partial_block_image.move_to(RIGHT * 2 + DOWN * 2)
        partial_pixel_grid.move_to(partial_block_image.get_center())

        self.play(
            FadeIn(partial_block_image),
            FadeIn(partial_pixel_grid)
        )
        self.wait()

    def build_final_image_component_wise(self, dct_block):
        original_block = np.zeros((8, 8))

    def get_dct_component(self, row, col):
        text = Tex(f"({row}, {col})").move_to(UP * 2)
        self.add(text)
        dct_matrix = np.zeros((8, 8))
        if row == 0 and col == 0:
            dct_matrix[row][col] = 1016
        else:
            dct_matrix[row][col] = 500
        pixel_array = idct_2d(dct_matrix) + 128
        all_in_range = (pixel_array >= 0) & (pixel_array <= 255)
        if not all(all_in_range.flatten()):
            print('Bad array\n', pixel_array)
            raise ValueError("All elements in pixel_array must be in range [0, 255]")

        image_mob = self.get_image_mob(pixel_array, height=2)
        pixel_grid = self.get_pixel_grid(image_mob, pixel_array.shape[0])
        self.wait()
        self.play(
            FadeIn(image_mob)
        )
        self.add(pixel_grid)
        self.wait()

        self.remove(image_mob, pixel_grid, text)

    def get_partial_block(self, dct_block, num_components):
        zigzag = get_zigzag_order()
        dct_matrix = np.zeros((8, 8))
        for basis_comp in range(num_components):
            row, col = zigzag[basis_comp]
            dct_matrix[row][col] = dct_block[row][col]
        
        pixel_array = idct_2d(dct_matrix)
        return invert_format_block(pixel_array)

    def get_pixel_block(self, image_mob, start_row, start_col, block_size=8):
        pixel_array = image_mob.get_pixel_array()
        block = pixel_array[start_row:start_row+block_size, start_col:start_col+block_size]
    
        block_image = self.get_image_mob(block, height=2)
        pixel_grid = self.get_pixel_grid(block_image, block_size)
        
        return block_image, pixel_grid, block

    def display_component(self, dct_matrix, row, col):
        pass

class DCTSliderExperiments(DCTComponents):
    def construct(self):
        image_mob = ImageMobject("dog").move_to(UP * 2)
        block_image, pixel_grid, block = self.get_pixel_block(image_mob, 125, 125)
        print('Before\n', block[:, :, 1])
        block_image.move_to(LEFT * 2 + DOWN * 0.5)
        pixel_grid.move_to(block_image.get_center())
        block_centered = format_block(block)
        print('After centering\n', block_centered)

        dct_block = dct_2d(block_centered)
        np.set_printoptions(suppress=True)
        print('DCT block (rounded)\n', np.round(dct_block, decimals=1))


        self.play(
            FadeIn(image_mob)
        )
        self.wait()

        self.play(
            FadeIn(block_image),
            FadeIn(pixel_grid)
        )
        self.wait()

        number_line = NumberLine(
            x_range=[0, 64, 8],
            length=10,
            color=REDUCIBLE_VIOLET,
            include_numbers=True,
            label_direction=UP,
        ).move_to(DOWN * 2.5)

        self.add(number_line)
        self.wait()

        self.animate_slider(number_line, dct_block, block)

    def animate_slider(self, number_line, dct_block, original_block):
        # count = 1
        # def update_image(tick, dt):
        #     print(dt)
        #     nonlocal count
        #     print(count)
        #     count += 1
        #     print(tick.get_center())
        #     num_components = number_line.point_to_number(tick.get_center())
        #     new_partial_block = self.get_partial_block(dct_block, num_components)
        #     print(f'Partial block - {num_components} components\n')

        #     new_partial_block_image = self.get_image_mob(new_partial_block, height=2)            
        #     new_partial_block_image.move_to(image_pos)
        #     partial_block_image.become(new_partial_block_image)
        
        tick = Triangle().scale(0.2).set_color(REDUCIBLE_YELLOW)
        tick.set_fill(color=REDUCIBLE_YELLOW, opacity=1)

        tracker = ValueTracker(0)
        tick.add_updater(
            lambda m: m.next_to(
                        number_line.n2p(tracker.get_value()),
                        DOWN
                    )
        )
        self.play( 
            FadeIn(tick),
        )
        self.wait()
        image_pos = RIGHT * 2 + DOWN * 0.5
        
        def get_new_block():
            new_partial_block = self.get_partial_block(dct_block, tracker.get_value())
            print(f'Partial block - {tracker.get_value()} components')
            print('MSE', np.mean((new_partial_block - original_block[:, :, 1]) ** 2), '\n')

            new_partial_block_image = self.get_image_mob(new_partial_block, height=2)            
            new_partial_block_image.move_to(image_pos)
            return new_partial_block_image

        partial_block = self.get_partial_block(dct_block, tracker.get_value())
        partial_block_image = always_redraw(get_new_block)
        partial_pixel_grid = self.get_pixel_grid(partial_block_image, partial_block.shape[0])
        partial_pixel_grid.move_to(image_pos)
        self.play(
            FadeIn(partial_block_image),
            FadeIn(partial_pixel_grid)
        )
        self.add_foreground_mobject(partial_pixel_grid)
        self.wait()
        
       
        self.play(
            tracker.animate.set_value(64),
            run_time=10,
            rate_func=linear,
        ),

        self.wait()


    def get_partial_block(self, dct_block, num_components):
        """
        @param: dct_block - dct coefficients of the block
        @param: num_components - float - number of dct components to
        include in partial block
        @return: pixel_array of partial block with num_components of DCT included
        """
        from math import floor
        zigzag = get_zigzag_order()
        dct_matrix = np.zeros((8, 8))
        floor_val = floor(num_components)
        remaining = num_components - floor_val
        for basis_comp in range(floor_val):
            row, col = zigzag[basis_comp]
            dct_matrix[row][col] = dct_block[row][col]
        
        if floor_val < dct_block.shape[0] ** 2:
            row, col = zigzag[floor_val]
            dct_matrix[row][col] = remaining * dct_block[row][col]

        pixel_array = idct_2d(dct_matrix)
        return invert_format_block(pixel_array)

class DCTEntireImageSlider(DCTSliderExperiments):
    def construct(self):
        image_mob = ImageMobject("dog").move_to(LEFT * 2)
        original_pixel_array = image_mob.get_pixel_array()[2:298, 3:331, 0]
        print(original_pixel_array.shape)
        new_pixel_array = self.get_all_blocks(image_mob, 2, 298, 3, 331, 6)
        relevant_section = new_pixel_array[2:298, 3:331]
        new_image = self.get_image_mob(new_pixel_array, height=None).move_to(RIGHT * 2)
        print('MSE\n', np.mean((relevant_section - original_pixel_array) ** 2))

        self.play(
            FadeIn(image_mob),
            FadeIn(new_image),
        )
        # print(new_image.get_pixel_array().shape)
        self.wait()

    def get_all_blocks(self, image_mob, start_row, end_row, start_col, end_col, num_components, block_size=8):
        pixel_array = image_mob.get_pixel_array()
        new_pixel_array = np.zeros((pixel_array.shape[0], pixel_array.shape[1]))
        for i in range(start_row, end_row, block_size):
            for j in range(start_col, end_col, block_size):
                pixel_block = self.get_pixel_block(pixel_array, i, j)
                block_centered = format_block(pixel_block)
                dct_block = dct_2d(block_centered)
                # quantized_block = quantize(dct_block)
                # dequantized_block = dequantize(quantized_block)
                # invert_dct_block = idct_2d(dequantized_block)
                # compressed_block = invert_format_block(invert_dct_block)
                # all_in_range = (compressed_block >= 0) & (compressed_block <= 255)
                # if not all(all_in_range.flatten()):
                #     print(i, j)
                #     print(all_in_range)
                #     print('Bad array\n', compressed_block)
                #     print('Original array\n', pixel_block[:, :, 0])
                #     raise ValueError("All elements in compressed_block must be in range [0, 255]")
                # new_pixel_array[i:i+block_size, j:j+block_size] = compressed_block
                partial_block = self.get_partial_block(dct_block, num_components)
                new_pixel_array[i:i+block_size, j:j+block_size] = partial_block

        return new_pixel_array

    def get_pixel_block(self, pixel_array, start_row, start_col, block_size=8):
        return pixel_array[start_row:start_row+block_size, start_col:start_col+block_size]
        

# Quick test of gray_scale_value_to_hex
class TestHexToGrayScale(Scene):
    def construct(self):
        for i in range(256):
            dot = Dot().set_color(gray_scale_value_to_hex(i))
            self.add(dot)
            self.wait()
            self.remove(dot)


def gray_scale_value_to_hex(value):
    hex_string = hex(value).split("x")[-1]
    if value < 16:
        hex_string = "0" + hex_string
    return "#" + hex_string * 3


def make_lut_u():
    return np.array([[[i, 255 - i, 0] for i in range(256)]], dtype=np.uint8)


def make_lut_v():
    return np.array([[[0, 255 - i, i] for i in range(256)]], dtype=np.uint8)


def dct1D(f, N):
    result = []
    constant = (2 / N) ** 0.5
    for u in range(N):
        component = 0
        if u == 0:
            factor = 1 / np.sqrt(2)
        else:
            factor = 1
        for i in range(N):
            component += (
                constant * factor * np.cos(np.pi * u / (2 * N) * (2 * i + 1)) * f(i)
            )

        result.append(component)

    return result


def f(i):
    return np.cos((2 * i + 1) * 3 * np.pi / 16)


def plot_function(f, N):
    import matplotlib.pyplot as plt

    x = np.arange(0, N, 0.001)
    y = f(x)

    plt.plot(x, y)
    plt.show()


def g(i):
    return np.cos((2 * i + 1) * 7 * np.pi / 16)


def h(i):
    return np.cos((2 * i + 1) * 3 * np.pi / 16) * np.cos((2 * i + 1) * 1.5 * np.pi / 16)


def get_dct_elem(i, j):
    return np.cos(j * (2 * i + 1) * np.pi / 16)


def get_dct_matrix():
    matrix = np.zeros((8, 8))
    for i in range(8):
        for j in range(8):
            matrix[j][i] = get_dct_elem(i, j)

    return matrix


def get_dot_product_matrix():
    dct_matrix = get_dct_matrix()
    dot_product_matrix = np.zeros((8, 8))
    for i in range(8):
        for j in range(8):
            value = np.dot(dct_matrix[i], dct_matrix[j])
            if np.isclose(value, 0):
                dot_product_matrix[i][j] = 0
            else:
                dot_product_matrix[i][j] = value

    return dot_product_matrix

def format_block(block):
    # [0, 255] -> [-128, 127]
    block_centered = block[:, :, 1].astype(float) - 128
    return block_centered

def invert_format_block(block):
    # [-128, 127] -> [0, 255]
    new_block = block + 128
    # in process of dct and inverse dct with quantization, 
    # some values can go out of range
    new_block[new_block > 255] = 255
    new_block[new_block < 0] = 0
    return new_block

def dct_2d(block):
    return fftpack.dct(fftpack.dct(block.T, norm='ortho').T, norm='ortho')

def idct_2d(block):
    return fftpack.idct(fftpack.idct(block.T, norm='ortho').T, norm='ortho')

def quantize(block):
    quant_table = get_quantization_table()
    return (block / quant_table).round().astype(np.int32)

def get_quantization_table():
    quant_table = np.array(
        [
        [16, 11, 10, 16, 24,  40,  51,  61],
        [12, 12, 14, 19, 26,  58,  60,  55],
        [14, 13, 16, 24, 40,  57,  69,  56],
        [14, 17, 22, 29, 51,  87,  80,  62],
        [18, 22, 37, 56, 68,  109, 103, 77],
        [24, 35, 55, 64, 81,  104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99],
        ]
    )
    return quant_table

def dequantize(block):
    quant_table = get_quantization_table()
    return (block * quant_table).astype(np.float)

def rgb2ycbcr(r, g, b):  # in (0,255) range
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 128 - 0.168736 * r - 0.331364 * g + 0.5 * b
    cr = 128 + 0.5 * r - 0.418688 * g - 0.081312 * b
    return y, cb, cr


def ycbcr2rgb(y, cb, cr):
    r = y + 1.402 * (cr - 128)
    g = y - 0.34414 * (cb - 128) - 0.71414 * (cr - 128)
    b = y + 1.772 * (cb - 128)
    return r, g, b


def coords2rgbcolor(i, j, k):
    """
    Function to transform coordinates in 3D space to hexadecimal RGB color.

    @param: i - x coordinate
    @param: j - y coordinate
    @param: k - z coordinate
    @return: hex value for the corresponding color
    """
    return rgb_to_hex(
        (
            (i) / 255,
            (j) / 255,
            (k) / 255,
        )
    )


def coords2ycbcrcolor(i, j, k):
    """
    Function to transform coordinates in 3D space to hexadecimal YCbCr color.

    @param: i - x coordinate
    @param: j - y coordinate
    @param: k - z coordinate
    @return: hex value for the corresponding color
    """
    y, cb, cr = rgb2ycbcr(i, j, k)
    return rgb_to_hex(
        (
            (y) / 255,
            (cb) / 255,
            (cr) / 255,
        )
    )

def index2coords(n, base):
    """
    Changes the base of `n` to `base`, assuming n is input in base 10.
    The result is then returned as coordinates in `len(result)` dimensions.

    This function allows us to iterate over the color cubes sequentially, using
    enumerate to index every cube, and convert the index of the cube to its corresponding
    coordinate in space.

    Example: if our ``color_res = 4``:

        - Cube #0 is located at (0, 0, 0)
        - Cube #15 is located at (0, 3, 3)
        - Cube #53 is located at (3, 1, 1)

    So, we input our index, and obtain coordinates.

    @param: n - number to be converted
    @param: base - base to change the input
    @return: list - coordinates that the number represent in their corresponding space
    """
    if base == 10:
        return n

    result = 0
    counter = 0

    while n:
        r = n % base
        n //= base
        result += r * 10 ** counter
        counter += 1

    coords = list(f"{result:03}")
    return coords

def get_zigzag_order(block_size=8):
    return zigzag(block_size)

def zigzag(n):
    '''zigzag rows'''
    def compare(xy):
        x, y = xy
        return (x + y, -y if (x + y) % 2 else y)
    xs = range(n)
    return {n: index for n, index in enumerate(sorted(
        ((x, y) for x in xs for y in xs),
        key=compare
    ))}
