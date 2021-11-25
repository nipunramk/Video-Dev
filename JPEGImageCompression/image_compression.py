from manim import *
import cv2


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
            coords2ycbcrcolor, color_res=8, cube_side_length=1
        )
        self.wait(2)
        self.add(cubes_vg)
        self.wait(2)

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

        max_color_res = 256
        discrete_ratio = max_color_res // color_res

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

                    cubes.append(
                        Cube(
                            side_length=side_length, fill_color=color, fill_opacity=1
                        ).shift((LEFT * i + UP * j + OUT * k) * offset)
                    )

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

    def get_yuv_image_from_rgb(self, pixel_array):

        # discard alpha channel
        rgb_img = pixel_array[:, :, :3]
        # channels need to be flipped to BGR for openCV processing
        rgb_img = rgb_img[:, :, [2, 1, 0]]
        img_yuv = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2YUV)
        y, u, v = cv2.split(img_yuv)

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
