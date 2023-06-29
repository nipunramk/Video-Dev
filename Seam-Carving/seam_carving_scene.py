import sys

### THIS IS A WORKAROUND FOR NOW
### IT REQUIRES RUNNING MANIM FROM INSIDE DIRECTORY VIDEO-DEV
sys.path.insert(1, "common/")

from manim import *
from seam_carving_utils import *
from reducible_colors import *
from functions import *

DEFAULT_SCALE = 0.8
DEFAULT_FREQ = 2

config["assets_dir"] = "assets"
from matplotlib import pyplot as plt
import itertools as it
from PIL import Image
from manim import get_full_raster_image_path
from scipy.ndimage.filters import convolve


class TestEnergy(Scene):
    def construct(self):
        image = ImageMobject("seam_carving_test", image_mode="RGB")
        # self.play(FadeIn(image))
        # self.wait()
        pixel_array = image.get_pixel_array()
        pixel_array = pixel_array[:, :, :3]
        print(pixel_array.shape)
        energy_array = calc_energy(pixel_array)
        energy_array_remapped = (energy_array / np.max(energy_array)) * 255
        print(energy_array_remapped)
        energy_image = ImageMobject(energy_array_remapped, image_mode="L")
        energy_image.move_to(ORIGIN)
        self.play(FadeIn(energy_image))
        self.wait()


class TestCarving(Scene):
    def construct(self):
        image = ImageMobject("seam_carving_test", image_mode="RGB").move_to(UP * 1.8)
        self.play(FadeIn(image))
        self.wait()
        pixel_array = image.get_pixel_array()
        pixel_array = pixel_array[:, :, :3]
        new_img_array = crop_c(pixel_array, 0.8)
        new_image_object = ImageMobject(new_img_array)
        print(new_img_array.shape)
        new_image_object.next_to(image, DOWN)
        self.play(FadeIn(new_image_object))
        self.wait()


class Convolution(Scene):
    def construct(self):
        pixel_grid, image_array = get_pixel_grid_vgroup("duck_16_16_nn", 5, 5)
        pixel_grid.move_to(LEFT * 3.5)
        self.play(FadeIn(pixel_grid))
        self.wait()

        self.show_kernel_convolution(pixel_grid, image_array)
        self.clear()

        image_array = self.get_custom_image_array(16, 16)
        print(image_array.shape)
        pixel_grid = get_pixel_grid_from_matrix(image_array)
        pixel_grid.set_height(5)
        pixel_grid.set_max_width(5)
        pixel_grid.set_stroke(WHITE, width=1.0, opacity=1.0)

        pixel_grid.move_to(LEFT * 3.5)
        self.play(FadeIn(pixel_grid))
        self.wait()

    def get_custom_image_array(self, width, height):
        array = np.zeros((16, 16, 3), dtype="int")
        for i in range(width):
            for j in range(height):
                if i + j >= 8 and i + j <= 24 and j - i <= 8 and i - j <= 8:
                    color = [128, 128, 0]
                else:
                    color = [0, 0, 0]
                array[i][j] = np.array(color)
        return np.array(array)

    def show_kernel_convolution(self, pixel_grid, image_array):
        sobel_filter_x = np.array(
            [
                [1.0, 2.0, 1.0],
                [0.0, 0.0, 0.0],
                [-1.0, -2.0, -1.0],
            ]
        )

        sobel_filter_y = np.array(
            [
                [1.0, 0.0, -1.0],
                [2.0, 0.0, -2.0],
                [1.0, 0.0, -1.0],
            ]
        )

        x_kernel, y_kernel = self.get_filter_pixel_array(
            sobel_filter_x
        ), self.get_filter_pixel_array(sobel_filter_y)
        kernels = VGroup(x_kernel, y_kernel).arrange(DOWN)
        kernels.move_to(RIGHT * 3.5)

        self.play(FadeIn(kernels))
        self.wait()

        kernel_width, kernel_height = sobel_filter_x.shape
        kernel_translated_height = pixel_grid[0].height * kernel_height
        self.play(
            x_kernel.animate.scale_to_fit_height(kernel_translated_height).move_to(
                pixel_grid[0]
            ),
            FadeOut(y_kernel),
        )
        self.wait()

        x_kernel_3d = np.stack([sobel_filter_x] * 3, axis=2)
        y_kernel_3d = np.stack([sobel_filter_y] * 3, axis=2)

        convolved_array_x = np.absolute(convolve(image_array, x_kernel_3d))
        energy_map_x = convolved_array_x.sum(axis=2)
        convolved_array_y = np.absolute(convolve(image_array, y_kernel_3d))
        energy_map_y = convolved_array_y.sum(axis=2)

        energy_map_remapped_x = (energy_map_x / np.max(energy_map_x)) * 255
        energy_map_remapped_y = (energy_map_y / np.max(energy_map_y)) * 255

        color_func = lambda x: interpolate_color(
            REDUCIBLE_PURPLE, REDUCIBLE_YELLOW, int(x) / 255
        )
        energy_map_pixel_grid_x = get_pixel_grid_from_matrix(
            energy_map_remapped_x, custom_color_func=color_func
        )
        energy_map_pixel_grid_y = get_pixel_grid_from_matrix(
            energy_map_remapped_y, custom_color_func=color_func
        )

        energy_map_pixel_grid_x.set_height(pixel_grid.height)
        energy_map_pixel_grid_x.set_width(pixel_grid.width)
        energy_map_pixel_grid_x.move_to(RIGHT * 3.5)
        self.play(FadeIn(energy_map_pixel_grid_x))
        self.wait()
        self.play(FadeOut(energy_map_pixel_grid_x))
        self.wait()

        y_kernel.move_to(x_kernel).scale_to_fit_height(x_kernel.height)
        self.play(FadeTransform(x_kernel, y_kernel))

        energy_map_pixel_grid_y.set_height(pixel_grid.height)
        energy_map_pixel_grid_y.set_width(pixel_grid.width)
        energy_map_pixel_grid_y.move_to(RIGHT * 3.5)
        self.play(FadeIn(energy_map_pixel_grid_y))
        self.wait()
        self.play(FadeOut(energy_map_pixel_grid_y))
        self.wait()

        combined_energy_map = energy_map_x + energy_map_y
        combined_energy_map_remapped = (
            combined_energy_map / np.max(combined_energy_map)
        ) * 255

        energy_map_pixel_grid_xy = get_pixel_grid_from_matrix(
            combined_energy_map_remapped, custom_color_func=color_func
        )
        energy_map_pixel_grid_xy.scale_to_fit_height(energy_map_pixel_grid_y.height)
        energy_map_pixel_grid_xy.move_to(energy_map_pixel_grid_y.get_center())

        self.play(FadeIn(energy_map_pixel_grid_xy))
        self.wait()

    def get_filter_pixel_array(self, filter, width=3, height=3):
        grid, _ = get_heat_map_from_matrix(filter, width=width, height=height)
        return grid


def get_pixel_grid_vgroup(image_file, image_height, max_width):
    im_path = get_full_raster_image_path(image_file)
    image = Image.open(im_path)
    array = np.array(image)[:, :, :3]

    pixel_array = get_pixel_grid_from_matrix(array)
    pixel_array.set_height(image_height)
    pixel_array.set_max_width(max_width)
    pixel_array.set_stroke(WHITE, width=1.0, opacity=1.0)

    return pixel_array, array


def get_pixel_grid_from_matrix(array, custom_color_func=None):
    height, width = array.shape[:2]
    pixel_array = VGroup(*[Square() for _ in range(height * width)]).arrange_in_grid(
        rows=width, cols=height, buff=0
    )
    for pixel, value in zip(pixel_array, it.chain(*array)):
        if custom_color_func is not None:
            color_func = custom_color_func
        elif isinstance(value, int) or isinstance(value, float):
            color_func = gray_scale_value_to_hex
        else:
            color_func = rgb2hex
        pixel.set_fill(color=color_func(value), opacity=1.0)
    return pixel_array


def rgb2hex(rgb):
    r, g, b = rgb
    return "#{:02x}{:02x}{:02x}".format(r, g, b)


def get_heat_map_from_matrix(
    matrix,
    height=3,
    width=3,
    min_color=REDUCIBLE_PURPLE,
    max_color=REDUCIBLE_YELLOW,
    integer_scale=0.3,
    grid_cell_stroke_width=1,
    color=WHITE,
):
    min_value, max_value = np.min(matrix), np.max(matrix)
    rows, cols = matrix.shape

    grid = get_grid(
        rows, cols, height, width, stroke_width=grid_cell_stroke_width, color=color
    )

    for i in range(rows):
        for j in range(cols):
            alpha = (matrix[i][j] - min_value) / (max_value - min_value)
            grid[i][j].set_fill(
                interpolate_color(min_color, max_color, alpha), opacity=1
            )

    scale = Line(grid.get_top(), grid.get_bottom())
    scale.set_stroke(width=10).set_color(color=[min_color, max_color])

    top_value = Text(str(int(max_value)), font="SF Mono").scale(integer_scale)
    top_value.next_to(scale, RIGHT, aligned_edge=UP)
    bottom_value = Text(str(int(min_value)), font="SF Mono").scale(integer_scale)
    bottom_value.next_to(scale, RIGHT, aligned_edge=DOWN)

    heat_map_scale = VGroup(scale, top_value, bottom_value)
    heat_map_scale.next_to(grid, LEFT)

    return VGroup(grid, heat_map_scale)


def get_grid(rows, cols, height, width, color=WHITE, stroke_width=1):
    cell_height = height / rows
    cell_width = width / cols
    grid = VGroup(
        *[
            VGroup(
                *[
                    Rectangle(height=cell_height, width=cell_width).set_stroke(
                        color=color, width=stroke_width
                    )
                    for j in range(cols)
                ]
            ).arrange(RIGHT, buff=0)
            for i in range(rows)
        ]
    ).arrange(DOWN, buff=0)
    return grid
