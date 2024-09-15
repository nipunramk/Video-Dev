import sys

### THIS IS A WORKAROUND FOR NOW
### IT REQUIRES RUNNING MANIM FROM INSIDE DIRECTORY VIDEO-DEV
sys.path.insert(1, "common/")

from manim import *
from seam_carving_utils import *
from itertools import product
import numpy as np
from pprint import pprint

config["assets_dir"] = "assets"
from classes import *

np.random.seed(1)


class ShowRandomSeam(Scene):
    def construct(self):
        pixel_array = np.random.randint(20, 200, (10, 10))
        pix_arr_mob = PixelArray(img=pixel_array, color_mode="GRAY").scale(0.7)
        self.play(FadeIn(pix_arr_mob))

        # we are going to start at pixel in column 4
        start = 4
        first_pixel_mob = SurroundingRectangle(pix_arr_mob[0, start])
        self.play(
            Write(first_pixel_mob),
            pix_arr_mob[0, start].animate.set_fill(REDUCIBLE_YELLOW, opacity=1),
        )
        last_col = start

        for r in range(1, pixel_array.shape[0]):
            random_n = np.random.random()
            offset_col = -1 if random_n < 0.3 else 1 if random_n > 0.6 else 0

            # clamp the value of current col to be within the bounds of our array
            current_col = (
                last_col + offset_col
                if last_col + offset_col < pixel_array.shape[1]
                else last_col
            )

            current_pixel_mob = pix_arr_mob[r, current_col]
            current_pixel_surr_rect = SurroundingRectangle(current_pixel_mob)
            self.play(
                Transform(first_pixel_mob, current_pixel_surr_rect),
                current_pixel_mob.animate.set_fill(REDUCIBLE_YELLOW, opacity=1),
            )

            last_col = current_col


class AllPossibleSeams(Scene):
    def construct(self):
        pixel_array = np.random.randint(20, 100, (3, 3))
        pix_arr_mob = (
            PixelArray(img=pixel_array, color_mode="GRAY", include_numbers=True)
            .scale(1.3)
            .set_opacity(0.8)
        )

        self.play(FadeIn(pix_arr_mob))

        start = 2

        all_seams = generate_all_seams_recursively(pixel_array, top_pixel=start)

        first_pixel_mob_surr_rect = SurroundingRectangle(pix_arr_mob[0, start])
        first_pixel_mob = pix_arr_mob[0, start].copy()
        self.play(
            Write(first_pixel_mob_surr_rect),
            first_pixel_mob.animate.set_fill(REDUCIBLE_YELLOW, opacity=1),
        )

        for seam in all_seams:
            marked_pixels = VGroup()
            for coord in seam:

                current_pixel_mob = pix_arr_mob[coord].copy()
                marked_pixels.add(current_pixel_mob)

                current_pixel_surr_rect = SurroundingRectangle(
                    current_pixel_mob, buff=0.01
                )
                self.play(
                    Transform(first_pixel_mob_surr_rect, current_pixel_surr_rect),
                    current_pixel_mob.animate.set_fill(REDUCIBLE_YELLOW, opacity=0.7),
                    run_time=1 / config.frame_rate,
                )
            self.play(FadeOut(marked_pixels), run_time=1 / config.frame_rate)

        self.play(FadeOut(first_pixel_mob_surr_rect, first_pixel_mob))

        self.play(FadeOut(pix_arr_mob))

        self.wait()


class ShowMapOfAllSeams(Scene):
    def construct(self):
        pixel_array = np.random.randint(20, 100, (3, 3))
        pix_arr_mob = (
            PixelArray(img=pixel_array, color_mode="GRAY", include_numbers=True)
            .scale(1.3)
            .set_opacity(0.8)
        )

        self.play(FadeIn(pix_arr_mob))

        start = 2

        all_seams = generate_all_seams_recursively(pixel_array, top_pixel=start)

        first_pixel_mob_surr_rect = SurroundingRectangle(pix_arr_mob[0, start])
        first_pixel_mob = pix_arr_mob[0, start].copy()
        self.play(
            Write(first_pixel_mob_surr_rect),
            first_pixel_mob.animate.set_fill(REDUCIBLE_YELLOW, opacity=1),
        )

        all_paths_grid = VGroup()
        for seam in all_seams:
            marked_pixels = VGroup()
            for coord in seam:

                current_pixel_mob = pix_arr_mob[coord].copy()
                marked_pixels.add(current_pixel_mob)

            all_paths_grid.add(
                VGroup(
                    pix_arr_mob.copy().set_opacity(0.1),
                    marked_pixels.set_color(REDUCIBLE_YELLOW),
                )
            )

        self.play(FadeOut(first_pixel_mob_surr_rect, first_pixel_mob))

        self.play(FadeOut(pix_arr_mob))

        self.add(
            all_paths_grid.arrange_in_grid(buff=1).scale_to_fit_height(
                config.frame_height - 0.5
            )
        )
        self.add(
            Text(
                str(
                    len(all_paths_grid),
                )
                + " total paths",
                font=REDUCIBLE_FONT,
                weight=BOLD,
            )
            .scale(0.4)
            .next_to(all_paths_grid, RIGHT, aligned_edge=DOWN)
        )

        self.wait()


class BuildDynamicArray(Scene):
    def construct(self):
        img_shape = (3, 3)
        pixel_array = np.random.randint(20, 100, img_shape)
        pix_arr_mob = PixelArray(
            img=pixel_array, color_mode="GRAY", include_numbers=True
        )

        new_array = np.zeros(img_shape, dtype=int)
        new_array_mob = PixelArray(new_array, color_mode="GRAY", include_numbers=True)

        VGroup(pix_arr_mob, new_array_mob).arrange(RIGHT, buff=1).scale_to_fit_width(
            config.frame_width - 1
        )

        self.play(FadeIn(pix_arr_mob, new_array_mob))

        for row in reversed(range(img_shape[0])):
            for pixel in range(img_shape[1]):
                curr_pix = (
                    pix_arr_mob[row, pixel]
                    .copy()
                    .set_color(REDUCIBLE_VIOLET)
                    .set_opacity(0.3)
                )

                self.play(new_array_mob.update_index((row, pixel), pixel))

                self.play(
                    FadeIn(curr_pix),
                    run_time=3 / config.frame_rate,
                )

        self.wait()
