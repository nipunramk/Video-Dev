import sys

### THIS IS A WORKAROUND FOR NOW
### IT REQUIRES RUNNING MANIM FROM INSIDE DIRECTORY VIDEO-DEV
sys.path.insert(1, "common/")

from manim import *
from seam_carving_utils import *
from itertools import product
import numpy as np

config["assets_dir"] = "assets"
from classes import *


class TestingSeams(Scene):
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
        pixel_array = np.random.randint(20, 200, (4, 4))
        pix_arr_mob = PixelArray(img=pixel_array, color_mode="GRAY").scale(0.7)
        self.play(FadeIn(pix_arr_mob))

        for c in range(pixel_array.shape[1]):
            all_seams = self.generate_all_possible_seams(pixel_array.shape)

            first_pixel_mob_surr_rect = SurroundingRectangle(pix_arr_mob[0, c])
            col_marker = pix_arr_mob[0, c].copy().set_fill(REDUCIBLE_VIOLET)

            self.play(Write(first_pixel_mob_surr_rect), Write(col_marker))

            last_col = c
            for seam in all_seams:
                marked_pixels = VGroup()
                print(seam)
                for row_index, col_offset in enumerate(seam):

                    # clamp the value of current col to be within the bounds of our array
                    current_col = last_col + col_offset

                    # we skip every invalid combination, otherwise it'd be animated
                    # such a waste of cpu time
                    if current_col > pixel_array.shape[1] or current_col < 0:
                        continue

                    print(row_index, current_col)

                    current_pixel_mob = pix_arr_mob[row_index + 1, current_col]
                    current_pixel_surr_rect = SurroundingRectangle(current_pixel_mob)
                    curr_marked_pixel = current_pixel_mob.copy().set_fill(
                        REDUCIBLE_YELLOW, opacity=0.3
                    )

                    marked_pixels.add(curr_marked_pixel)
                    self.play(
                        Transform(first_pixel_mob_surr_rect, current_pixel_surr_rect),
                        FadeIn(curr_marked_pixel),
                    )

                    last_col = current_col

                self.play(FadeOut(marked_pixels, first_pixel_mob_surr_rect))

    def generate_all_possible_seams(self, shape: tuple):
        return product([-1, 0, 1], repeat=shape[0] - 1)
