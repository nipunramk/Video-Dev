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
        pix_arr_mob = PixelArray(
            img=pixel_array, color_mode="GRAY", include_numbers=True
        ).scale(0.7)
        self.add(pix_arr_mob.scale(2))

        all_seams = generate_all_seams_recursively(pixel_array, top_pixel=2)
