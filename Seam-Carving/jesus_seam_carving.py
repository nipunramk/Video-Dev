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
        pixel_array = np.random.randint(20, 200, (3, 3))
        pix_arr_mob = PixelArray(
            img=pixel_array, color_mode="GRAY", include_numbers=True
        ).scale(0.7)
        self.add(pix_arr_mob.scale(2))

        self.generate_all_seams_recursively(pixel_array, top_pixel=2)

    def generate_all_possible_seams(self, top_pixel=2, shape: tuple = (5, 5)):
        """
        Generate all possible seams for a single top pixel. That is, given the start of the seam,
        the top pixel, what are all the possible seams under it.
        """

        all_combinations = product([-1, 0, 1], repeat=shape[0])
        filtered_combinations = filter(lambda x: x[0] == 0, all_combinations)

        def overall_seam_span(arr):
            """
            Returns the maximum, or overall span of a given seam.
            If a seam is defined by a top pixel and an array of [-1,0,1] values that define
            what is the offset with respect to the last column, return how much a given sequence
            extends either to the left (negative value) or right (positive)

            For example, the sequence [0, -1, -1, -1, -1, 1, 1] should output -4, since
            there are 4 consecutive steps to the left.
            """
            arrays = []
            sub_array = []
            # Traverse the array
            for i in range(len(arr) - 1):
                # If element is same as previous
                # increment temp value
                if arr[i] == arr[i + 1]:
                    sub_array.append(arr[i])
                    print("sub: ", sub_array)
                else:
                    sub_array.append(arr[i])
                    arrays.append(sub_array)

                    sub_array = []

            # this means we were already done with the last sequence
            # of equal elements. thus, the last element must be different
            # and included in its own array. otherwise, the element belong to the
            # last subarray created and we proceed as usual.
            if len(sub_array) == 0:
                arrays.append([arr[-1]])
            else:
                sub_array.append(arr[-1])
                arrays.append(sub_array)

            sums = list(map(sum, arrays))
            absolutes = list(map(abs, sums))
            biggest_absolute = np.argmax(absolutes)

            return sums[biggest_absolute]

        def valid_combination(x):
            right_leeway = (shape[1] - 1) - top_pixel
            left_leeway = top_pixel
            print("rl:", right_leeway, "ll:", left_leeway)

            span = overall_seam_span(x)
            print(span)

            # means we go more to the left
            if np.sign(span) == -1:
                return abs(span) <= left_leeway
            else:
                return abs(span) <= right_leeway

        valid_combinations = filter(valid_combination, filtered_combinations)

        print(list(valid_combinations))

        # print(list(valid_combinations))

        return filtered_combinations

    def generate_all_seams_recursively(self, pixel_array, top_pixel=2):
        rows, cols = pixel_array.shape
        seams_array = []

        def traverse(arr, seam: list, r, c):
            # If the current position is the bottom-right corner of
            # the matrix
            if r >= rows - 1:

                seam.append((r, c))
                seams_array.append(seam[: r + 1])
                seam.pop()

                print(seams_array)
                return

            # Print the value at the current position
            seam.append((r, c))
            print(r, c, arr[r, c])

            # If the end of the current row has not been reached
            if c - 1 >= 0:
                traverse(arr, seam, r + 1, c - 1)

            traverse(arr, seam, r + 1, c)

            if c + 1 < cols:
                traverse(arr, seam, r + 1, c + 1)

            seam.pop()

        traverse(pixel_array, [], 0, top_pixel)
        pprint(seams_array)
        return seams_array
