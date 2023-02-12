import sys

### THIS IS A WORKAROUND FOR NOW
### IT REQUIRES RUNNING MANIM FROM INSIDE DIRECTORY VIDEO-DEV
sys.path.insert(1, "common/")

from manim import *
from seam_carving_utils import *

DEFAULT_SCALE = 0.8
DEFAULT_FREQ = 2

config["assets_dir"] = "assets"
from matplotlib import pyplot as plt


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
