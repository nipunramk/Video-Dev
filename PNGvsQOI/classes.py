from manim import *
from functions import *
from reducible_colors import *

class Pixel(Square):
    def __init__(self, n: int, color_mode: str):
        assert color_mode in ("RGB", "GRAY"), "Color modes are RGB and GRAY"
        if color_mode == "RGB":
            color = rgb_to_hex(n / 255)
        else:
            color = g2h(n / 255)

        super().__init__(side_length=1)

        self.set_stroke(BLACK, width=0.2)
        self.set_fill(color, opacity=1)
        self.color = color


class PixelArray(VGroup):
    def __init__(self, img: np.ndarray, include_numbers=False, color_mode="RGB"):
        self.img = img
        if len(img.shape) == 3:
            rows, cols, channels = img.shape
        else:
            rows, cols = img.shape

        self.shape = img.shape

        pixels = []
        for row in img:
            for p in row:
                if include_numbers:
                    self.number = (
                        Text(str(p), font="SF Mono", weight=MEDIUM)
                        .scale(0.7)
                        .set_color(g2h(1) if p < 180 else g2h(0))
                    )
                    pixels.append(VGroup(Pixel(p, color_mode), self.number))
                else:
                    print(p)
                    pixels.append(Pixel(p, color_mode))

        super().__init__(*pixels)
        self.arrange_in_grid(rows, cols, buff=0)

        self.dict = {index: p for index, p in enumerate(self)}

    def __getitem__(self, value) -> VGroup:
        if isinstance(value, slice):
            return VGroup(*list(self.dict.values())[value])
        elif isinstance(value, tuple):
            i, j = value
            one_d_index = get_1d_index(i, j, self.img)
            return self.dict[one_d_index]
        else:
            return self.dict[value]

class Byte(VGroup):
    def __init__(
        self,
        text,
        stroke_color=REDUCIBLE_VIOLET,
        stroke_width=5,
        text_scale=0.5,
        h_buff=MED_SMALL_BUFF+SMALL_BUFF,
        v_buff=MED_SMALL_BUFF,
        width=6,
        height=1.5,
        edge_buff=1,
        **kwargs,
    ):

        self.h_buff = h_buff
        self.text_scale = text_scale
        self.rect = Rectangle(
            height=height, width=width
        ).set_stroke(width=stroke_width, color=stroke_color)

        if isinstance(text, list):
            text_mobs = []
            for string in text:
                text_mobs.append(self.get_text_mob(string))
            self.text = VGroup(*text_mobs).arrange(DOWN, buff=v_buff)
        else:
            self.text = self.get_text_mob(text)

        self.text.move_to(self.rect.get_center())
        self.text.scale_to_fit_width(self.rect.width - edge_buff)
        super().__init__(self.rect, self.text, **kwargs)


    def get_text_mob(self, string):
        text = VGroup(*[Text(c, font="SF Mono", weight=MEDIUM).scale(self.text_scale) for c in string.split(',')])
        if len(text) > 1:
            text.arrange(RIGHT, buff=self.h_buff)
        return text


