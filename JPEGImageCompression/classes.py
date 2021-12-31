"""
File to define all the classes used in the animations.
"""


from manim import *
from typing import Iterable, List
from reducible_colors import *
from functions import *


class ReducibleBarChart(BarChart):
    """
    Redefinition of the BarChart class to add font personalization
    """

    def __init__(
        self,
        values: Iterable[float],
        height: float = 4,
        width: float = 6,
        n_ticks: int = 4,
        tick_width: float = 0.2,
        chart_font: str = "SF Mono",
        label_y_axis: bool = True,
        y_axis_label_height: float = 0.25,
        max_value: float = 1,
        bar_colors=...,
        bar_fill_opacity: float = 0.8,
        bar_stroke_width: float = 3,
        bar_names: List[str] = ...,
        bar_label_scale_val: float = 0.75,
        **kwargs,
    ):
        self.chart_font = chart_font

        super().__init__(
            values,
            height=height,
            width=width,
            n_ticks=n_ticks,
            tick_width=tick_width,
            label_y_axis=label_y_axis,
            y_axis_label_height=y_axis_label_height,
            max_value=max_value,
            bar_colors=bar_colors,
            bar_fill_opacity=bar_fill_opacity,
            bar_stroke_width=bar_stroke_width,
            bar_names=bar_names,
            bar_label_scale_val=bar_label_scale_val,
            **kwargs,
        )

    def add_axes(self):
        x_axis = Line(self.tick_width * LEFT / 2, self.total_bar_width * RIGHT)
        y_axis = Line(ORIGIN, self.total_bar_height * UP)
        ticks = VGroup()
        heights = np.linspace(0, self.total_bar_height, self.n_ticks + 1)
        values = np.linspace(0, self.max_value, self.n_ticks + 1)
        for y, _value in zip(heights, values):
            tick = Line(LEFT, RIGHT)
            tick.width = self.tick_width
            tick.move_to(y * UP)
            ticks.add(tick)
        y_axis.add(ticks)

        self.add(x_axis, y_axis)
        self.x_axis, self.y_axis = x_axis, y_axis

        if self.label_y_axis:
            labels = VGroup()
            for tick, value in zip(ticks, values):
                label = Text(str(np.round(value, 2)), font=self.chart_font)
                label.height = self.y_axis_label_height
                label.next_to(tick, LEFT, SMALL_BUFF)
                labels.add(label)
            self.y_axis_labels = labels
            self.add(labels)

    def add_bars(self, values):
        buff = float(self.total_bar_width) / (2 * len(values) + 1)
        bars = VGroup()
        for i, value in enumerate(values):
            bar = Rectangle(
                height=(value / self.max_value) * self.total_bar_height,
                width=buff,
                stroke_width=self.bar_stroke_width,
                fill_opacity=self.bar_fill_opacity,
            )
            bar.move_to((2 * i + 1) * buff * RIGHT, DOWN + LEFT)
            bars.add(bar)
        bars.set_color_by_gradient(*self.bar_colors)

        bar_labels = VGroup()
        for bar, name in zip(bars, self.bar_names):
            label = Text(str(name), font="SF Mono")
            label.scale(self.bar_label_scale_val)
            label.next_to(bar, DOWN, SMALL_BUFF)
            bar_labels.add(label)

        self.add(bars, bar_labels)
        self.bars = bars
        self.bar_labels = bar_labels


class Module(VGroup):
    def __init__(
        self,
        text,
        fill_color=REDUCIBLE_PURPLE_DARKER,
        stroke_color=REDUCIBLE_VIOLET,
        stroke_width=5,
        text_scale=0.9,
        text_position=ORIGIN,
        text_weight=NORMAL,
        width=4,
        height=2,
        **kwargs,
    ):

        self.rect = (
            RoundedRectangle(
                corner_radius=0.1, fill_color=fill_color, width=width, height=height
            )
            .set_opacity(1)
            .set_stroke(stroke_color, width=stroke_width)
        )

        self.text = Text(str(text), weight=text_weight, font="CMU Serif").scale(
            text_scale
        )
        self.text.next_to(
            self.rect,
            direction=ORIGIN,
            coor_mask=text_position * 0.8,
            aligned_edge=text_position,
        )

        super().__init__(self.rect, self.text, **kwargs)
        # super().arrange(ORIGIN)


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
                        Text(str(p), font="SF Mono")
                        .scale(0.7)
                        .set_color(g2h(1) if p < 180 else g2h(0))
                    )
                    pixels.append(VGroup(Pixel(p, color_mode), self.number))
                else:
                    pixels.append(Pixel(p, color_mode))

        super().__init__(*pixels)
        self.arrange_in_grid(rows, cols, buff=0)

        self.dict = {index: p for index, p in enumerate(self)}

    def __getitem__(self, value) -> VGroup:
        if isinstance(value, slice):
            return VGroup(*list(self.dict.values())[value])
        else:
            return self.dict[value]
