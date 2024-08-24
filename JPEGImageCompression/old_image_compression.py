from math import sqrt
from manim import *
import cv2

from scipy import fftpack
from typing import Iterable, List

np.random.seed(23)
config["assets_dir"] = "assets"

"""
Make sure you run manim CE with --disable_caching flag
If you run with caching, since there are some scenes that change pixel arrays,
there might be some unexpected behavior
E.g manim -pql JPEGImageCompression/image_compression.py --disable_caching
"""

REDUCIBLE_PURPLE_DARKER = "#3B0893"
REDUCIBLE_BLUE = "#650FFA"
REDUCIBLE_PURPLE = "#8c4dfb"
REDUCIBLE_VIOLET = "#d7b5fe"
REDUCIBLE_YELLOW = "#ffff5c"
REDUCIBLE_YELLOW_DARKER = "#7F7F2D"
REDUCIBLE_GREEN_LIGHTER = "#00cc70"
REDUCIBLE_GREEN = "#008f4f"
REDUCIBLE_GREEN_DARKER = "#004F2C"


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

class IntroduceRGBAndJPEG(Scene):
    def construct(self):
        r_t = Text("R", font="SF Mono").scale(3).set_color(RED)
        g_t = Text("G", font="SF Mono").scale(3).set_color(GREEN)
        b_t = Text("B", font="SF Mono").scale(3).set_color(BLUE)

        rgb_vg_h = VGroup(r_t, g_t, b_t).arrange(RIGHT, buff=2)
        rgb_vg_v = rgb_vg_h.copy().arrange(DOWN, buff=1).shift(LEFT * 0.7)

        self.play(LaggedStartMap(FadeIn, rgb_vg_h, lag_ratio=0.5))
        self.wait()
        self.play(Transform(rgb_vg_h, rgb_vg_v))

        red_t = (
            Text("ed", font="SF Mono")
            .set_color(RED)
            .next_to(r_t, RIGHT, buff=0.3, aligned_edge=DOWN)
        )
        green_t = (
            Text("reen", font="SF Mono")
            .set_color(GREEN)
            .next_to(g_t, RIGHT, buff=0.3, aligned_edge=DOWN)
        )
        blue_t = (
            Text("lue", font="SF Mono")
            .set_color(BLUE)
            .next_to(b_t, RIGHT, buff=0.3, aligned_edge=DOWN)
        )
        self.play(LaggedStartMap(FadeIn, [red_t, green_t, blue_t]))

        self.play(LaggedStartMap(FadeOut, [rgb_vg_h, red_t, green_t, blue_t]))

        # pixels
        black = (
            Square(side_length=1)
            .set_color(BLACK)  # 0
            .set_opacity(1)
            .set_stroke(REDUCIBLE_VIOLET, width=3)
        )
        gray1 = (
            Square(side_length=1)
            .set_color(GRAY_E)  # 34
            .set_opacity(1)
            .set_stroke(REDUCIBLE_VIOLET, width=3)
        )
        gray2 = (
            Square(side_length=1)
            .set_color(GRAY_D)  # 68
            .set_opacity(1)
            .set_stroke(REDUCIBLE_VIOLET, width=3)
        )
        gray3 = (
            Square(side_length=1)
            .set_color(GRAY_B)  # 187
            .set_opacity(1)
            .set_stroke(REDUCIBLE_VIOLET, width=3)
        )
        white = (
            Square(side_length=1)
            .set_color("#FFFFFF")  # 255
            .set_opacity(1)
            .set_stroke(REDUCIBLE_VIOLET, width=3)
        )

        # pixel values

        pixels_vg = VGroup(black, gray1, gray2, gray3, white).arrange(RIGHT, buff=1)

        bk_t = Text("0", font="SF Mono").next_to(black, DOWN, buff=0.5).scale(0.5)
        g1_t = Text("34", font="SF Mono").next_to(gray1, DOWN, buff=0.5).scale(0.5)
        g2_t = Text("68", font="SF Mono").next_to(gray2, DOWN, buff=0.5).scale(0.5)
        g3_t = Text("187", font="SF Mono").next_to(gray3, DOWN, buff=0.5).scale(0.5)
        wh_t = Text("255", font="SF Mono").next_to(white, DOWN, buff=0.5).scale(0.5)

        self.play(LaggedStartMap(FadeIn, pixels_vg))
        self.play(LaggedStartMap(FadeIn, [bk_t, g1_t, g2_t, g3_t, wh_t]))

        self.play(LaggedStartMap(FadeOut, [pixels_vg, bk_t, g1_t, g2_t, g3_t, wh_t]))

        red_channel = (
            Rectangle(RED, width=3)
            .set_color(BLACK)
            .set_opacity(1)
            .set_stroke(RED, width=3)
        )
        green_channel = (
            Rectangle(GREEN, width=3)
            .set_color(BLACK)
            .set_opacity(1)
            .set_stroke(GREEN, width=3)
        )
        blue_channel = (
            Rectangle(BLUE, width=3)
            .set_color(BLACK)
            .set_opacity(1)
            .set_stroke(BLUE, width=3)
        )

        channels_vg_h = VGroup(red_channel, green_channel, blue_channel).arrange(
            RIGHT, buff=0.8
        )

        channels_vg_diagonal = (
            channels_vg_h.copy()
            .arrange(DOWN * 0.7 + RIGHT * 1.3, buff=-1.4)
            .shift(LEFT * 3)
        )

        self.play(LaggedStartMap(FadeIn, channels_vg_h))
        self.wait()
        self.play(Transform(channels_vg_h, channels_vg_diagonal))

        pixel_r = (
            Square(side_length=0.1)
            .set_color(RED)
            .set_opacity(1)
            .align_to(red_channel, LEFT)
            .align_to(red_channel, UP)
        )
        pixel_g = (
            Square(side_length=0.1)
            .set_color(GREEN)
            .set_opacity(1)
            .align_to(green_channel, LEFT)
            .align_to(green_channel, UP)
        )
        pixel_b = (
            Square(side_length=0.1)
            .set_color(BLUE)
            .set_opacity(1)
            .align_to(blue_channel, LEFT)
            .align_to(blue_channel, UP)
        )

        self.play(FadeIn(pixel_r), FadeIn(pixel_g), FadeIn(pixel_b))

        pixel_r_big = pixel_r.copy().scale(5).move_to(ORIGIN + UP * 1.5 + RIGHT * 1.7)
        pixel_g_big = pixel_g.copy().scale(5).next_to(pixel_r_big, DOWN, buff=1)
        pixel_b_big = pixel_b.copy().scale(5).next_to(pixel_g_big, DOWN, buff=1)

        self.play(
            TransformFromCopy(pixel_r, pixel_r_big),
            TransformFromCopy(pixel_g, pixel_g_big),
            TransformFromCopy(pixel_b, pixel_b_big),
        )

        eight_bits_r = (
            Text("8 bits", font="SF Mono")
            .scale(0.4)
            .next_to(pixel_r_big, RIGHT, buff=0.3)
        )

        eight_bits_g = eight_bits_r.copy().next_to(pixel_g_big)
        eight_bits_b = eight_bits_r.copy().next_to(pixel_b_big)

        self.play(FadeIn(eight_bits_r), FadeIn(eight_bits_g), FadeIn(eight_bits_b))

        brace = Brace(VGroup(eight_bits_r, eight_bits_g, eight_bits_b), RIGHT)

        self.play(Write(brace))

        twenty_four_bits = (
            Text("24 bits / pixel", font="SF Mono").scale(0.4).next_to(brace, RIGHT)
        )

        self.play(Write(twenty_four_bits))

        self.play(Transform(twenty_four_bits, twenty_four_bits.copy().shift(UP * 0.5)))

        three_bytes = (
            Text("3 bytes / pixel", font="SF Mono")
            .scale(0.4)
            .next_to(twenty_four_bits, DOWN, buff=0.7)
        )
        self.play(Write(three_bytes))

        self.play(*[FadeOut(mob) for mob in self.mobjects])

        # flower image
        flower_image = ImageMobject("rose.jpg").scale(0.4)

        dimensions = (
            Text("2592 × 1944", font="SF Mono")
            .scale(0.7)
            .next_to(flower_image, DOWN, buff=0.3)
        )

        img_and_dims = Group(flower_image, dimensions).arrange(DOWN)
        img_and_dims_sm = img_and_dims.copy().scale(0.8).to_edge(LEFT, buff=1)

        self.play(FadeIn(img_and_dims))
        self.wait()
        self.play(Transform(img_and_dims, img_and_dims_sm), run_time=2)

        # I don't like how this barchart looks by default, and the fields they exposed
        # dont quite allow for what i want to do. Ask Nipun on how to approach personalization of
        # an existing class.
        chart = (
            ReducibleBarChart(
                [15, 0.8],
                height=6,
                max_value=15,
                n_ticks=4,
                label_y_axis=True,
                y_axis_label_height=0.2,
                bar_label_scale_val=0.5,
                bar_names=["Uncompressed", "Compressed"],
                bar_colors=[REDUCIBLE_PURPLE, REDUCIBLE_YELLOW],
            )
            .scale(0.8)
            .to_edge(RIGHT, buff=1)
        )
        annotation = (
            Text("MB", font="SF Mono").scale(0.4).next_to(chart.y_axis, UP, buff=0.3)
        )

        self.play(Create(chart.x_axis), Create(chart.y_axis), run_time=3)
        self.play(
            Write(chart.y_axis_labels),
            Write(chart.bar_labels),
            Write(annotation),
            run_time=3,
        )

        # makes the bars grow from bottom to top
        for bar in chart.bars:
            self.add(bar)
            bar.generate_target()

            def update(mob, alpha):

                mob.become(mob.target)
                mob.move_to(bar.get_bottom())
                mob.stretch_to_fit_height(
                    alpha * bar.height,
                )
                mob.move_to(bar.get_top())

            self.play(UpdateFromAlphaFunc(bar, update_function=update))

        self.wait(3)

class OLDJPEGDiagram(Scene):
    def construct(self):

        # animation section

        # self.intro()

        # self.play(*[FadeOut(mob) for mob in self.mobjects])

        # self.intro_2()

        self.play(*[FadeOut(mob) for mob in self.mobjects])

        self.data_flow()

        self.play(*[FadeOut(mob) for mob in self.mobjects])

        self.zoom_jpeg_encoder()

        self.play(*[FadeOut(mob) for mob in self.mobjects])

    # definition of animations
    def intro(self):
        sq_array = [Square(color=WHITE) for _ in range(8 * 8)]
        intro_image = (
            VGroup(*sq_array)
            .arrange_in_grid(rows=8, cols=8, buff=0)
            .stretch_to_fit_height(3)
            .stretch_to_fit_width(3)
        )

        intro_image_buff = intro_image.copy().arrange_in_grid(rows=8, cols=8, buff=0.1)

        self.play(LaggedStartMap(GrowFromCenter, intro_image))
        self.wait()
        self.play(Transform(intro_image, intro_image_buff))

        for _ in range(10):
            rand_index = np.random.randint(0, 63)
            self.play(
                Transform(
                    sq_array[rand_index],
                    sq_array[rand_index].copy().set_color(REDUCIBLE_YELLOW),
                )
            )
            self.play(ShrinkToCenter(sq_array[rand_index]), run_time=2)

        self.wait()

    def intro_2(self):

        img_og = self.make_nested_squares(8, 2).scale_to_fit_width(4)

        img_sm = self.make_nested_squares(4, 1).scale_to_fit_width(4)

        self.play(FadeIn(img_og), run_time=2)

        anims = []
        for i in range(len(img_og.submobjects)):
            anims.append(Transform(img_og[i], img_sm[i]))

        self.play(LaggedStart(*anims, lag_ratio=0.05), run_time=10)

        self.wait(3)

    def data_flow(self):
        input_rows = 8
        input_cols = 8

        output_rows = 4
        output_cols = 4

        # object instantiation
        input_image = (
            VGroup(*[Square(color=WHITE) for _ in range(input_rows * input_cols)])
            .arrange_in_grid(rows=input_rows, cols=input_cols, buff=0)
            .stretch_to_fit_height(2)
            .stretch_to_fit_width(2)
        )

        final_image = (
            VGroup(*[Square(color=WHITE) for _ in range(output_rows * output_cols)])
            .arrange_in_grid(rows=output_rows, cols=output_cols, buff=0)
            .stretch_to_fit_height(2)
            .stretch_to_fit_width(2)
        )

        # encoding part
        jpeg_encoder = self.module("JPEG Encoder")
        compressed_data = VGroup(
            Square(),
            DashedLine(
                Square().get_bottom(), Square().get_top(), dash_length=0.1
            ).set_stroke(width=10),
        )

        data_flow_encode = VGroup(input_image, jpeg_encoder, compressed_data).arrange(
            RIGHT, buff=1
        )

        # arrows
        arr1 = Arrow(input_image.get_right(), jpeg_encoder.get_left())
        arr2 = Arrow(jpeg_encoder.get_right(), compressed_data.get_left())
        data_flow_encode.add(arr1, arr2)

        # decoding part
        jpeg_decoder = self.module("JPEG Decoder")

        data_flow_decode = (
            VGroup(compressed_data.copy(), jpeg_decoder, final_image)
            .arrange(RIGHT, buff=1)
            .shift(DOWN * 2)
        )
        arr3 = Arrow(data_flow_decode[0].get_right(), jpeg_decoder.get_left())
        arr4 = Arrow(jpeg_decoder.get_right(), final_image.get_left())
        data_flow_decode.add(arr3, arr4)

        # animations
        self.play(LaggedStartMap(Write, input_image, lag_ratio=0.1), run_time=2)

        self.play(
            Write(jpeg_encoder),
            Write(arr1),
        )
        self.play(
            Write(compressed_data),
            Write(arr2),
        )

        self.wait()

        self.play(data_flow_encode.animate.shift(UP * 2))

        self.wait()

        self.play(LaggedStartMap(Write, data_flow_decode, lag_ratio=0.1), run_time=1)

        self.wait(4)

    def zoom_jpeg_encoder(self):
        jpeg_encoder = self.module("JPEG Encoder")
        self.play(Write(jpeg_encoder.move_to(ORIGIN)))
        self.wait()
        self.play(
            ScaleInPlace(jpeg_encoder, 3),
            FadeOut(jpeg_encoder[1]),
        )

        forward_dct = self.module(
            "Forward DCT", fill_color="#7F7F2D", stroke_color=REDUCIBLE_YELLOW
        )
        forward_dct_icon = ImageMobject("dct.png").scale(0.4)
        forward_dct_module = Group(forward_dct, forward_dct_icon).arrange(DOWN, buff=1)

        quantization = self.module(
            "Quantization", fill_color="#7F7F2D", stroke_color=REDUCIBLE_YELLOW
        )
        quantization_icon = ImageMobject("quantization.png").scale(0.4)
        quantization_module = Group(quantization, quantization_icon).arrange(
            DOWN, buff=1
        )

        lossless_comp = self.module(
            "Lossless \\\\ compression",
            fill_color="#7F7F2D",
            stroke_color=REDUCIBLE_YELLOW,
        )
        lossless_icon = ImageMobject("lossless.png").scale(0.4)
        lossless_module = Group(lossless_comp, lossless_icon).arrange(DOWN, buff=1)

        modules_g = (
            Group(forward_dct_module, quantization_module, lossless_module)
            .arrange(RIGHT)
            .scale_to_fit_width(jpeg_encoder.width - 1)
        )

        self.play(LaggedStartMap(FadeIn, modules_g), run_time=2)

        self.wait(3)

    # definition of util functions
    def make_nested_squares(self, total_side=8, groups_side=2):
        """
        Creates an arrangement of squares based on blocks of `groups_side` squares.
        """
        groups_ratio = total_side ** 2 // groups_side ** 2

        output_vg = VGroup()
        for i in range(groups_ratio):
            aux_vg = VGroup()
            for _ in range(groups_side * groups_side):
                aux_vg.add(Square())

            aux_vg.arrange_in_grid(rows=groups_side, cols=groups_side, buff=0)

            output_vg.add(aux_vg)

        print(len(output_vg))

        return output_vg.arrange_in_grid(
            rows=int(sqrt(groups_ratio)), cols=int(sqrt(groups_ratio)), buff=0
        )


class JPEGDiagram(MovingCameraScene):
    def construct(self):
        self.build_diagram()

    def build_diagram(self):

        self.play(self.camera.frame.animate.scale(2))
        # input image
        red_channel = (
            RoundedRectangle(corner_radius=0.1, fill_color=RED, width=3)
            .set_color(BLACK)
            .set_opacity(1)
            .set_stroke(RED, width=4)
        )
        green_channel = (
            RoundedRectangle(corner_radius=0.1, fill_color=GREEN, width=3)
            .set_color(BLACK)
            .set_opacity(1)
            .set_stroke(GREEN, width=4)
        )
        blue_channel = (
            RoundedRectangle(corner_radius=0.1, fill_color=BLUE, width=3)
            .set_color(BLACK)
            .set_opacity(1)
            .set_stroke(BLUE, width=4)
        )

        channels_vg_diagonal = VGroup(red_channel, green_channel, blue_channel).arrange(
            DOWN * 1.1 + RIGHT * 1.7, buff=-1.4
        )

        # output image
        output_image = SVGMobject("jpg_file.svg").set_stroke(
            WHITE, width=5, background=True
        )

        # big modules
        jpeg_encoder = Module(
            "JPEG Encoder",
            width=7,
            height=3,
            text_position=DOWN,
            text_weight=BOLD,
            text_scale=0.8,
        )
        jpeg_decoder = Module(
            "JPEG Decoder",
            width=7,
            height=3,
            text_position=UP,
            text_weight=BOLD,
            text_scale=0.8,
        )

        # color treatment
        color_treatment = Module(
            "Color treatment",
            REDUCIBLE_GREEN_DARKER,
            REDUCIBLE_GREEN_LIGHTER,
            height=jpeg_encoder.height,
            width=3,
            text_scale=0.5,
            text_position=UP,
            text_weight=BOLD,
        )

        ycbcr_m = Module(
            "YCbCr",
            fill_color=REDUCIBLE_YELLOW_DARKER,
            stroke_color=REDUCIBLE_YELLOW,
            height=1,
        ).scale_to_fit_width(color_treatment.width - 0.5)

        chroma_sub_m = Module(
            "Chroma Subsampling",
            fill_color=REDUCIBLE_YELLOW_DARKER,
            stroke_color=REDUCIBLE_YELLOW,
            text_scale=0.5,
            height=1,
        ).scale_to_fit_width(color_treatment.width - 0.5)

        color_modules = VGroup(ycbcr_m, chroma_sub_m).arrange(DOWN, buff=0.5)

        color_treatment_w_modules = VGroup(color_treatment, color_modules).arrange(
            ORIGIN
        )
        color_modules.shift(DOWN * 0.4)

        # small modules

        # encoding
        forward_dct_m = Module("Forward DCT", REDUCIBLE_YELLOW_DARKER, REDUCIBLE_YELLOW)
        forward_dct_icon = ImageMobject("dct.png").scale(0.2)
        forward_dct = Group(forward_dct_m, forward_dct_icon).arrange(DOWN, buff=0.5)

        quantizer_m = Module("Quantizer", REDUCIBLE_YELLOW_DARKER, REDUCIBLE_YELLOW)
        quantizer_icon = ImageMobject("quantization.png").scale(0.2)
        quantizer = Group(quantizer_m, quantizer_icon).arrange(DOWN, buff=0.5)

        lossless_comp_m = Module(
            "Encoder",
            REDUCIBLE_YELLOW_DARKER,
            REDUCIBLE_YELLOW,
        )
        lossless_icon = ImageMobject("lossless.png").scale(0.2)
        lossless_comp = Group(lossless_comp_m, lossless_icon).arrange(DOWN, buff=0.5)

        encoding_modules = (
            Group(forward_dct, quantizer, lossless_comp)
            .arrange(RIGHT, buff=0.7)
            .scale_to_fit_width(jpeg_encoder.width - 0.5)
        )
        jpeg_encoder_w_modules = Group(jpeg_encoder, encoding_modules).arrange(
            ORIGIN,
        )
        encoding_modules.shift(DOWN * 0.5)

        # decoding

        inverse_dct = Module("Inverse DCT", REDUCIBLE_YELLOW_DARKER, REDUCIBLE_YELLOW)
        dequantizer = Module("Dequantizer", REDUCIBLE_YELLOW_DARKER, REDUCIBLE_YELLOW)
        decoder = Module("Decoder", REDUCIBLE_YELLOW_DARKER, REDUCIBLE_YELLOW)

        decoding_modules = (
            VGroup(inverse_dct, dequantizer, decoder)
            .arrange(RIGHT, buff=0.5)
            .scale_to_fit_width(jpeg_decoder.width - 0.5)
        )
        jpeg_decoder_w_modules = VGroup(jpeg_decoder, decoding_modules).arrange(ORIGIN)
        decoding_modules.shift(DOWN * 0.5)

        # first row = encoding flow
        encoding_flow = Group(
            channels_vg_diagonal.scale(0.6),
            color_treatment_w_modules,
            jpeg_encoder_w_modules,
            output_image,
        ).arrange(RIGHT, buff=3)

        # second row = decoding flow
        decoding_flow = VGroup(
            output_image.copy(), jpeg_decoder_w_modules, channels_vg_diagonal.copy()
        ).arrange(RIGHT, buff=3)

        whole_map = Group(encoding_flow, decoding_flow).arrange(DOWN, buff=8)

        encode_arrows = VGroup()
        for i in range(len(encoding_flow.submobjects) - 1):
            encode_arrows.add(
                Arrow(
                    color=GRAY_B,
                    start=encoding_flow[i].get_right(),
                    end=encoding_flow[i + 1].get_left(),
                    stroke_width=3,
                    buff=0.3,
                    max_tip_length_to_length_ratio=0.08,
                    max_stroke_width_to_length_ratio=2,
                )
            )
        decode_arrows = VGroup()
        for i in range(len(decoding_flow.submobjects) - 1):
            decode_arrows.add(
                Arrow(
                    color=GRAY_B,
                    start=decoding_flow[i].get_right(),
                    end=decoding_flow[i + 1].get_left(),
                    stroke_width=3,
                    buff=0.3,
                    max_tip_length_to_length_ratio=0.08,
                    max_stroke_width_to_length_ratio=2,
                )
            )

        # whole map view state
        self.camera.frame.save_state()

        self.play(FadeIn(encoding_flow))
        self.play(FadeIn(encode_arrows))
        self.play(FadeIn(decoding_flow))
        self.play(FadeIn(decode_arrows))

        self.focus_on(encoding_flow, buff=1.3)

        self.wait(3)

        self.focus_on(channels_vg_diagonal)

        self.wait(3)

        self.focus_on(color_treatment_w_modules)

        self.wait(3)

        self.focus_on(encoding_flow, buff=1.3)

        self.wait(3)

        self.focus_on(jpeg_encoder_w_modules, buff=1.3)

        self.wait(3)

        self.focus_on(forward_dct)

        self.wait(3)

        self.focus_on(quantizer)

        self.wait(3)

        self.play(Restore(self.camera.frame), run_time=3)

    def focus_on(self, mobject, buff=2):
        self.play(
            self.camera.frame.animate.set_width(mobject.width * buff).move_to(mobject),
            run_time=3,
        )

class MotivateAndExplainYCbCr(ThreeDScene):
    def construct(self):
        self.set_camera_orientation(phi=75 * DEGREES, theta=-45 * DEGREES)
        self.move_camera(zoom=0.2)

        color_resolution = 8
        cubes_rgb = self.create_color_space_cube(
            coords2rgbcolor, color_res=color_resolution, cube_side_length=1
        )
        cubes_yuv = self.create_color_space_cube(
            coords2ycbcrcolor, color_res=color_resolution, cube_side_length=1
        )
        self.wait(2)
        self.add(
            cubes_rgb,
        )
        self.wait(2)

        anim_group = []
        # this loop removes every cube that is not in the grayscale diagonal of the RGB colorspace.
        # to do that, we calculate what coordinates a particular cube lives in, via their index.
        # any diagonal cube will have their coordinates matching, so we remove everything else.
        for index, cube in enumerate(cubes_rgb):
            coords = index2coords(index, base=color_resolution)
            print(coords)
            if not coords[0] == coords[1] == coords[2]:
                anim_group.append(FadeOut(cube))
                cubes_rgb.remove(cube)

        self.play(*anim_group)

        self.play(Rotate(cubes_rgb, angle=PI * 2))
        cubes_arranged = cubes_rgb.copy().arrange(OUT, buff=0)
        self.play(Transform(cubes_rgb, cubes_arranged))

        self.wait()

        cubes_yuv.move_to(cubes_arranged.get_center())

        # this is a very bad way of transforming the grayscale line to
        # the cube obviously but it illustrates the point at least for now
        self.play(Transform(cubes_arranged, cubes_yuv))
        self.play(Rotate(cubes_arranged, angle=PI * 2))
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

        cubes_rgb = Group(*cubes)

        return cubes_rgb


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
        self.play(FadeIn(image_mob))
        self.wait()

        print("Image size:", image_mob.get_pixel_array().shape)

        self.perform_JPEG(image_mob)

    def perform_JPEG(self, image_mob):
        # Performs encoding/decoding steps on a gray scale block
        block_image, pixel_grid, block = self.highlight_pixel_block(image_mob, 125, 125)
        print("Before\n", block[:, :, 1])
        block_centered = format_block(block)
        print("After centering\n", block_centered)

        dct_block = dct_2d(block_centered)
        np.set_printoptions(suppress=True)
        print("DCT block (rounded)\n", np.round(dct_block, decimals=1))

        heat_map = self.get_heat_map(block_image, dct_block)
        heat_map.move_to(block_image.get_center() + RIGHT * 6)
        self.play(FadeIn(heat_map))
        self.wait()

        self.play(FadeOut(heat_map))
        self.wait()

        quantized_block = quantize(dct_block)
        print("After quantization\n", quantized_block)

        dequantized_block = dequantize(quantized_block)
        print("After dequantize\n", dequantized_block)

        invert_dct_block = idct_2d(dequantized_block)
        print("Invert DCT block\n", invert_dct_block)

        compressed_block = invert_format_block(invert_dct_block)
        print("After reformat\n", compressed_block)

        print("MSE\n", np.mean((compressed_block - block[:, :, 1]) ** 2))

        final_image = self.get_image_mob(compressed_block, height=2)
        final_image.move_to(block_image.get_center() + RIGHT * 6)

        final_image_grid = self.get_pixel_grid(final_image, 8).move_to(
            final_image.get_center()
        )
        self.play(FadeIn(final_image), FadeIn(final_image_grid))
        self.wait()

        # self.get_dct_component(0, 0)

    def highlight_pixel_block(self, image_mob, start_row, start_col, block_size=8):
        pixel_array = image_mob.get_pixel_array()
        block = pixel_array[
            start_row : start_row + block_size, start_col : start_col + block_size
        ]
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

        self.play(Create(tiny_square_highlight))
        self.wait()

        block_position = DOWN * 2
        block_image = self.get_image_mob(block, height=2).move_to(block_position)
        pixel_grid = self.get_pixel_grid(block_image, block_size).move_to(
            block_position
        )
        surround_rect = SurroundingRectangle(pixel_grid, buff=0).set_color(
            REDUCIBLE_YELLOW
        )
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
            square.set_fill(
                color=interpolate_color(min_color, max_color, alpha), opacity=1
            )

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
        print("Before\n", block[:, :, 1])
        block_image.move_to(LEFT * 2 + DOWN * 2)
        pixel_grid.move_to(block_image.get_center())
        block_centered = format_block(block)
        print("After centering\n", block_centered)

        dct_block = dct_2d(block_centered)
        np.set_printoptions(suppress=True)
        print("DCT block (rounded)\n", np.round(dct_block, decimals=1))

        self.play(FadeIn(image_mob))
        self.wait()

        self.play(
            FadeIn(block_image),
            FadeIn(pixel_grid),
        )
        self.wait()
        num_components = 40
        partial_block = self.get_partial_block(dct_block, num_components)
        print(f"Partial block - {num_components} components\n", partial_block)

        partial_block_image = self.get_image_mob(partial_block, height=2)
        partial_pixel_grid = self.get_pixel_grid(
            partial_block_image, partial_block.shape[0]
        )

        partial_block_image.move_to(RIGHT * 2 + DOWN * 2)
        partial_pixel_grid.move_to(partial_block_image.get_center())

        self.play(
            FadeIn(partial_block_image),
            FadeIn(partial_pixel_grid),
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
            print("Bad array\n", pixel_array)
            raise ValueError("All elements in pixel_array must be in range [0, 255]")

        image_mob = self.get_image_mob(pixel_array, height=2)
        pixel_grid = self.get_pixel_grid(image_mob, pixel_array.shape[0])
        self.wait()
        self.play(
            FadeIn(image_mob),
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
        block = pixel_array[
            start_row : start_row + block_size, start_col : start_col + block_size
        ]

        block_image = self.get_image_mob(block, height=2)
        pixel_grid = self.get_pixel_grid(block_image, block_size)

        return block_image, pixel_grid, block

    def display_component(self, dct_matrix, row, col):
        pass


class DCTSliderExperiments(DCTComponents):
    def construct(self):
        image_mob = ImageMobject("dog").move_to(UP * 2)
        block_image, pixel_grid, block = self.get_pixel_block(image_mob, 125, 125)
        print("Before\n", block[:, :, 1])
        block_image.move_to(LEFT * 2 + DOWN * 0.5)
        pixel_grid.move_to(block_image.get_center())
        block_centered = format_block(block)
        print("After centering\n", block_centered)

        dct_block = dct_2d(block_centered)
        np.set_printoptions(suppress=True)
        print("DCT block (rounded)\n", np.round(dct_block, decimals=1))

        self.play(
            FadeIn(image_mob),
        )
        self.wait()

        self.play(
            FadeIn(block_image),
            FadeIn(pixel_grid),
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
            lambda m: m.next_to(number_line.n2p(tracker.get_value()), DOWN)
        )
        self.play(
            FadeIn(tick),
        )
        self.wait()
        image_pos = RIGHT * 2 + DOWN * 0.5

        def get_new_block():
            new_partial_block = self.get_partial_block(dct_block, tracker.get_value())
            print(f"Partial block - {tracker.get_value()} components")
            print(
                "MSE", np.mean((new_partial_block - original_block[:, :, 1]) ** 2), "\n"
            )

            new_partial_block_image = self.get_image_mob(new_partial_block, height=2)
            new_partial_block_image.move_to(image_pos)
            return new_partial_block_image

        partial_block = self.get_partial_block(dct_block, tracker.get_value())
        partial_block_image = always_redraw(get_new_block)
        partial_pixel_grid = self.get_pixel_grid(
            partial_block_image, partial_block.shape[0]
        )
        partial_pixel_grid.move_to(image_pos)
        self.play(
            FadeIn(partial_block_image),
            FadeIn(partial_pixel_grid),
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
        image_mob = ImageMobject("dog").move_to(UP * 1)
        

        number_line = self.initialize_slider()
        dct_components = Tex("DCT Components").next_to(number_line, UP)

        self.play(
            FadeIn(image_mob),
            FadeIn(number_line),
            Write(dct_components)
        )
        # print(new_image.get_pixel_array().shape)
        self.wait()

        self.animate_slider(image_mob, number_line)


    def get_all_blocks(
        self,
        image_mob,
        start_row,
        end_row,
        start_col,
        end_col,
        num_components,
        block_size=8,
    ):
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
                new_pixel_array[i : i + block_size, j : j + block_size] = partial_block

        return new_pixel_array

    def get_pixel_block(self, pixel_array, start_row, start_col, block_size=8):
        return pixel_array[
            start_row : start_row + block_size, start_col : start_col + block_size
        ]

    def initialize_slider(self):
        number_line = NumberLine(
            x_range=[0, 64, 4],
            length=10,
            color=REDUCIBLE_VIOLET,
            include_numbers=True,
            label_direction=UP,
        )
        number_line.move_to(DOWN * 2.5)

        return number_line

    def animate_slider(self, image_mob, number_line):
        original_pixel_array = image_mob.get_pixel_array()[2:298, 3:331, 0]
        component_tracker = ValueTracker(0)
        
        def get_new_image():
            new_pixel_array = self.get_all_blocks(image_mob, 2, 298, 3, 331, component_tracker.get_value())
            relevant_section = new_pixel_array[2:298, 3:331]
            new_image = self.get_image_mob(new_pixel_array, height=None).move_to(UP * 1 + RIGHT * 2)
            return new_image

        tick = Triangle().scale(0.2).set_color(REDUCIBLE_YELLOW)
        tick.set_fill(color=REDUCIBLE_YELLOW, opacity=1)

        tick.add_updater(
            lambda m: m.next_to(number_line.n2p(component_tracker.get_value()), DOWN)
        )

        new_image = always_redraw(get_new_image)
        self.play(
            FadeIn(tick),
            image_mob.animate.shift(LEFT * 2),
            FadeIn(new_image),
        )
        self.wait()
    

        self.play(
            component_tracker.animate.set_value(64),
            run_time=32,
            rate_func=linear,
        ),

        # self.wait()

class DCT1DExperiments(DCTComponents):
    def construct(self):
        image_mob = ImageMobject("dog").move_to(UP * 2)
        block_image, pixel_grid, block = self.get_pixel_block(image_mob, 125, 125)
        block_image.shift(UP * 2)
        self.play(
            FadeIn(block_image),
        )
        self.wait()
        row = 7
        print(block[:, :, 0])
        print(f"Block row: {row}\n", block[:, :, 0][row])
        pixel_row_mob, row_values = self.get_pixel_row_mob(block, row)
        print("Selected row values\n", row_values)
        pixel_row_mob.next_to(block_image, DOWN)
        self.play(
            FadeIn(pixel_row_mob),
        )
        self.wait()

        row_values_centered = format_block(row_values)
        print("After centering\n", row_values_centered)

        dct_row_pixels = dct_1d(row_values_centered)
        np.set_printoptions(suppress=True)
        print("DCT block (rounded)\n", np.round(dct_row_pixels, decimals=1))

        inverted_row = idct_1d(dct_row_pixels) + 128
        self.play(
            FadeOut(block_image),
            FadeOut(pixel_row_mob),
        )
        self.wait()
        print("Inverted row:\n", inverted_row)
        num_pixels = dct_row_pixels.shape[0]
        height_pixel = 0.6
        for col in range(num_pixels):
            text = Tex(str(col)).move_to(UP * 2)
            new_value = 250
            basis_component, dct_row = self.get_dct_component(col, new_value=new_value)
            print(f"Basis value for {col}\n", basis_component)

            component_mob = self.make_row_of_pixels(
                basis_component, height=height_pixel
            ).shift(RIGHT * 0.25)
            ax, graph = self.get_graph_dct_component(col)

            self.add(text, component_mob, ax, graph)
            self.wait(2)

            self.remove(text, component_mob, ax, graph)

        self.wait()

        self.play(
            FadeIn(block_image),
            FadeIn(pixel_row_mob),
        )
        self.wait()

        self.draw_image_graph(dct_row_pixels)

    def get_pixel_row_mob(self, pixel_array, row, height=SMALL_BUFF * 5, num_pixels=8):
        row_length = height * num_pixels
        row_values = [pixel_array[row][i][0] for i in range(num_pixels)]
        pixel_row_mob = VGroup(
            *[
                Rectangle(height=height, width=row_length / num_pixels)
                .set_stroke(color=REDUCIBLE_GREEN_LIGHTER)
                .set_fill(color=gray_scale_value_to_hex(value), opacity=1)
                for value in row_values
            ]
        ).arrange(RIGHT, buff=0)
        return pixel_row_mob, np.array(row_values)

    def make_row_of_pixels(self, row_values, height=SMALL_BUFF * 5, num_pixels=8):
        row_length = height * num_pixels
        adjusted_row_values = []
        for val in row_values:
            adjusted_row_values.append(int(round(val)))
        pixel_row_mob = VGroup(
            *[
                Rectangle(height=height, width=row_length / num_pixels)
                .set_stroke(color=REDUCIBLE_GREEN_LIGHTER)
                .set_fill(color=gray_scale_value_to_hex(value), opacity=1)
                for value in adjusted_row_values
            ]
        ).arrange(RIGHT, buff=0)
        return pixel_row_mob

    def get_dct_component(self, col, new_value=250, num_pixels=8):
        dct_row = np.zeros((num_pixels,))
        dct_row[col] = new_value
        return idct_1d(dct_row) + 128, dct_row

    def get_graph_dct_component(self, col, num_pixels=8):
        ax = Axes(
            x_range=[0, num_pixels - 1, 1],
            y_range=[-1, 1],
            y_length=2,
            x_length=4.2,
            tips=False,
            axis_config={"include_numbers": True, "include_ticks": True},
            x_axis_config={
                "numbers_to_exclude": list(range(1, num_pixels - 1)),
            },
            # y_axis_config={"numbers_to_exclude": list(range(1, 255))},
        ).move_to(DOWN * 2)
        func = lambda n: np.cos((2 * n + 1) * col * np.pi / (2 * num_pixels))
        if col == 0:
            func = (
                lambda n: 1
                / np.sqrt(2)
                * np.cos((2 * n + 1) * col * np.pi / (2 * num_pixels))
            )
        graph = ax.plot(func)
        graph.set_color(REDUCIBLE_YELLOW)
        return ax, graph

    def draw_image_graph(self, dct_row):
        """
        @param: 1D DCT of a row of pixels
        @return: graph composed of a linear combination of cosine functions weighted by dct_row
        """

        def get_basis_function(col):
            factor = dct_row[col] * np.sqrt(2 / dct_row.shape[0])

            def f(n):
                if col == 0:
                    return factor * 1 / np.sqrt(2)
                return factor * np.cos(
                    (2 * n + 1) * col * np.pi / (2 * dct_row.shape[0])
                )

            return f

        basis_functions = [get_basis_function(i) for i in range(dct_row.shape[0])]
        final_func = (
            lambda n: sum(basis_function(n) for basis_function in basis_functions) + 128
        )
        ax = Axes(
            x_range=[0, dct_row.shape[0] - 1, 1],
            y_range=[0, 255, 1],
            y_length=2,
            x_length=4.375,
            tips=False,
            axis_config={"include_numbers": True, "include_ticks": False},
            x_axis_config={
                "numbers_to_exclude": list(range(1, dct_row.shape[0] - 1)),
            },
            y_axis_config={"numbers_to_exclude": list(range(1, 255))},
        )
        graph = ax.plot(final_func)
        graph.set_color(REDUCIBLE_YELLOW)
        ax.add_coordinates()
        row_values = idct_1d(dct_row) + 128
        pixel_coordinates = list(enumerate(row_values))
        dots = VGroup(
            *[
                Dot()
                .scale(0.7)
                .move_to(ax.coords_to_point(x, y))
                .set_color(REDUCIBLE_YELLOW)
                for x, y in pixel_coordinates
            ]
        )

        return ax, graph, dots


class DCT1DStepsVisualized(DCT1DExperiments):
    """
    Animations we need:
    1. Take a 8 x 8 block and highlight a row of pixels
    2. Build an array of pixels from a given set of row values
    3. Given a row of pixels, draw the exact signal it represents in terms of cosine waves
    4. Show shift of pixels and signal down by 128 to center around 0
    5. Build up original signal using DCT components, show components summing together one by one
    """

    GRAPH_ADJUST = LEFT * 0.35

    def construct(self):
        image_mob = ImageMobject("dog").move_to(UP * 2)
        block_image, pixel_grid, block = self.get_pixel_block(image_mob, 125, 125)
        block_image.shift(UP * 2.5)
        self.play(FadeIn(block_image))
        self.wait()
        row = 7
        print(block[:, :, 0])
        print(f"Block row: {row}\n", block[:, :, 0][row])
        pixel_row_mob, row_values, highlight = self.highlight_pixel_row(
            block, block_image, row, height=0.625
        )
        print("Selected row values\n", row_values)
        pixel_row_mob.move_to(UP)
        self.play(
            FadeIn(highlight),
        )
        self.wait()

        self.show_highlight(pixel_row_mob, highlight)

        self.play(
            FadeOut(block_image),
        )
        self.wait()

        array_mob = self.get_array_obj(row_values)
        array_mob.next_to(pixel_row_mob, UP)
        self.play(
            FadeIn(array_mob),
        )
        self.wait()

        row_values_centered = format_block(row_values)
        print("After centering\n", row_values_centered)

        dct_row_pixels = dct_1d(row_values_centered)

        ax, graph, dots = self.draw_image_graph(dct_row_pixels)

        self.show_graph(ax, graph, dots, pixel_row_mob)

        graph_components = VGroup(ax, graph, dots)

        specific_step, rect, center_step = self.show_centering_step(
            row_values, dct_row_pixels, array_mob, pixel_row_mob, graph_components
        )

        label, apply_dct = self.prepare_to_shift(specific_step, center_step, array_mob)

        left_group = self.shift_components(
            pixel_row_mob, array_mob, graph_components, label, rect
        )

        dct_labels, dct_array_mob, dct_graph_components = self.show_dct_step(
            label, array_mob, graph_components, dct_row_pixels
        )

        self.show_how_dct_works(
            apply_dct,
            left_group,
            dct_labels,
            dct_array_mob,
            dct_graph_components,
            dct_row_pixels,
        )

    def show_dct_step(self, label, array_mob, graph_components, dct_row):
        x_label, brace = label
        new_label_x_hat = MathTex(r"\hat{X} = \text{DCT}(X)").scale(0.8)
        new_array_mob = self.get_array_obj(dct_row, color=REDUCIBLE_PURPLE)
        new_brace = Brace(new_array_mob, direction=UP)
        new_label_x_hat.next_to(new_brace, UP, buff=SMALL_BUFF)

        new_label = VGroup(new_label_x_hat, new_brace)

        dct_ax = self.get_dct_axis(dct_row, -80, 80)

        dct_graph, dct_points = self.plot_row_values(
            dct_ax, dct_row, color=REDUCIBLE_PURPLE
        )

        dct_graph_components = VGroup(dct_ax, dct_graph, dct_points)
        dct_graph_components.next_to(new_array_mob, DOWN).shift(
            DCT1DStepsVisualized.GRAPH_ADJUST
        )
        group = VGroup(new_label, new_array_mob, dct_graph_components)

        group.move_to(RIGHT * 3.5)

        self.play(
            TransformFromCopy(label, new_label),
            TransformFromCopy(array_mob, new_array_mob),
            run_time=2,
        )
        self.wait()

        self.play(Write(dct_ax))
        self.play(
            *[GrowFromCenter(dot) for dot in dct_points],
            Create(dct_graph),
        )
        self.wait()

        return new_label, new_array_mob, dct_graph_components

    def show_how_dct_works(
        self,
        label,
        left_group,
        dct_labels,
        dct_array_mob,
        dct_graph_components,
        dct_row,
    ):
        dct_group = VGroup(dct_labels, dct_array_mob)
        self.play(
            FadeOut(left_group),
            FadeOut(dct_graph_components),
            FadeOut(label),
            dct_group.animate.move_to(UP * 3),
        )
        self.wait()

        key_sum = MathTex(r"X = \sum_{k=0}^{7} X[k] \cdot C_k").scale(0.8)
        key_sum.next_to(dct_array_mob, DOWN)
        self.play(
            Write(key_sum),
        )
        self.wait()

        self.remind_individual_cosine_comp(dct_array_mob)

        self.build_up_signal(dct_array_mob, dct_row)

    def build_up_signal(self, dct_array_mob, dct_row):
        ALIGNMENT_SHIFT = LEFT * 0.4
        right_label = (
            MathTex(r"\vec{0}").move_to(RIGHT * 3.5).set_color(REDUCIBLE_YELLOW)
        )
        zero_dct = np.zeros(dct_row.shape[0])
        right_component_mob = self.make_row_of_pixels(zero_dct + 128, height=0.625)
        right_component_mob.next_to(right_label, DOWN)

        right_ax, right_graph, right_dots = self.draw_image_graph(
            zero_dct, centered=True
        )
        right_ax, right_graph, right_dots = self.show_graph(
            right_ax,
            right_graph,
            right_dots,
            right_component_mob,
            animate=False,
            alignment_shift=ALIGNMENT_SHIFT,
        )

        self.play(
            FadeIn(right_label),
            FadeIn(right_component_mob),
            FadeIn(right_ax),
            *[GrowFromCenter(dot) for dot in right_dots],
            Create(right_graph),
        )
        self.wait()

        left_text = MathTex("C_0").set_color(REDUCIBLE_YELLOW)
        left_text.move_to(LEFT * 3.5)
        basis_component, left_dct_row = self.get_dct_component(0)
        left_component_mob = self.make_row_of_pixels(
            basis_component, height=SMALL_BUFF * 6.25
        )
        left_component_mob.next_to(left_text, DOWN)
        left_ax, left_graph = self.get_graph_dct_component(0)
        VGroup(left_ax, left_graph).next_to(left_component_mob, DOWN).shift(
            ALIGNMENT_SHIFT
        )
        self.play(
            Write(left_text),
            FadeIn(left_component_mob),
            FadeIn(left_ax),
            Create(left_graph),
        )
        self.wait()

        left_graph_components = VGroup(left_ax, left_graph)
        right_graph_components = VGroup(right_ax, right_graph, right_dots)

        sub_array = dct_array_mob[0][0]
        current_highlight = Rectangle(height=sub_array.height, width=sub_array.width)
        current_highlight.move_to(sub_array.get_center()).set_color(REDUCIBLE_YELLOW)

        self.play(
            Create(current_highlight),
        )
        self.wait()

        for step in range(dct_row.shape[0]):
            self.perform_update_step(
                left_graph_components,
                right_graph_components,
                left_component_mob,
                right_component_mob,
                left_text,
                right_label,
                step,
                dct_row,
                dct_array_mob,
                current_highlight,
                alignment_shift=ALIGNMENT_SHIFT,
            )

    def perform_update_step(
        self,
        left_graph_components,
        right_graph_components,
        left_component_mob,
        right_component_mob,
        left_text,
        right_label,
        step,
        dct_row,
        dct_array_mob,
        current_highlight,
        alignment_shift=LEFT * 0.4,
    ):
        sub_array = dct_array_mob[0][: step + 1]
        highlight = Rectangle(height=sub_array.height, width=sub_array.width)
        highlight.move_to(sub_array.get_center()).set_color(REDUCIBLE_YELLOW)
        self.play(Transform(current_highlight, highlight))
        self.wait()

        left_ax, left_graph = left_graph_components
        right_ax, right_graph, right_dots = right_graph_components
        isolated_dct_row = np.zeros(dct_row.shape[0])
        isolated_dct_row[step] = dct_row[step]

        iso_right_ax, iso_graph, iso_dots = self.draw_image_graph(
            isolated_dct_row, centered=True, color=REDUCIBLE_VIOLET
        )
        iso_right_ax, iso_graph, iso_dots = self.show_graph(
            iso_right_ax,
            iso_graph,
            iso_dots,
            right_component_mob,
            animate=False,
            alignment_shift=alignment_shift,
        )
        self.align_graph_and_dots(right_dots, iso_dots, iso_graph)

        self.play(
            TransformFromCopy(left_graph, iso_graph),
        )
        intermediate_text = self.generate_intermediate_text(right_component_mob)
        self.play(
            *[GrowFromCenter(dot) for dot in iso_dots],
            Transform(right_label, intermediate_text[step]),
        )
        self.wait()

        cumulative_dct_row = self.get_partial_row_dct(dct_row, step + 1)
        cum_right_ax, cum_graph, cum_dots = self.draw_image_graph(
            cumulative_dct_row, centered=True, color=REDUCIBLE_YELLOW
        )
        cum_right_ax, cum_graph, cum_dots = self.show_graph(
            cum_right_ax,
            cum_graph,
            cum_dots,
            right_component_mob,
            animate=False,
            alignment_shift=alignment_shift,
        )

        final_text = self.generate_final_text(right_component_mob)

        self.align_graph_and_dots(right_dots, cum_dots, cum_graph)

        new_right_component_mob = self.make_row_of_pixels(
            idct_1d(cumulative_dct_row) + 128, height=0.625
        )
        new_right_component_mob.move_to(right_component_mob.get_center())

        self.play(
            Transform(right_graph, cum_graph),
            Transform(right_dots, cum_dots),
            FadeOut(iso_graph),
            FadeOut(iso_dots),
            Transform(right_component_mob, new_right_component_mob),
            Transform(right_label, final_text[step]),
        )
        self.wait()

        if step + 1 == dct_row.shape[0]:
            return

        new_left_text = MathTex(f"C_{step + 1}").set_color(REDUCIBLE_YELLOW)
        new_left_text.move_to(left_text.get_center())
        new_basis_component, new_left_dct_row = self.get_dct_component(step + 1)
        new_left_component_mob = self.make_row_of_pixels(
            new_basis_component, height=SMALL_BUFF * 6.25
        )
        new_left_component_mob.next_to(new_left_text, DOWN)
        new_left_ax, new_left_graph = self.get_graph_dct_component(step + 1)
        VGroup(new_left_ax, new_left_graph).next_to(new_left_component_mob, DOWN).shift(
            alignment_shift
        )
        self.play(
            Transform(left_text, new_left_text),
            Transform(left_component_mob, new_left_component_mob),
            Transform(left_graph, new_left_graph),
        )
        self.wait()

    def align_graph_and_dots(self, original_dots, new_dots, new_graph):
        horiz_diff_adjust = (
            original_dots[0].get_center()[0] - new_dots[0].get_center()[0]
        )
        new_graph.shift(RIGHT * horiz_diff_adjust)
        new_dots.shift(RIGHT * horiz_diff_adjust)

    def get_partial_row_dct(self, dct_row, num_components):
        new_dct_row = np.zeros(dct_row.shape[0])
        for index in range(num_components):
            new_dct_row[index] = dct_row[index]
        return new_dct_row

    def generate_intermediate_text(self, right_component_mob):
        zero = MathTex(r"\vec{0}", "+", r"X[0] \cdot C_0")
        zero[0].set_color(REDUCIBLE_YELLOW)
        zero[-1].set_color(REDUCIBLE_VIOLET)

        one = MathTex(r"X[0] \cdot C_0", "+", r"X[1] \cdot C_1")
        one[0].set_color(REDUCIBLE_YELLOW)
        one[-1].set_color(REDUCIBLE_VIOLET)

        other_representations = [
            self.get_intermediate_representation(k) for k in range(2, 8)
        ]

        all_reprs = [zero, one] + other_representations
        for represent in all_reprs:
            represent.scale(0.8).next_to(right_component_mob, UP)
        return all_reprs

    def get_intermediate_representation(self, i):
        represent = MathTex(
            r"\sum_{k=0}^{" + str(i - 1) + r"} X[k] \cdot C_k",
            "+",
            r"X[{0}] \cdot C_{0}".format(i),
        )
        represent[0].set_color(REDUCIBLE_YELLOW)
        represent[-1].set_color(REDUCIBLE_VIOLET)
        return represent

    def generate_final_text(self, right_component_mob):
        final_zero = MathTex(r"X[0] \cdot C_0").set_color(REDUCIBLE_YELLOW)
        final_six = [
            MathTex(r"\sum_{k=0}^{" + str(i) + r"} X[k] \cdot C_k").set_color(
                REDUCIBLE_YELLOW
            )
            for i in range(1, 8)
        ]

        all_reprs = [final_zero] + final_six
        for represent in all_reprs:
            represent.scale(0.8).next_to(right_component_mob, UP)
        return all_reprs

    def remind_individual_cosine_comp(self, dct_array_mob):
        ALIGNMENT_SHIFT = RIGHT * 0.25
        dct_array, dct_array_text = dct_array_mob
        highlight_box = None
        basis_component, dct_row = self.get_dct_component(0)
        component_mob = self.make_row_of_pixels(
            basis_component, height=SMALL_BUFF * 6
        ).shift(ALIGNMENT_SHIFT)
        ax, graph = self.get_graph_dct_component(0)
        text = MathTex("C_0").set_color(REDUCIBLE_YELLOW)
        text.next_to(ax, DOWN).shift(ALIGNMENT_SHIFT)
        for col, elem in enumerate(dct_array):
            new_text = MathTex(f"C_{col}").set_color(REDUCIBLE_YELLOW)
            animations = []
            basis_component, dct_row = self.get_dct_component(col)
            new_component_mob = self.make_row_of_pixels(
                basis_component, height=SMALL_BUFF * 6
            ).shift(ALIGNMENT_SHIFT)
            new_ax, new_graph = self.get_graph_dct_component(col)
            new_text.next_to(new_ax, DOWN).shift(ALIGNMENT_SHIFT)
            if not highlight_box:
                highlight_box = elem.copy().set_color(REDUCIBLE_YELLOW)
                animations.append(Create(highlight_box))
                animations.append(Write(text))
                animations.extend([FadeIn(component_mob), FadeIn(ax), FadeIn(graph)])
            else:
                animations.append(highlight_box.animate.move_to(elem.get_center()))
                animations.append(Transform(text, new_text))
                animations.extend(
                    [
                        Transform(component_mob, new_component_mob),
                        Transform(ax, new_ax),
                        Transform(graph, new_graph),
                    ],
                )

            self.play(
                *animations,
            )
            self.wait()

        self.play(
            FadeOut(highlight_box),
            FadeOut(component_mob),
            FadeOut(ax),
            FadeOut(graph),
            FadeOut(text),
        )
        self.wait()

    def show_centering_step(
        self, row_values, dct_row, array_mob, pixel_row_mob, graph_components
    ):
        ax, graph, dots = graph_components
        entire_group = VGroup(array_mob, pixel_row_mob, graph_components)
        rect = Rectangle(height=entire_group.height + 2, width=entire_group.width + 1)
        rect.set_color(REDUCIBLE_VIOLET)
        self.play(
            Create(rect),
        )
        self.wait()

        center_step = Tex("Center pixel values around 0").next_to(rect, UP)

        self.play(
            Write(center_step),
        )
        self.wait()

        specific_step = MathTex(r"[0, 255] \rightarrow [-128, 127]").scale(0.8)
        specific_step.next_to(center_step, DOWN * 2)
        self.play(
            Write(specific_step),
        )
        self.wait()

        new_values = format_block(row_values)
        array, array_values = array_mob
        self.play(
            *[
                value.animate.set_value(new_values[i]).move_to(array[i].get_center())
                for i, value in enumerate(array_values)
            ],
        )
        self.wait()

        new_ax, new_graph, new_dots = self.draw_image_graph(dct_row, centered=True)
        new_graph_components = VGroup(new_ax, new_graph, new_dots)
        new_graph_components.move_to(graph_components.get_center())
        self.play(
            Transform(ax, new_ax),
            Transform(graph, new_graph),
            Transform(dots, new_dots),
        )
        self.wait()

        return specific_step, rect, center_step

    def prepare_to_shift(self, specific_step, center_step, array_mob):
        apply_dct = Tex("Apply DCT").move_to(center_step.get_center())
        self.play(ReplacementTransform(center_step, apply_dct))
        self.wait()

        self.play(
            FadeOut(specific_step),
        )

        x_label = MathTex("X").scale(0.8)
        brace_up = Brace(array_mob, direction=UP)
        x_label.next_to(brace_up, UP, buff=SMALL_BUFF)

        self.play(
            Write(x_label),
            GrowFromCenter(brace_up),
        )
        self.wait()

        return VGroup(x_label, brace_up), apply_dct

    def show_highlight(self, pixel_row_mob, highlight):
        new_highlight = SurroundingRectangle(pixel_row_mob, buff=0).set_color(
            REDUCIBLE_GREEN_LIGHTER
        )
        self.play(
            LaggedStart(
                TransformFromCopy(highlight, new_highlight),
                FadeIn(pixel_row_mob),
                lag_ratio=0.4,
            )
        )
        self.wait()
        self.remove(new_highlight)
        self.play(
            FadeOut(highlight),
        )
        self.wait()

    def show_graph(
        self, ax, graph, dots, mob_above, animate=True, alignment_shift=None
    ):
        if alignment_shift is None:
            alignment_shift = DCT1DStepsVisualized.GRAPH_ADJUST
        graph_components = VGroup(ax, graph, dots).next_to(mob_above, DOWN)
        graph_components.shift(alignment_shift)
        if animate:
            self.play(
                Write(ax),
            )
            self.wait()
            self.play(
                *[GrowFromCenter(dot) for dot in dots],
            )
            self.wait()

            self.play(
                Create(graph),
            )
            self.wait()

        return ax, graph, dots

    def make_component(self, text, color=REDUCIBLE_VIOLET, scale=0.8):
        # geometry is first index, TextMob is second index
        text_mob = Tex(text).scale(scale)
        rect = Rectangle(color=color, height=1.1, width=3)
        return VGroup(rect, text_mob)

    def shift_components(self, pixel_row_mob, array_mob, graph, label, surround_rect):
        scale = 1
        new_position = LEFT * 3.5
        group = VGroup(pixel_row_mob, array_mob, graph, label, surround_rect)
        self.play(
            group.animate.scale(scale).move_to(new_position),
        )
        self.wait()
        return group

    def highlight_pixel_row(
        self, pixel_array, block_image_mob, row, height=SMALL_BUFF * 5, num_pixels=8
    ):
        row_length = height * num_pixels
        block_row_height = block_image_mob.height / num_pixels
        row_values = [pixel_array[row][i][0] for i in range(num_pixels)]
        highlight = Rectangle(height=block_row_height, width=block_image_mob.width)
        highlight_pos = (
            block_image_mob.get_top()
            + row * DOWN * block_row_height
            + DOWN * block_row_height / 2
        )
        highlight.move_to(highlight_pos).set_color(REDUCIBLE_GREEN_LIGHTER)
        pixel_row_mob = VGroup(
            *[
                Rectangle(height=height, width=row_length / num_pixels)
                .set_stroke(color=REDUCIBLE_GREEN_LIGHTER)
                .set_fill(color=gray_scale_value_to_hex(value), opacity=1)
                for value in row_values
            ]
        ).arrange(RIGHT, buff=0)
        return pixel_row_mob, np.array(row_values), highlight

    def get_array_obj(self, values, length=5, height=0.5, color=REDUCIBLE_GREEN_LIGHTER):
        array = VGroup(
            *[Rectangle(height=height, width=length / len(values)) for _ in values]
        ).arrange(RIGHT, buff=0)
        array.set_color(color)
        array_text = VGroup(
            *[
                Integer(val).scale(0.6).move_to(array[i].get_center())
                for i, val in enumerate(values)
            ]
        )
        return VGroup(array, array_text)

    def draw_image_graph(self, dct_row, centered=False, color=REDUCIBLE_YELLOW):
        """
        @param: 1D DCT of a row of pixels
        @param: centered, if true, then plot range is from [-128, 127]
        @return: graph composed of a linear combination of cosine functions weighted by dct_row
        """

        def get_basis_function(col):
            factor = dct_row[col] * np.sqrt(2 / dct_row.shape[0])

            def f(n):
                if col == 0:
                    return factor * 1 / np.sqrt(2)
                return factor * np.cos(
                    (2 * n + 1) * col * np.pi / (2 * dct_row.shape[0])
                )

            return f

        basis_functions = [get_basis_function(i) for i in range(dct_row.shape[0])]
        final_func = lambda n: sum(
            basis_function(n) for basis_function in basis_functions
        )
        if not centered:
            final_func = (
                lambda n: sum(basis_function(n) for basis_function in basis_functions)
                + 128
            )
        ax = self.get_axis(dct_row, centered=centered)
        graph = ax.plot(final_func)
        graph.set_color(color)
        ax.add_coordinates()
        row_values = idct_1d(dct_row)
        if not centered:
            row_values = row_values + 128
        pixel_coordinates = list(enumerate(row_values))
        dots = VGroup(
            *[
                Dot().scale(0.7).move_to(ax.coords_to_point(x, y)).set_color(color)
                for x, y in pixel_coordinates
            ]
        )

        return ax, graph, dots

    def get_axis(self, dct_row, centered=False):
        if not centered:
            return Axes(
                x_range=[0, dct_row.shape[0] - 1, 1],
                y_range=[0, 255, 1],
                y_length=2,
                x_length=4.375,
                tips=False,
                axis_config={"include_numbers": True, "include_ticks": False},
                x_axis_config={
                    "numbers_to_exclude": list(range(1, dct_row.shape[0] - 1))
                },
                y_axis_config={"numbers_to_exclude": list(range(1, 255))},
            )
        return Axes(
            x_range=[0, dct_row.shape[0] - 1, 1],
            y_range=[-128, 127, 1],
            y_length=2,
            x_length=4.375,
            tips=False,
            axis_config={"include_numbers": True, "include_ticks": False},
            x_axis_config={
                "numbers_to_exclude": list(range(1, dct_row.shape[0] - 1)),
                "label_direction": UP,
            },
            y_axis_config={"numbers_to_exclude": list(range(-127, 127))},
        )

    def get_dct_axis(self, dct_row, min_y, max_y):
        ax = Axes(
            x_range=[0, dct_row.shape[0] - 1, 1],
            y_range=[min_y, max_y, 1],
            y_length=3,
            x_length=4.375,
            tips=False,
            axis_config={"include_numbers": True, "include_ticks": False},
            x_axis_config={"numbers_to_exclude": list(range(1, dct_row.shape[0]))},
            y_axis_config={"numbers_to_exclude": list(range(min_y + 1, max_y))},
        )
        return ax

    def plot_row_values(self, axes, pixel_row_values, color=REDUCIBLE_YELLOW):
        pixel_coordinates = list(enumerate(pixel_row_values))
        axes.add_coordinates()
        path = VGroup()
        path_points = [axes.coords_to_point(x, y) for x, y in pixel_coordinates]
        path.set_points_smoothly(*[path_points]).set_color(color)
        dots = VGroup(
            *[
                Dot(axes.coords_to_point(x, y), radius=SMALL_BUFF / 2, color=color)
                for x, y in pixel_coordinates
            ]
        )
        return path, dots

class MotivateDCT(DCT1DStepsVisualized):
    def construct(self):
        all_equations = self.show_equations()

        self.ask_why(all_equations)

        self.clear()

        image_mob = ImageMobject("dog").move_to(UP * 2)
        block_image, pixel_grid, block = self.get_pixel_block(image_mob, 126, 126)
        row = 7
        print(block[:, :, 0])
        print(f"Block row: {row}\n", block[:, :, 0][row])
        pixel_row_mob, row_values = self.get_pixel_row_mob(block, row)
        print("Selected row values\n", row_values)
        pixel_row_mob.move_to(UP * 3)
        self.play(
            FadeIn(pixel_row_mob)
        )
        self.wait()

        row_values_centered = format_block(row_values)
        print('After centering\n', row_values_centered)

        dct_row_pixels = dct_1d(row_values_centered)

        ax, graph, dots = self.draw_image_graph(dct_row_pixels)

        self.show_graph(ax, graph, dots, pixel_row_mob)

        group, arrow = self.show_summing_different_cosine_waves(graph)

        self.show_pixel_rows_for_cosine_freqs(group, arrow)

        self.describe_dct_broadly(ax, graph, dots, pixel_row_mob, dct_row_pixels)

        self.derive_matrix()

    def show_equations(self):
        dct_text = Tex("Discrete Cosine Transform (DCT)").scale(1.2)
        forward_dct = MathTex(r"X_k = \left(\frac{2}{N}\right)^{\frac{1}{2}} \sum_{n=0}^{N-1} \Lambda(n) \cdot \cos \left[\frac{\pi k}{2N}(2n+1)\right]x_n")

        lambda_def = MathTex(r"\Lambda(n) = \left\{\begin{array}{ll} \frac{1}{\sqrt{2}} & \quad n = 0 \\ 1 & \quad n \neq 0 \end{array}\right.").scale(0.8)
        k_def = MathTex(r"k \in \{0, 1, \ldots , N - 1\}").scale(0.8)
        additional_def_group = VGroup(k_def, lambda_def).arrange(RIGHT, buff=1)

        dct_group = VGroup(dct_text, forward_dct, additional_def_group).arrange(DOWN)

        idct_text = Tex("Inverse Discrete Cosine Transform (IDCT)").scale(1.2)
        inverse_dct = MathTex(r"x_k = \frac{X_0}{\sqrt{N}} + \left(\frac{2}{N}\right)^{\frac{1}{2}} \sum_{n=1}^{N-1} \cos \left[\frac{\pi n}{2N}(2k+1)\right]X_n")

        idct_group = VGroup(idct_text, inverse_dct).arrange(DOWN)

        group = VGroup(dct_group, idct_group).arrange(DOWN, buff=1)

        self.play(
            FadeIn(group)
        )
        self.wait()

        return group

    def ask_why(self, equations):
        self.play(
            equations.animate.scale(0.7)
        )
        surround_rect = SurroundingRectangle(equations, buff=SMALL_BUFF)
        self.play(
            Create(surround_rect)
        )

        question_1 = Tex("Where do these equations from?")
        question_2 = Tex("Why do we use these transforms?")
        questions = VGroup(question_1, question_2).arrange(DOWN)
        questions.next_to(surround_rect, UP)

        self.play(
            Write(questions[0])
        )
        self.wait()

        self.play(
            Write(questions[1])
        )

        self.wait()

    def show_summing_different_cosine_waves(self, axes):
        arrow = MathTex(r"\Updownarrow")

        arrow.next_to(axes, DOWN).shift(DOWN * 1)

        self.play(
            Write(arrow)
        )
        self.wait()

        first_freq = self.get_cosine_wave(lambda x: np.cos(x))
        second_freq = self.get_cosine_wave(lambda x: np.cos(2 * x))
        last_freq = self.get_cosine_wave(lambda x: np.cos(7 * x))

        plus = MathTex("+")
        ellipses = MathTex(r"\cdots")

        group = VGroup(first_freq, plus, second_freq, plus.copy(), ellipses, plus.copy(), last_freq).arrange(RIGHT)

        group.next_to(arrow, DOWN * 2)

        self.play(
            FadeIn(group)
        )
        self.wait()

        return group, arrow

    def get_cosine_wave(self, cosine_function):
        ax = Axes(
            x_range=[0, np.pi],
            y_range=[-1, 1],
            x_length=2,
            y_length=2,
        )

        graph = ax.plot(cosine_function).set_color(REDUCIBLE_YELLOW)

        box = SurroundingRectangle(graph, color=REDUCIBLE_VIOLET)
        return VGroup(graph, box)

    def get_cosine_wave_with_ax(self, cosine_function):
        ax = Axes(
            x_range=[0, np.pi],
            y_range=[-1, 1],
            x_length=4.375,
            y_length=2,
            tips=False,
            x_axis_config={"include_numbers": False, "include_ticks": False},
            y_axis_config={"include_numbers": True, "numbers_to_exclude": [0], "include_ticks": False}
        )

        graph = ax.plot(cosine_function).set_color(REDUCIBLE_YELLOW)
        pi_label = MathTex(r"\pi")
        pi_label.next_to(ax.x_axis, DOWN, aligned_edge=RIGHT)
        ax.add(pi_label)
        group = VGroup(ax, graph)

        return group

    def show_pixel_rows_for_cosine_freqs(self, cosine_waves, arrow):
        new_group = VGroup(*[mob.copy() for mob in cosine_waves]).arrange(RIGHT, buff=0.7)
        new_group.move_to(cosine_waves.get_center())
        self.play(
            FadeOut(arrow),
            Transform(cosine_waves, new_group)
        )
        self.wait()

        first_freq = new_group[0]
        second_freq = new_group[2]
        last_freq = new_group[-1]

        first_freq_dct_pixels, _ = self.get_dct_component(1)
        first_freq_pixel_row = self.make_row_of_pixels(first_freq_dct_pixels, height=SMALL_BUFF * 3)
        first_freq_pixel_row.next_to(first_freq, UP)

        second_freq_dct_pixels, _ = self.get_dct_component(2)
        second_freq_pixel_row = self.make_row_of_pixels(second_freq_dct_pixels, height=SMALL_BUFF * 3)
        second_freq_pixel_row.next_to(second_freq, UP)

        last_freq_dct_pixels, _ = self.get_dct_component(7)
        last_freq_pixel_row =  self.make_row_of_pixels(last_freq_dct_pixels, height=SMALL_BUFF * 3)
        last_freq_pixel_row.next_to(last_freq, UP)

        self.play(
            FadeIn(first_freq_pixel_row),
            FadeIn(second_freq_pixel_row),
            FadeIn(last_freq_pixel_row),
        )

        self.wait()

        weight1 = MathTex(r"X_1").next_to(first_freq_pixel_row, UP)
        weight2 = MathTex(r"X_2").next_to(second_freq_pixel_row, UP)
        weightn = MathTex(r"X_N").next_to(last_freq_pixel_row, UP)

        self.play(
            FadeIn(weight1),
            FadeIn(weight2),
            FadeIn(weightn)
        )
        self.wait()

        cross = Cross(VGroup(last_freq_pixel_row, last_freq, weightn))
        self.play(
            Create(cross)
        )
        self.wait()

        self.play(
            FadeOut(cross),
            FadeOut(weight1),
            FadeOut(weight2),
            FadeOut(weightn),
            FadeOut(cosine_waves),
            FadeOut(first_freq_pixel_row),
            FadeOut(second_freq_pixel_row),
            FadeOut(last_freq_pixel_row),
        )

    def describe_dct_broadly(self, ax, graph, dots, pixel_row_mob, dct_row_pixels):
        group = VGroup(ax, graph, dots, pixel_row_mob)
        self.play(
            group.animate.move_to(LEFT * 3.5 + DOWN * 0.5)
        )
        self.wait()
        general_vals = [f"x_{i}" for i in range(len(pixel_row_mob))]
        array_mob_symbols = self.get_gen_array_obj(general_vals, length=pixel_row_mob.width, height=pixel_row_mob.height)
        array_mob_symbols.next_to(pixel_row_mob, UP)

        self.play(
            FadeIn(array_mob_symbols)
        )
        self.wait()

        forward_arrow = MathTex(r"\Rightarrow").scale(1.5).shift(RIGHT * SMALL_BUFF * 3)
        self.play(
            Write(forward_arrow)
        )
        self.wait()

        pixel_space_group = VGroup(group, array_mob_symbols)

        dct_ax = self.get_dct_axis(dct_row_pixels, -80, 80)

        dct_graph, dct_points = self.plot_row_values(dct_ax, dct_row_pixels, color=REDUCIBLE_PURPLE)

        dct_graph_components = VGroup(dct_ax, dct_graph, dct_points).move_to(RIGHT * 3.5 + DOWN * 0.5)
        self.play(
            Write(dct_ax),
        )

        vertical_lines = self.get_vertical_lines_from_points(dct_ax, dct_points)

        self.play(
            *[Create(line) for line in vertical_lines],
            *[GrowFromCenter(dot) for dot in dct_points],
        )
        self.wait()

        self.play(
            Create(dct_graph)
        )
        self.wait()

        general_dct_vals = [f"X_{i}" for i in range(len(pixel_row_mob))]

        array_mob_dct_symbols = self.get_gen_array_obj(general_dct_vals, length=pixel_row_mob.width + 0.5, height=pixel_row_mob.height + SMALL_BUFF, color=REDUCIBLE_VIOLET)
        array_mob_dct_symbols.next_to(pixel_row_mob, UP)
        shift_amount = dct_graph_components.get_center()[0] - array_mob_dct_symbols.get_center()[0]
        array_mob_dct_symbols.shift(RIGHT * (shift_amount + 0.3))

        self.play(
            FadeIn(array_mob_dct_symbols)
        )
        self.wait()

        dct_space_group = VGroup(dct_graph_components, vertical_lines, array_mob_dct_symbols)

        dct_coeff_description = Tex(r"$X_k$ is the contribution of cosine wave $C_k$")

        dct_coeff_description.move_to(UP * 3.5)

        next_question = Tex(r"What cosine waves $C_k$ should we use?").scale(0.8)

        next_question.next_to(dct_coeff_description, DOWN)

        self.play(
            Write(dct_coeff_description)
        )
        self.wait()

        self.play(
            Write(next_question)
        )

        self.wait()

        general_properties = Tex("What properties do we want?")
        general_properties.move_to(UP * 3.5)

        self.play(
            ReplacementTransform(dct_coeff_description, general_properties),
            FadeOut(next_question)
        )
        self.wait()

        invertibility = Tex("Invertibility").move_to(general_properties.get_center())

        forward_dct_group = VGroup(pixel_space_group, forward_arrow, dct_space_group)

        self.play(
            ReplacementTransform(general_properties, invertibility)
        )
        self.wait()

        surround_rect_forward = SurroundingRectangle(forward_dct_group, color=REDUCIBLE_YELLOW, buff=MED_SMALL_BUFF)

        self.play(
            Create(surround_rect_forward)
        )
        self.wait()

        self.play(
            forward_dct_group.animate.scale(0.65).shift(UP * 1.5),
            surround_rect_forward.animate.scale(0.65).shift(UP * 1.5)
        )

        self.wait()
        shift_down = DOWN * 3.5
        new_idct_group_left = dct_space_group.copy().move_to(pixel_space_group.get_center()).shift(shift_down)
        new_idct_arrow = forward_arrow.copy().shift(shift_down)
        new_idct_group_right = pixel_space_group.copy().move_to(dct_space_group.get_center()).shift(shift_down)

        self.play(
            TransformFromCopy(dct_space_group, new_idct_group_left)
        )
        self.wait()

        self.play(
            TransformFromCopy(forward_arrow, new_idct_arrow)
        )
        self.wait()

        self.play(
            TransformFromCopy(pixel_space_group, new_idct_group_right)
        )
        self.wait()

        inverse_dct_group = VGroup(new_idct_group_left, new_idct_arrow, new_idct_group_right)

        surround_rect_inverse = SurroundingRectangle(inverse_dct_group, color=REDUCIBLE_PURPLE, buff=MED_SMALL_BUFF)

        self.play(
            Create(surround_rect_inverse)
        )
        self.wait()

        shift_left = LEFT * 2

        self.play(
            surround_rect_forward.animate.shift(shift_left),
            surround_rect_inverse.animate.shift(shift_left),
            forward_dct_group.animate.shift(shift_left),
            inverse_dct_group.animate.shift(shift_left),
        )
        self.wait()

        forward_transform = MathTex(r"\vec{X} = M \vec{x}")
        forward_transform.next_to(surround_rect_forward, RIGHT).shift(RIGHT * 1.3)
        inverse_transform = MathTex(r"\vec{x} = M^{-1} \vec{X}")
        inverse_transform.next_to(surround_rect_inverse, RIGHT).shift(RIGHT * 1)

        self.play(
            FadeIn(forward_transform),
            FadeIn(inverse_transform)
        )
        self.wait()

        forward_dct_text = Tex("DCT").scale(1.2)
        inverse_dct_text = Tex("IDCT").scale(1.2)

        forward_dct_text.next_to(forward_transform, UP)
        inverse_dct_text.next_to(inverse_transform, UP)

        self.play(
            FadeIn(forward_dct_text),
            FadeIn(inverse_dct_text)
        )
        self.wait()

        self.clear()

    def derive_matrix(self):
        matrix_m_def = Tex("How should we define matrix $M$?").move_to(UP * 3.5)
        self.play(
            Write(matrix_m_def)
        )
        self.wait()
        row_values = np.ones(8) * 255
        pixel_row_mob = self.make_row_of_pixels(row_values)
        pixel_row_mob.next_to(matrix_m_def, DOWN * 2)

        self.play(
            FadeIn(pixel_row_mob)
        )
        self.wait()
        row_values_centered = format_block(row_values)
        print('After centering\n', row_values_centered)

        dct_row_pixels = dct_1d(row_values_centered)
        print(dct_row_pixels)

        ax, graph, dots = self.draw_image_graph(dct_row_pixels)

        self.show_graph(ax, graph, dots, pixel_row_mob)

        cosine_graph = self.get_cosine_wave_with_ax(lambda x: np.cos(0))
        cosine_graph.next_to(ax, DOWN)
        cosine_freq_0_func = MathTex(r"y = \cos (0 \cdot x)")
        cosine_freq_0_func.move_to(DOWN  * 3.5)
        self.play(
            FadeIn(cosine_graph),
            FadeIn(cosine_freq_0_func)
        )
        self.wait()

        group = VGroup(ax, graph, dots, pixel_row_mob)
        self.play(
            group.animate.move_to(LEFT * 3.5 + DOWN * 0.5),
            FadeOut(cosine_graph),
            FadeOut(cosine_freq_0_func)
        )
        self.wait()
        array_mob = self.get_array_obj(row_values)
        array_mob.next_to(pixel_row_mob, UP)

        self.play(
            FadeIn(array_mob)
        )
        self.wait()

        forward_arrow = MathTex(r"\Rightarrow").scale(1.5).shift(RIGHT * SMALL_BUFF * 3)
        self.play(
            Write(forward_arrow)
        )
        self.wait()

        pixel_space_group = VGroup(group, array_mob)

        dct_ax = self.get_dct_axis(dct_row_pixels, -360, 360)

        dct_graph, dct_points = self.plot_row_values(dct_ax, dct_row_pixels, color=REDUCIBLE_PURPLE)

        dct_graph_components = VGroup(dct_ax, dct_graph, dct_points).move_to(RIGHT * 3.5 + DOWN * 0.5)
        self.play(
            Write(dct_ax),
        )

        vertical_lines = self.get_vertical_lines_from_points(dct_ax, dct_points)

        self.play(
            *[Create(line) for line in vertical_lines],
            *[GrowFromCenter(dot) for dot in dct_points],
        )
        self.wait()

        self.play(
            Create(dct_graph)
        )
        self.wait()

        general_dct_vals = ["X_0"] + ["0"] * 7

        array_mob_dct_symbols = self.get_gen_array_obj(general_dct_vals, length=pixel_row_mob.width + 0.5, height=pixel_row_mob.height + SMALL_BUFF, color=REDUCIBLE_VIOLET)
        array_mob_dct_symbols.next_to(pixel_row_mob, UP)
        shift_amount = dct_graph_components.get_center()[0] - array_mob_dct_symbols.get_center()[0]
        array_mob_dct_symbols.shift(RIGHT * (shift_amount + 0.3))

        self.play(
            FadeIn(array_mob_dct_symbols)
        )
        self.wait()

        dct_space_group = VGroup(dct_graph_components, vertical_lines, array_mob_dct_symbols)
        entire_group = VGroup(pixel_space_group, forward_arrow, dct_space_group)

        self.play(
            entire_group.animate.scale(0.7).shift(UP * 1.5)
        )
        self.wait()

        dct_graph_group = VGroup(dct_ax, dct_graph, dct_points, vertical_lines)

        matrix = self.get_matrix_m()

        vector = self.make_column_vector(row_values)
        vector.next_to(matrix, RIGHT)

        equals = MathTex("=").scale(1.5)

        result_vector = self.make_column_vector(general_dct_vals)
        result_vector.next_to(vector, RIGHT)

        matrix_equation = VGroup(matrix, vector, equals, result_vector).arrange(RIGHT, buff=0.5)

        matrix_equation.move_to(DOWN * 2)

        self.play(
            FadeIn(matrix_equation)
        )

        self.wait()

        self.play(
            Indicate(matrix[1][1:])
        )
        self.wait()

        self.play(
            Indicate(vector)
        )
        self.wait()

        self.play(
            LaggedStartMap(Indicate, result_vector[0][1:])
        )
        self.wait()

        original_first_row_matrix = matrix[1][0]

        first_row = VGroup(*[Integer(1).scale(0.8) for _ in range(8)]).arrange(RIGHT, buff=0.4).move_to(original_first_row_matrix.get_center())
        # first_row.stretch_to_fit_width(original_first_row_matrix.width)
        self.play(
            Transform(original_first_row_matrix, first_row)
        )
        self.wait()

        norm_first_row = VGroup(*[MathTex(r"\frac{1}{\sqrt{8}}").scale(0.6).move_to(element.get_center()) for element in first_row])

        self.play(
            Transform(original_first_row_matrix, norm_first_row)
        )
        self.wait()

        self.center_about_zero_and_animate(matrix_m_def, pixel_space_group, dct_graph_group, vector)

    def center_about_zero_and_animate(self, matrix_m_def, pixel_space_group, dct_graph_group, original_vec):
        group, array = pixel_space_group
        ax, graph, dots, pixel_row_mob = group

        tracker = ValueTracker(127)

        def get_new_pixel_space_group():
            row_values = np.ones(8) * tracker.get_value()
            pixel_row_mob = self.make_row_of_pixels(row_values + 128)
            pixel_row_mob.next_to(matrix_m_def, DOWN * 2)

            dct_row_pixels = dct_1d(row_values)

            ax, graph, dots = self.draw_image_graph(dct_row_pixels, centered=True)

            self.show_graph(ax, graph, dots, pixel_row_mob, animate=False)

            group = VGroup(ax, graph, dots, pixel_row_mob)
            group.move_to(LEFT * 3.5 + DOWN * 0.5)
            length = 5 
            if tracker.get_value() <= -99.5:
                length = 5.5
            array_mob = self.get_array_obj(row_values, length=length)
            array_mob.next_to(pixel_row_mob, UP)

            new_pixel_space_group = VGroup(group, array_mob)
            new_pixel_space_group.scale(0.75).move_to(pixel_space_group.get_center())

            return new_pixel_space_group

        def get_new_dct_space_group():
            row_values = np.ones(8) * tracker.get_value()
            dct_row_pixels = dct_1d(row_values)
            dct_ax = self.get_dct_axis(dct_row_pixels, -360, 360)

            dct_graph, dct_points = self.plot_row_values(dct_ax, dct_row_pixels, color=REDUCIBLE_PURPLE)

            dct_graph_components = VGroup(dct_ax, dct_graph, dct_points).move_to(RIGHT * 3.5 + DOWN * 0.5)

            vertical_lines = self.get_vertical_lines_from_points(dct_ax, dct_points)

            new_dct_graph_group = VGroup(dct_ax, dct_graph, dct_points, vertical_lines)
            new_dct_graph_group.scale(0.7).move_to(dct_graph_group.get_center())
            return new_dct_graph_group

        def get_new_column_vector():
            row_values = np.ones(8) * tracker.get_value()
            column_vec = self.make_column_vector(row_values)
            return column_vec.move_to(original_vec.get_center())

        new_pixel_space_group = always_redraw(get_new_pixel_space_group)
        new_dct_graph_group = always_redraw(get_new_dct_space_group)
        new_column_vector = always_redraw(get_new_column_vector)

        self.play(
            ReplacementTransform(pixel_space_group, new_pixel_space_group),
            ReplacementTransform(dct_graph_group, new_dct_graph_group),
            ReplacementTransform(original_vec, new_column_vector)
        )
        self.wait()

        self.play(
            tracker.animate.set_value(-128),
            run_time=5,
            rate_func=linear,
        )

        self.play(
            tracker.animate.set_value(127),
            run_time=5,
            rate_func=linear,
        )

        self.wait()

    def get_matrix_m(self):
        row0 = self.get_cosine_row_tex(0)
        row1 = self.get_cosine_row_tex(1)
        vdots = MathTex(r"\vdots")
        row7 = self.get_cosine_row_tex(7)

        rows = VGroup(row0, row1, vdots, row7).arrange(DOWN).move_to(DOWN * 2)
        bracket_pair = MathTex("[", "]")
        bracket_pair.scale(2)
        bracket_v_buff = MED_SMALL_BUFF
        bracket_h_buff = MED_SMALL_BUFF
        bracket_pair.stretch_to_fit_height(rows.height + 2 * bracket_v_buff)
        l_bracket, r_bracket = bracket_pair.split()
        l_bracket.next_to(rows, LEFT, bracket_h_buff)
        r_bracket.next_to(rows, RIGHT, bracket_h_buff)
        brackets = VGroup(l_bracket, r_bracket)

        return VGroup(brackets, rows)

    def make_column_vector(self, values):
        integer_values = []
        for value in values:
            if isinstance(value, str):
                integer_values.append(value)
            else:
                integer_values.append(int(value))
        vector = Matrix([[value] for value in integer_values], v_buff=0.6, element_alignment_corner=DOWN)
        return vector.scale(0.6)

    def get_cosine_row_tex(self, index):
        text = MathTex(f"C_{index}^T").scale(0.8)
        left_arrow = Arrow(RIGHT * 2, ORIGIN, stroke_width=3, max_tip_length_to_length_ratio=0.15).next_to(text, LEFT).set_color(WHITE)
        right_arrow = Arrow(ORIGIN, RIGHT * 2, stroke_width=3, max_tip_length_to_length_ratio=0.15).next_to(text, RIGHT).set_color(WHITE)

        return VGroup(left_arrow, text, right_arrow)

    def get_vertical_lines_from_points(self, dct_ax, dct_points):
        x_axis_points = [dct_ax.x_axis.n2p(i) for i in range(len(dct_points))]
        vertical_lines = [Line(start, end_point.get_center()).set_stroke(color=REDUCIBLE_VIOLET, width=8) for start, end_point in zip(x_axis_points, dct_points)]
        return VGroup(*vertical_lines)

    def get_gen_array_obj(self, values, length=5, height=0.5, color=REDUCIBLE_GREEN_LIGHTER):
        array = VGroup(*[Rectangle(height=height, width=length/len(values)) for _ in values]).arrange(RIGHT, buff=0)
        array.set_color(color)
        array_text = VGroup(*[MathTex(val).scale(0.6).move_to(array[i].get_center()) for i, val in enumerate(values)])
        return VGroup(array, array_text)

    def make_row_of_pixels(self, row_values, height=SMALL_BUFF*5, num_pixels=8):
        row_length = height * num_pixels
        adjusted_row_values = []
        for val in row_values:
            adjusted_row_values.append(int(round(val)))
        pixel_row_mob = VGroup(
            *[
                Rectangle(height=height, width=row_length / num_pixels)
                .set_stroke(color=REDUCIBLE_GREEN_LIGHTER)
                .set_fill(color=gray_scale_value_to_hex(value), opacity=1)
                for value in adjusted_row_values
            ]
        ).arrange(RIGHT, buff=0)
        return pixel_row_mob

    def get_dct_component(self, col, new_value=250, num_pixels=8):
        dct_row = np.zeros((num_pixels,))
        dct_row[col] = new_value
        return idct_1d(dct_row) + 128, dct_row

class CosineSampling(MotivateDCT):
    def construct(self):
        ax, graph, cosine_label = self.introduce_cosine()
        dots = self.get_dots(ax, graph)
        line_intervals = self.get_intervals(ax, graph)
        ticks = self.get_ticks_for_x_axis()
        vertical_lines = self.get_vertical_lines_from_points(ax, dots)

        self.play(
            *[GrowFromCenter(line) for line in line_intervals],
            *[Write(tick) for tick in ticks]
        )
        self.wait()

        self.play(
            LaggedStartMap(Create, vertical_lines),
            run_time=2
        )
        self.play(
            LaggedStartMap(GrowFromCenter, dots)
        )
        self.wait()

        labels = self.show_sample_x_vals(vertical_lines)

    def introduce_cosine(self):
        ax, graph  = self.get_cosine_wave_with_ax(lambda x: np.cos(x))
        self.play(
            Write(ax)
        )
        self.wait()

        self.play(
            Create(graph)
        )
        self.wait()

        cosine_label = MathTex(r"y = \cos(x)")
        cosine_label.next_to(graph, DOWN)
        self.play(
            Write(cosine_label)
        )
        self.wait()
        return ax, graph, cosine_label

    def get_x_y_points(self):
        x_points = [(j * 2 + 1) * np.pi / 16 for j in range(8)]
        y_points = [np.cos(x) for x in x_points]
        return x_points, y_points

    def get_dots(self, ax, graph, color=REDUCIBLE_YELLOW):
        x_points, y_points = self.get_x_y_points()
        points = [ax.coords_to_point(x, y) for x, y in zip(x_points, y_points)]

        dots = VGroup(*[Dot().set_color(color).move_to(p) for p in points])
        return dots

    def get_intervals(self, ax, graph):
        proportions = np.arange(0, np.pi + 0.0001, np.pi / 8)
        lines = []
        for i in range(len(proportions) - 1):
            start, end = proportions[i], proportions[i + 1]
            start_point, end_point = ax.x_axis.n2p(start), ax.x_axis.n2p(end)
            line = Line(start_point, end_point).set_stroke(width=5)
            if i % 2 == 0:
                line.set_color(REDUCIBLE_GREEN_LIGHTER)
            else:
                line.set_color(REDUCIBLE_GREEN_DARKER)

            lines.append(line)

        return lines

    def show_sample_x_vals(self, vertical_lines):
        labels = VGroup(*[MathTex(r"\frac{\pi}{16}").scale(0.7)] + [MathTex(r"\frac{" + str(2 * i + 1) + r"\pi}{16}").scale(0.6) for i in range(1, len(vertical_lines))])
        for label, line in zip(labels, vertical_lines):
            direction = normalize(line.get_start() - line.get_end())
            direction = np.array([int(c) for c in direction])
            label.next_to(line, direction)

        self.play(
            FadeIn(labels)
        )
        self.wait()
        return labels

    def get_cosine_wave_with_ax(self, cosine_function):
        ax = Axes(
            x_range=[0, np.pi],
            y_range=[-1, 1],
            x_length=10,
            y_length=5.5,
            tips=False,
            x_axis_config={"include_numbers": False, "include_ticks": False},
            y_axis_config={"include_numbers": True, "numbers_to_exclude": [0], "include_ticks": False}
        )

        graph = ax.plot(cosine_function).set_color(REDUCIBLE_YELLOW)
        pi_label = MathTex(r"\pi")
        pi_label.next_to(ax.x_axis, DOWN, aligned_edge=RIGHT)
        ax.add(pi_label)

        group = VGroup(ax, graph)

        return group

    def get_ticks_for_x_axis(self):
        ax = Axes(
            x_range=[0, np.pi, np.pi / 8],
            y_range=[-1, 1],
            x_length=10,
            y_length=5.5,
            tips=False,
            x_axis_config={"include_numbers": False, "include_ticks": True},
            y_axis_config={"include_numbers": True, "numbers_to_exclude": [0], "include_ticks": False}
        )
        return ax.x_axis.ticks

    def get_vertical_lines_from_points(self, ax, points):
        x_points = [ax.x_axis.n2p(p) for p in self.get_x_y_points()[0]]
        vertical_lines = [Line(start_point, end.get_center()).set_stroke(color=REDUCIBLE_VIOLET, width=8) for start_point, end in zip(x_points, points)]
        return VGroup(*vertical_lines)

class RevisedMotivateDCT(MotivateDCT):
    def construct(self):
        image_mob = ImageMobject("dog").move_to(UP * 2)
        block_image, pixel_grid, block = self.get_pixel_block(image_mob, 126, 126)
        row = 7
        print(block[:, :, 0])
        print(f"Block row: {row}\n", block[:, :, 0][row])
        pixel_row_mob, row_values = self.get_pixel_row_mob(block, row)
        print("Selected row values\n", row_values)
        pixel_row_mob.move_to(UP * 3)
        self.play(
            FadeIn(pixel_row_mob)
        )
        self.wait()

        row_values_centered = format_block(row_values)
        print('After centering\n', row_values_centered)

        dct_row_pixels = dct_1d(row_values_centered)

        ax, graph, dots = self.draw_image_graph(dct_row_pixels)

        self.show_graph(ax, graph, dots, pixel_row_mob)

        self.show_summing_different_cosine_waves(graph, dots)

        self.describe_dct_broadly(ax, graph, dots, pixel_row_mob, dct_row_pixels)

        self.clear()

        self.experiment_with_cosine()

    def show_summing_different_cosine_waves(self, graph, original_dots):
        arrow = MathTex(r"\Updownarrow")

        arrow.next_to(graph, DOWN).shift(DOWN * 1)

        self.play(
            Write(arrow)
        )
        self.wait()

        first_freq, first_axes = self.get_cosine_wave(lambda x: np.cos(x))
        second_freq, second_axes = self.get_cosine_wave(lambda x: np.cos(2 * x))
        last_freq, last_axes = self.get_cosine_wave(lambda x: np.cos(7 * x))

        first_freq_dots = self.get_dots(first_axes, first_freq, 1)
        second_freq_dots = self.get_dots(second_axes, second_freq, 2)
        last_freq_dots = self.get_dots(last_axes, last_freq, 7)

        first_cosine_graph = VGroup(first_freq, first_freq_dots)
        second_cosine_graph = VGroup(second_freq, second_freq_dots)
        last_cosine_graph = VGroup(last_freq, last_freq_dots)

        plus = MathTex("+")
        ellipses = MathTex(r"\cdots")

        group = VGroup(first_cosine_graph, plus, second_cosine_graph, plus.copy(), ellipses, plus.copy(), last_cosine_graph).arrange(RIGHT)

        group.next_to(arrow, DOWN * 2)

        self.play(
            FadeIn(group)
        )
        self.wait()

        self.emphasize_sampled_points(original_dots, first_freq_dots, second_freq_dots, last_freq_dots)

        self.emphasize_continuous_funcs(graph, first_freq[0], second_freq[0], last_freq[0])

        self.second_empasize_points(original_dots, first_freq_dots, second_freq_dots, last_freq_dots)

        self.play(
            FadeOut(group),
            FadeOut(arrow)
        )
        self.wait()

    def emphasize_sampled_points(self, original_dots, cosine_1_dots, cosine_2_dots, cosine_7_dots): 
        group_of_dots = []
        for i in range(len(original_dots)):
            group_of_dots.append(VGroup(original_dots[i], cosine_1_dots[i], cosine_2_dots[i], cosine_7_dots[i]))

        self.play(
            LaggedStartMap(Indicate, group_of_dots),
            run_time=3
        )
        self.wait()

    def emphasize_continuous_funcs(self, original_graph, cosine_1_graph, cosine_2_graph, cosine_7_graph):
        self.play(
            ApplyWave(original_graph),
            ApplyWave(cosine_1_graph),
            ApplyWave(cosine_2_graph),
            ApplyWave(cosine_7_graph),
            run_time=2
        )
        self.wait()

    def second_empasize_points(self, original_dots, cosine_1_dots, cosine_2_dots, cosine_7_dots):
        self.play(
            Indicate(VGroup(original_dots, cosine_1_dots, cosine_2_dots, cosine_7_dots))
        )
        self.wait()

    def describe_dct_broadly(self, ax, graph, dots, pixel_row_mob, dct_row_pixels):
        group = VGroup(ax, graph, dots, pixel_row_mob)
        self.play(
            group.animate.move_to(LEFT * 3.5 + DOWN * 0.5)
        )
        self.wait()
        general_vals = [f"x_{i}" for i in range(len(pixel_row_mob))]
        array_mob_symbols = self.get_gen_array_obj(general_vals, length=pixel_row_mob.width, height=pixel_row_mob.height)
        array_mob_symbols.next_to(pixel_row_mob, UP)

        self.play(
            FadeIn(array_mob_symbols)
        )
        self.wait()

        forward_arrow = MathTex(r"\Rightarrow").scale(1.5).shift(RIGHT * SMALL_BUFF * 3)
        self.play(
            Write(forward_arrow)
        )
        self.wait()

        pixel_space_group = VGroup(group, array_mob_symbols)

        dct_ax = self.get_dct_axis(dct_row_pixels, -80, 80)

        dct_graph, dct_points = self.plot_row_values(dct_ax, dct_row_pixels, color=REDUCIBLE_PURPLE)

        dct_graph_components = VGroup(dct_ax, dct_graph, dct_points).move_to(RIGHT * 3.5 + DOWN * 0.5)
        self.play(
            Write(dct_ax),
        )

        vertical_lines = self.get_vertical_lines_from_points(dct_ax, dct_points)

        self.play(
            *[Create(line) for line in vertical_lines],
            *[GrowFromCenter(dot) for dot in dct_points],
        )
        self.wait()

        self.play(
            Create(dct_graph)
        )
        self.wait()

        general_dct_vals = [f"X_{i}" for i in range(len(pixel_row_mob))]

        array_mob_dct_symbols = self.get_gen_array_obj(general_dct_vals, length=pixel_row_mob.width + 0.5, height=pixel_row_mob.height + SMALL_BUFF, color=REDUCIBLE_VIOLET)
        array_mob_dct_symbols.next_to(pixel_row_mob, UP)
        shift_amount = dct_graph_components.get_center()[0] - array_mob_dct_symbols.get_center()[0]
        array_mob_dct_symbols.shift(RIGHT * (shift_amount + 0.3))

        self.play(
            FadeIn(array_mob_dct_symbols)
        )
        self.wait()

        dct_space_group = VGroup(dct_graph_components, vertical_lines, array_mob_dct_symbols)

        dct_coeff_label = Tex("DCT coefficients").scale(0.8)
        brace = Brace(array_mob_dct_symbols, direction=UP)
        self.play(
            GrowFromCenter(brace)
        )
        dct_coeff_label.next_to(brace, UP)
        self.play(
            Write(dct_coeff_label)
        )
        self.wait()

        dct_coeff_description = Tex(r"Coefficient $X_k$ is the contribution of cosine wave $C_k$")

        dct_coeff_description.move_to(UP * 3.5)

        shift_up = UP * 1
        self.play(
            FadeOut(dct_coeff_label),
            FadeOut(brace),
            forward_arrow.animate.shift(shift_up),
            pixel_space_group.animate.shift(shift_up),
            dct_space_group.animate.shift(shift_up),
            Write(dct_coeff_description)
        )
        self.wait()


        next_question = Tex(r"What cosine waves $C_k$ should we use?")

        next_question.move_to(dct_coeff_description.get_center())

        self.show_dct_intuiton(graph, dots, array_mob_dct_symbols)
        self.play(
            ReplacementTransform(dct_coeff_description, next_question)
        )

        self.wait()

        image_connection = Tex("How do cosine waves relate to pixels on an image?")
        image_connection.move_to(next_question.get_center())

        self.play(
            ReplacementTransform(next_question, image_connection)
        )
        self.wait()

    def show_dct_intuiton(self, graph, dots, array_mob_dct_symbols):
        original_smaller_wave = VGroup(graph.copy().scale(0.7), dots.copy().scale(0.7))
        original_smaller_wave.move_to(DOWN * 2.5).to_edge(LEFT * 2)
        surround_rect = SurroundingRectangle(original_smaller_wave)

        original_wave_component = VGroup(original_smaller_wave, surround_rect)

        equals = MathTex("=")

        plus = MathTex("+")

        ellipses = MathTex(r"\cdots")

        cosine_0 = self.make_cosine_component_with_weight(0, 1)

        cosine_1 = self.make_cosine_component_with_weight(1, 2)

        cosine_7 = self.make_cosine_component_with_weight(7, 7)

        intuition_equation = VGroup(
            original_wave_component,
            equals,
            cosine_0,
            plus,
            cosine_1,
            plus.copy(),
            ellipses,
            plus.copy(),
            cosine_7
        ).arrange(RIGHT).move_to(DOWN * 2.6)

        self.play(
            TransformFromCopy(graph, original_smaller_wave[0]),
            TransformFromCopy(dots, original_smaller_wave[1]),
        )

        self.play(
            Create(surround_rect)
        )
        self.wait()

        self.play(
            Write(equals)
        )
        self.wait()

        transforms = self.get_transforms_for_coefficients(array_mob_dct_symbols[1], [cosine_0[0], cosine_1[0], cosine_7[0]], ellipses)

        self.play(
            FadeIn(cosine_0[1]),
            FadeIn(cosine_0[2]),
            FadeIn(cosine_1[1]),
            FadeIn(cosine_1[2]),
            FadeIn(cosine_7[1]),
            FadeIn(cosine_7[2]),
            FadeIn(intuition_equation[3]),
            FadeIn(intuition_equation[5]),
            FadeIn(intuition_equation[6]),
            FadeIn(intuition_equation[7]),
            *transforms,
            run_time=2
        )
        self.wait()

    def get_transforms_for_coefficients(self, array_mob_dct_symbols, new_weights, ellipses):
        transforms = []
        for i, element in enumerate(array_mob_dct_symbols):
            if i not in [0, 1, 7]:
                new_element = element.copy().move_to(ellipses.get_center()).set_stroke(opacity=0).set_fill(opacity=0)
            elif i == 7:
                new_element = new_weights[2]
            else:
                new_element = new_weights[i]
            transforms.append(TransformFromCopy(element, new_element))
        return transforms

    def make_cosine_component_with_weight(self, index, k):
        graph, _ = self.get_cosine_wave(lambda x: np.cos(x * k))
        text = MathTex(f"C_{index}")
        graph[0].set_stroke(opacity=0.3)
        text.scale(1.5).move_to(graph.get_center())
        weight_cosine = MathTex(f"X_{index}")
        weight_cosine.next_to(graph, LEFT, buff=SMALL_BUFF)
        return VGroup(weight_cosine, graph, text).scale(0.75)

    def experiment_with_cosine(self):
        ax, graph, cosine_label = self.introduce_cosine(1)

        cosine_group = VGroup(ax, graph, cosine_label)

        question = self.show_input_to_dct(cosine_group)

        cosine_graph_component = self.show_sampling_scheme(cosine_group, question)

    def introduce_cosine(self, k):
        ax, graph  = self.get_cosine_wave_with_ax(lambda x: np.cos(k * x))
        self.play(
            Write(ax)
        )
        self.wait()

        self.play(
            Create(graph)
        )
        self.wait()

        cosine_label = MathTex(r"y = \cos(x)").scale(1.2)
        cosine_label.next_to(graph, DOWN)
        self.play(
            Write(cosine_label)
        )
        self.wait()
        return ax, graph, cosine_label

    def show_input_to_dct(self, cosine_group):
        self.play(
            cosine_group.animate.scale(0.5).shift(LEFT * 3.5)
        )
        self.wait()

        right_arrow = MathTex(r"\Rightarrow").scale(1.5)
        right_arrow.next_to(cosine_group[0], RIGHT)

        right_arrow.shift(RIGHT * -right_arrow.get_center()[0])

        dct_component_label = Tex(r"DCT($y$)")
        dct_component_label.next_to(right_arrow, UP)

        arrow_and_label = VGroup(right_arrow, dct_component_label)

        box_around_cosine = SurroundingRectangle(cosine_group, color=REDUCIBLE_VIOLET, buff=SMALL_BUFF)

        dct_mystery_component, box_around_dct = self.get_dct_mystery_component()

        arrow_and_label.move_to((box_around_cosine.get_right() + box_around_dct.get_left()) / 2)
        
        self.play(
            Write(right_arrow),
            Write(dct_component_label),
        )
        self.wait()

        question_mark = Tex("?").scale(4)

        question_mark.move_to(dct_mystery_component.get_center())

        self.play(
            FadeIn(dct_mystery_component),
            FadeIn(question_mark),
            Create(box_around_dct)
        )
        self.wait()
        problem = Tex("Problem: we need sampled points on our cosine wave")
        problem.move_to(UP * 3.5)

        question = Tex("How should we sample the cosine function?").move_to(problem.get_center())

        self.play(
            Create(box_around_cosine),
            Write(problem)
        )
        self.wait()

        self.play(
            FadeOut(right_arrow),
            FadeOut(dct_component_label),
            FadeOut(dct_mystery_component),
            FadeOut(question_mark),
            FadeOut(box_around_cosine),
            FadeOut(problem),
            FadeOut(box_around_dct),
            cosine_group.animate.scale(2).shift(RIGHT * 3.5),
            ReplacementTransform(problem, question)
        )

        return question

    def get_dct_mystery_component(self):
        min_x, min_y = -80, 80
        random_dct_row = np.array([np.random.uniform(min_x, min_y) for _ in range(8)])
        dct_ax = self.get_random_dct_axis(random_dct_row, min_x, min_y)

        dct_graph, dct_points = self.plot_row_values(dct_ax, random_dct_row, color=REDUCIBLE_PURPLE)

        vertical_lines = self.get_vertical_lines_from_points(dct_ax, dct_points)
        dct_graph_components = VGroup(dct_ax, dct_graph, dct_points, vertical_lines).scale(1.1).move_to(RIGHT * 3.7 + DOWN * 0.3)
        surround_rect = SurroundingRectangle(dct_graph_components, color=REDUCIBLE_PURPLE)
        # dct_graph_components.set_stroke(opacity=0.5).set_fill(opacity=0.5)

        return dct_graph_components.fade(0.8), surround_rect

    def get_random_dct_axis(self, dct_row, min_y, max_y):
        ax = Axes(
            x_range=[0, dct_row.shape[0] - 1, 1],
            y_range=[min_y, max_y, 1],
            y_length=3,
            x_length=4.375,
            tips=False,
            axis_config={"include_numbers": True, "include_ticks": False},
            x_axis_config={"numbers_to_exclude": list(range(1, dct_row.shape[0] + 1))},
            y_axis_config={"numbers_to_exclude": list(range(min_y, max_y + 1))},
        )
        return ax

    def show_sampling_scheme(self, cosine_group, question):
        ax, graph, cosine_label = cosine_group
        dots = self.get_dots(ax, graph, 1, scale=1)
        line_intervals = self.get_intervals(ax, graph)
        ticks = self.get_ticks_for_x_axis()
        vertical_lines = self.get_cosine_vertical_lines_from_points(ax, dots, 1)

        self.play(
            *[GrowFromCenter(line) for line in line_intervals],
            *[Write(tick) for tick in ticks]
        )
        self.wait()

        self.play(
            LaggedStartMap(Create, vertical_lines)
        )
        self.wait()

        self.play(
            LaggedStartMap(GrowFromCenter, dots),
        )
        self.wait()

        labels = self.show_sample_x_vals(vertical_lines)

        cosine_sampling_math = MathTex(r"y_n = \cos \left[\frac{(2n+1) \pi}{16}\right]", r"\quad n \in \{0, 1, \ldots , 7\}").scale(0.9)

        cosine_sampling_math.move_to(cosine_label.get_center())

        self.play(
            ReplacementTransform(cosine_label, cosine_sampling_math)
        )
        self.wait()

        cosine_sampling_general = MathTex(r"y_n = \cos \left[\frac{(2n+1) \pi}{2N}\right]", r"\quad n \in \{0, 1, \ldots , N - 1\}").scale(0.9)

        cosine_sampling_general.move_to(cosine_label.get_center())

        self.play(
            ReplacementTransform(cosine_sampling_math, cosine_sampling_general)
        )
        self.wait()
        
        cosine_function_sample_group = VGroup(ax, graph, dots, line_intervals, ticks, vertical_lines)
        self.play(
            FadeOut(labels),
            FadeOut(question),
            FadeOut(cosine_sampling_general),
            cosine_function_sample_group.animate.scale(0.5).shift(LEFT * 3.5)
        )

        self.play(
            *[ScaleInPlace(dot, 1.4) for dot in dots],
        )
        self.wait()

        y_labels = [f"y_{i}" for i in range(8)]

        cosine_y_array = self.get_gen_array_obj(y_labels, color=REDUCIBLE_GREEN_LIGHTER)

        cosine_y_array.next_to(cosine_function_sample_group, UP * 2.5)

        self.play(
            FadeIn(cosine_y_array)
        )

        upper_brace = Brace(cosine_y_array, direction=UP)
        vector_y_label = MathTex(r"\vec{y}").next_to(upper_brace, UP)

        self.play(
            GrowFromCenter(upper_brace),
            Write(vector_y_label)
        )
        self.wait()

        right_arrow = MathTex(r"\Rightarrow").scale(1.5)
        right_arrow.next_to(cosine_function_sample_group[0], RIGHT)

        right_arrow.shift(RIGHT * -right_arrow.get_center()[0])

        dct_component_label = Tex(r"DCT($\vec{y}$)")
        dct_component_label.next_to(right_arrow, UP)

        arrow_and_label = VGroup(right_arrow, dct_component_label)

        box_around_cosine = SurroundingRectangle(cosine_function_sample_group, color=REDUCIBLE_VIOLET, buff=SMALL_BUFF)

        dct_mystery_component, box_around_dct = self.get_dct_mystery_component()

        arrow_and_label.move_to((box_around_cosine.get_right() + box_around_dct.get_left()) / 2)
        
        self.play(
            Write(right_arrow),
            Write(dct_component_label),
        )

        self.wait()

        question_mark = Tex("?").scale(4)

        question_mark.move_to(dct_mystery_component.get_center())

        self.play(
            FadeIn(dct_mystery_component),
            FadeIn(question_mark),
            Create(box_around_dct)
        )
        self.wait()

        dct_coeff_labels = [f"X_{i}" for i in range(8)]
        dct_coeff_array = self.get_gen_array_obj(dct_coeff_labels, color=REDUCIBLE_VIOLET)

        dct_coeff_array.next_to(box_around_dct, UP)

        self.play(
            FadeIn(dct_coeff_array)
        )
        self.wait()

        self.play(
            arrow_and_label.animate.shift(UP * 2.8 + LEFT * SMALL_BUFF)
        )

        self.play(
            Circumscribe(dct_coeff_array, color=REDUCIBLE_YELLOW)
        )
        self.wait()

        self.begin_various_cosine_experiments(cosine_function_sample_group, dct_mystery_component, question_mark, box_around_dct)

        self.clear()

        components = VGroup(*[self.get_generic_cosine_component(i) for i in range(8)])
        components.arrange_in_grid(rows=2, buff=0.5)

        surround_rect = SurroundingRectangle(components, buff=MED_SMALL_BUFF)

        self.play(
            FadeIn(components)
        )
        self.wait()

        self.play(
            Create(surround_rect)
        )
        self.wait()

        sigma = MathTex(r"\Sigma").scale(5).move_to(surround_rect.get_center())
        self.play(
            Write(sigma),
            components.animate.fade(0.7)
        )
        self.wait()

        down_arrow = MathTex(r"\Downarrow").next_to(surround_rect, DOWN).scale(1.5)

        self.play(
            Write(down_arrow)
        )
        self.wait()

        def get_random_pixel_row():
            random_values = np.array([np.random.uniform(0, 255) for _ in range(8)])
            random_pix_mob = self.make_row_of_pixels(random_values, height=0.8)
            random_pix_mob.next_to(down_arrow, DOWN)
            return random_pix_mob

        random_pix_mob = get_random_pixel_row()

        self.play(
            FadeIn(random_pix_mob)
        )
        self.wait()

        for _ in range(20):
            random_pix_mob.become(get_random_pixel_row())
            self.wait(0.5)

    def begin_various_cosine_experiments(self, cosine_function_sample_group, dct_mystery_component, question_mark, box_around_dct):
        frequency_tracker = ValueTracker(1)
        amplitude_tracker = ValueTracker(1)
        y_intercept_tracker = ValueTracker(0)

        def get_input_points():
            k = frequency_tracker.get_value()
            a = amplitude_tracker.get_value()
            b = y_intercept_tracker.get_value()
            return np.array([a * np.cos((j * 2 + 1) * k * np.pi / 16) + b for j in range(8)])

        def get_dct_values():
            input_points = get_input_points()
            return dct_1d(input_points)

        def get_dots(ax, graph, scale=1, color=REDUCIBLE_YELLOW):
            x_points = [(j * 2 + 1) * np.pi / 16 for j in range(8)]
            y_points = get_input_points()
            points = [ax.coords_to_point(x, y) for x, y in zip(x_points, y_points)]
            dots = VGroup(*[Dot().scale(scale).set_color(color).move_to(p) for p in points])
            return dots

        def get_cosine_vertical_lines_from_points(ax, points, color=REDUCIBLE_VIOLET):
            x_points = [(j * 2 + 1) * np.pi / 16 for j in range(8)]
            x_axis_points = [ax.x_axis.n2p(p) for p in x_points]
            # print([(start_point, end.get_center()) start_point, end in zip(x_points, points)])
            vertical_lines = [Line(start_point, end.get_center()).set_stroke(color=color, width=8) for start_point, end in zip(x_axis_points, points)]
            return VGroup(*vertical_lines)

        min_cos_y, max_cos_y = -5, 5
        def get_input_cosine_graph():
            nonlocal min_cos_y, max_cos_y
            scale = 0.5
            input_points = get_input_points()
            #TODO: Fix this logic
            if max(input_points) > max_cos_y or min(input_points) < min_cos_y:
                max_cos_y = int((max(input_points) * 3) // 1 + 1)
                min_cos_y = int((min(input_points) * 3) // 1 - 1)
                max_cos_y = max(abs(max_cos_y), abs(min_cos_y))
                min_cos_y = -max_cos_y
            
                if max_cos_y > 127:
                    max_cos_y = 128
                    min_cos_y = -128

            location = cosine_function_sample_group.get_center()
            k = frequency_tracker.get_value()
            a = amplitude_tracker.get_value()
            b = y_intercept_tracker.get_value()
            ax, graph  = self.get_cosine_wave_with_ax(lambda x: a * np.cos(x * k) + b, min_y=min_cos_y, max_y=max_cos_y)
            
            dots = get_dots(ax, graph)
            for dot in dots:
                dot.scale(1.4)

            line_intervals = self.get_intervals(ax, graph)
            ticks = self.get_ticks_for_x_axis()
            vertical_lines = get_cosine_vertical_lines_from_points(ax, dots)

            new_cosine_function_sample_group = VGroup(ax, graph, dots, line_intervals, ticks, vertical_lines)
            new_cosine_function_sample_group.scale(scale).move_to(location)
            for label in ax.y_axis.numbers:
                label.scale(2).shift(LEFT * SMALL_BUFF)

            new_cosine_function_sample_group.move_to(location)

            return new_cosine_function_sample_group

        min_y, max_y = -5, 5
        def get_output_dct_graph():
            nonlocal min_y, max_y
            dct_values = get_dct_values()
            # TODO: fix this logic
            if max(dct_values) > max_y or min(dct_values) < min_y:
                max_y = int((max(dct_values) * 3) // 1 + 1)
                min_y = int((min(dct_values) * 3) // 1 - 1)
                max_y = max(abs(max_y), abs(min_y))
                min_y = -max_y

                if max_y > 250:
                    max_y = 250
                    min_y = -250
            dct_ax = self.get_dct_axis(dct_values, min_y, max_y)

            dct_graph, dct_points = self.plot_row_values(dct_ax, dct_values, color=REDUCIBLE_PURPLE)

            vertical_lines = self.get_vertical_lines_from_points(dct_ax, dct_points)
            dct_graph_components = VGroup(dct_ax, dct_graph, dct_points, vertical_lines).scale(1.1).move_to(RIGHT * 3.7 + DOWN * 0.3)
            return dct_graph_components

        output_dct_graph = always_redraw(get_output_dct_graph)

        self.play(
            FadeOut(question_mark),
            FadeOut(box_around_dct),
            ReplacementTransform(dct_mystery_component, output_dct_graph),
        )
        self.wait()

        input_cosine_graph = always_redraw(get_input_cosine_graph)
        self.play(
            ReplacementTransform(cosine_function_sample_group, input_cosine_graph)
        )
        self.wait()

        self.play(
            amplitude_tracker.animate.set_value(50),
            run_time=4
        )
        self.wait()

        self.play(
            amplitude_tracker.animate.set_value(-50),
            run_time=4
        )
        self.wait()

        self.play(
            amplitude_tracker.animate.set_value(50),
            run_time=2
        )
        self.wait()

        original_pixel_space_rep = self.get_pixel_space_representation()
        original_pixel_space_rep.move_to(DOWN * 2.8)
        lower_bound = Integer(0).next_to(original_pixel_space_rep, LEFT)
        upper_bound = Integer(255).next_to(original_pixel_space_rep, RIGHT)
        self.play(
            FadeIn(original_pixel_space_rep),
            FadeIn(lower_bound),
            FadeIn(upper_bound)
        )
        self.wait()

        new_pixel_space_rep = original_pixel_space_rep.copy().next_to(original_pixel_space_rep, DOWN)
        new_lower_bound = Integer(-128).next_to(new_pixel_space_rep, LEFT)
        new_upper_bound = Integer(127).next_to(new_pixel_space_rep, RIGHT)
        self.play(
            FadeIn(new_pixel_space_rep),
            TransformFromCopy(lower_bound, new_lower_bound),
            TransformFromCopy(upper_bound, new_upper_bound)
        )
        self.wait()

        def get_pixel_rep_of_cosine():
            input_points = get_input_points() + 128
            pixel_array_mob = self.make_row_of_pixels(input_points, height=SMALL_BUFF*6)
            pixel_array_mob.next_to(input_cosine_graph, DOWN).shift(RIGHT * SMALL_BUFF * 2)
            return pixel_array_mob

        pixel_row_mob = always_redraw(get_pixel_rep_of_cosine)
        self.play(
            FadeIn(pixel_row_mob),
            FadeOut(new_pixel_space_rep),
            FadeOut(original_pixel_space_rep),
            FadeOut(lower_bound),
            FadeOut(new_lower_bound),
            FadeOut(upper_bound),
            FadeOut(new_upper_bound)
        )
        self.wait()

        self.play(
            amplitude_tracker.animate.set_value(120),
            run_time=3,
        )
        self.wait()

        self.play(
            amplitude_tracker.animate.set_value(-120),
            run_time=3
        )

        self.play(
            amplitude_tracker.animate.set_value(80),
            run_time=3
        )
        self.wait()

        self.play(
            y_intercept_tracker.animate.set_value(45),
            run_time=2
        )

        self.wait()

        self.play(
            y_intercept_tracker.animate.set_value(-45),
            run_time=4
        )

        self.wait()

        self.play(
            y_intercept_tracker.animate.set_value(0),
            run_time=2
        )

        self.wait()

        self.play(
            frequency_tracker.animate.set_value(2),
            run_time=6,
            rate_func=linear,
        )
        self.wait()

        cosine_2x = MathTex(r"\cos(2x)", r" \rightarrow X_2")
        cosine_2x.move_to(DOWN * 3.2)
        self.play(
            Write(cosine_2x[0])
        )
        self.wait()

        self.play(
            Write(cosine_2x[1])
        )
        self.wait()

        self.play(
            amplitude_tracker.animate.set_value(120),
            run_time=2,
        )

        self.play(
            amplitude_tracker.animate.set_value(80),
            run_time=2,
        )

        self.play(
            y_intercept_tracker.animate.set_value(45),
            run_time=2
        )

        self.play(
            y_intercept_tracker.animate.set_value(-45),
            run_time=2
        )

        self.play(
            y_intercept_tracker.animate.set_value(0),
            run_time=2
        )
        self.wait()

        self.play(
            frequency_tracker.animate.set_value(3),
            run_time=3,
            rate_func=linear,
        )
        cosine_3x = MathTex(r"\cos(3x)", r" \rightarrow X_3")
        cosine_3x.move_to(DOWN * 3.2)
        self.play(
            ReplacementTransform(cosine_2x, cosine_3x)
        )
        self.wait()

        self.play(
            frequency_tracker.animate.set_value(4),
            run_time=2,
            rate_func=linear,
        )

        cosine_kx = MathTex(r"\cos(kx)", r" \rightarrow X_k")
        cosine_kx.move_to(DOWN * 3.2)
        self.play(
            ReplacementTransform(cosine_3x, cosine_kx)
        )
        self.wait()

        self.play(
            frequency_tracker.animate.set_value(5),
            run_time=2,
            rate_func=linear,
        )
        self.wait()

        self.play(
            frequency_tracker.animate.set_value(6),
            run_time=2,
            rate_func=linear,
        )
        self.wait()

        self.play(
            frequency_tracker.animate.set_value(7),
            run_time=2,
            rate_func=linear,
        )
        self.wait()

        self.play(
            y_intercept_tracker.animate.set_value(45),
            run_time=2
        )

        self.play(
            y_intercept_tracker.animate.set_value(-45),
            run_time=2
        )

        self.play(
            y_intercept_tracker.animate.set_value(0),
            run_time=2
        )
        self.wait()

        self.play(
            frequency_tracker.animate.set_value(0),
            run_time=4,
            rate_func=linear,
        )


        self.play(
            amplitude_tracker.animate.set_value(120),
            run_time=2
        )

        self.play(
            amplitude_tracker.animate.set_value(-120),
            run_time=2
        )

        self.play(
            amplitude_tracker.animate.set_value(80),
            run_time=2
        )

        self.wait()

    def get_generic_cosine_component(self, frequency, cosine_scale=0.5, general_scale=0.5):
        frequency_tracker = ValueTracker(frequency)
        amplitude_tracker = ValueTracker(120)
        y_intercept_tracker = ValueTracker(0)

        def get_input_points():
            k = frequency_tracker.get_value()
            a = amplitude_tracker.get_value()
            b = y_intercept_tracker.get_value()
            return np.array([a * np.cos((j * 2 + 1) * k * np.pi / 16) + b for j in range(8)])

        def get_dct_values():
            input_points = get_input_points()
            return dct_1d(input_points)

        def get_dots(ax, graph, scale=1, color=REDUCIBLE_YELLOW):
            x_points = [(j * 2 + 1) * np.pi / 16 for j in range(8)]
            y_points = get_input_points()
            points = [ax.coords_to_point(x, y) for x, y in zip(x_points, y_points)]
            dots = VGroup(*[Dot().scale(scale).set_color(color).move_to(p) for p in points])
            return dots

        def get_cosine_vertical_lines_from_points(ax, points, color=REDUCIBLE_VIOLET):
            x_points = [(j * 2 + 1) * np.pi / 16 for j in range(8)]
            x_axis_points = [ax.x_axis.n2p(p) for p in x_points]
            vertical_lines = [Line(start_point, end.get_center()).set_stroke(color=color, width=8) for start_point, end in zip(x_axis_points, points)]
            return VGroup(*vertical_lines)

        input_points = get_input_points()
        #TODO: Fix this logic
        max_cos_y, min_cos_y = 120, -120

        k = frequency_tracker.get_value()
        a = amplitude_tracker.get_value()
        b = y_intercept_tracker.get_value()
        ax, graph  = self.get_cosine_wave_with_ax(lambda x: a * np.cos(x * k) + b, min_y=min_cos_y, max_y=max_cos_y, include_end_y_axis_nums=False)
        
        dots = get_dots(ax, graph)
        for dot in dots:
            dot.scale(1.4)

        line_intervals = self.get_intervals(ax, graph)
        ticks = self.get_ticks_for_x_axis()
        vertical_lines = get_cosine_vertical_lines_from_points(ax, dots)

        new_cosine_function_sample_group = VGroup(ax, graph, line_intervals, ticks, vertical_lines, dots)
        new_cosine_function_sample_group.scale(cosine_scale)

        def get_pixel_rep_of_cosine():
            input_points = get_input_points() + 128
            pixel_array_mob = self.make_row_of_pixels(input_points, height=SMALL_BUFF*6.3)
            pixel_array_mob.next_to(new_cosine_function_sample_group, UP, aligned_edge=LEFT)
            return pixel_array_mob

        pixel_space_mob_cos = get_pixel_rep_of_cosine()
        x_comp = MathTex(f"X_{frequency}").scale(1.5).next_to(pixel_space_mob_cos, UP)

        return VGroup(new_cosine_function_sample_group, pixel_space_mob_cos, x_comp).scale(general_scale)

    def get_cosine_wave(self, cosine_function, color=REDUCIBLE_VIOLET):
        ax = Axes(
            x_range=[0, np.pi],
            y_range=[-1, 1],
            x_length=2,
            y_length=2,
        )

        graph = ax.plot(cosine_function).set_color(REDUCIBLE_VIOLET)

        box = SurroundingRectangle(graph, color=REDUCIBLE_VIOLET)
        return VGroup(graph, box), ax

    def get_cosine_wave_with_ax(self, cosine_function, min_y=-1, max_y=1, color=REDUCIBLE_VIOLET, include_end_y_axis_nums=True):
        if include_end_y_axis_nums:
            y_range_exclude = range(min_y + 1, max_y)
        else:
            y_range_exclude = range(min_y, max_y + 1)
        ax = Axes(
            x_range=[0, np.pi],
            y_range=[min_y, max_y],
            x_length=10,
            y_length=5.5,
            tips=False,
            x_axis_config={"include_numbers": False, "include_ticks": False},
            y_axis_config={"include_numbers": True, "numbers_to_exclude": y_range_exclude, "include_ticks": False}
        )

        graph = ax.plot(cosine_function).set_color(color)
        pi_label = MathTex(r"\pi")
        pi_label.next_to(ax.x_axis, DOWN, aligned_edge=RIGHT)
        ax.add(pi_label)

        group = VGroup(ax, graph)

        return group

    def get_x_y_points(self, k):
        x_points = [(j * 2 + 1) * np.pi / 16 for j in range(8)]
        y_points = [np.cos(x * k) for x in x_points]
        return x_points, y_points

    def get_dots(self, ax, graph, k, color=REDUCIBLE_YELLOW, scale=0.7):
        x_points, y_points = self.get_x_y_points(k)
        points = [ax.coords_to_point(x, y) for x, y in zip(x_points, y_points)]

        dots = VGroup(*[Dot().scale(scale).set_color(color).move_to(p) for p in points])
        return dots

    def get_intervals(self, ax, graph):
        proportions = np.arange(0, np.pi + 0.0001, np.pi / 8)
        lines = []
        for i in range(len(proportions) - 1):
            start, end = proportions[i], proportions[i + 1]
            start_point, end_point = ax.x_axis.n2p(start), ax.x_axis.n2p(end)
            line = Line(start_point, end_point).set_stroke(width=5)
            if i % 2 == 0:
                line.set_color(REDUCIBLE_GREEN_LIGHTER)
            else:
                line.set_color(REDUCIBLE_GREEN_DARKER)

            lines.append(line)

        return VGroup(*lines)

    def show_sample_x_vals(self, vertical_lines):
        labels = VGroup(*[MathTex(r"\frac{\pi}{16}").scale(0.7)] + [MathTex(r"\frac{" + str(2 * i + 1) + r"\pi}{16}").scale(0.6) for i in range(1, len(vertical_lines))])
        for label, line in zip(labels, vertical_lines):
            direction = normalize(line.get_start() - line.get_end())
            direction = np.array([int(c) for c in direction])
            label.next_to(line, direction)

        self.play(
            FadeIn(labels)
        )
        self.wait()
        return labels

    def get_cosine_vertical_lines_from_points(self, ax, points, k, color=REDUCIBLE_VIOLET):
        x_points = [ax.x_axis.n2p(p) for p in self.get_x_y_points(k)[0]]
        vertical_lines = [Line(start_point, end.get_center()).set_stroke(color=color, width=8) for start_point, end in zip(x_points, points)]
        return VGroup(*vertical_lines)

    def get_ticks_for_x_axis(self):
        ax = Axes(
            x_range=[0, np.pi, np.pi / 8],
            y_range=[-1, 1],
            x_length=10,
            y_length=5.5,
            tips=False,
            x_axis_config={"include_numbers": False, "include_ticks": True},
            y_axis_config={"include_numbers": True, "numbers_to_exclude": [0], "include_ticks": False}
        )
        return ax.x_axis.ticks

    def make_component(self, text, color=REDUCIBLE_YELLOW, scale=0.8):
        # geometry is first index, Tex is second index
        text_mob = Tex(text).scale(scale)
        rect = Rectangle(color=color, height=1.1, width=3)
        return VGroup(rect, text_mob)

    def get_pixel_space_representation(self, height=SMALL_BUFF*5, row_length=6):
        num_pixels = 256
        row_values = range(256)
        pixel_row_mob = VGroup(
            *[
                Rectangle(height=height, width=row_length / num_pixels)
                .set_stroke(color=gray_scale_value_to_hex(value))
                .set_fill(color=gray_scale_value_to_hex(value), opacity=1)
                for value in row_values
            ]
        ).arrange(RIGHT, buff=0)
        surround_rect = SurroundingRectangle(pixel_row_mob, color=REDUCIBLE_GREEN_LIGHTER, buff=0)
        return VGroup(pixel_row_mob, surround_rect)

class MathematicallyDefineDCT(RevisedMotivateDCT):
    def construct(self):
        image_mob = ImageMobject("dog").move_to(UP * 2)
        block_image, pixel_grid, block = self.get_pixel_block(image_mob, 126, 126)
        row = 7
        print(block[:, :, 0])
        print(f"Block row: {row}\n", block[:, :, 0][row])
        pixel_row_mob, row_values = self.get_pixel_row_mob(block, row)
        print("Selected row values\n", row_values)
        pixel_row_mob.move_to(UP * 3)

        row_values_centered = format_block(row_values)
        print('After centering\n', row_values_centered)

        dct_row_pixels = dct_1d(row_values_centered)

        ax, graph, dots = self.draw_image_graph(dct_row_pixels)

        self.show_graph(ax, graph, dots, pixel_row_mob, animate=False)


        entire_group = self.get_broad_dct_components(ax, graph, dots, pixel_row_mob, dct_row_pixels)

        entire_group.shift(UP * 1)

        pixel_space_group, forward_arrow, dct_space_group = entire_group
        self.play(
            FadeIn(pixel_space_group),
            FadeIn(forward_arrow),
            FadeIn(dct_space_group)
        )
        self.wait()

        pixel_array_symbols = pixel_space_group[-1]
        dct_array_mob_symbols = dct_space_group[-1]

        self.play(
            Circumscribe(pixel_array_symbols)
        )
        self.wait()


        self.play(
            Circumscribe(dct_array_mob_symbols)
        )
        self.wait()

        dct_math = MathTex(r"X_k = ", r"\sum_{n=0}^{N-1} x_n", r"\cos \left[\frac{(2n+1) \pi k}{2N}\right]")

        dct_math.move_to(DOWN * 2.5)

        self.play(
            Write(dct_math[0])
        )
        self.wait()

        self.play(
            FadeIn(dct_math[1]),
            FadeIn(dct_math[2])
        )
        self.wait()

        self.play(
            dct_math[0].animate.set_fill(color=None, opacity=0.5),
            dct_math[1].animate.set_fill(color=None, opacity=0.5)
        )
        self.wait()

        self.play(
            dct_math[0].animate.set_fill(color=None, opacity=1),
            dct_math[1].animate.set_fill(color=None, opacity=1),
        )

        self.play(
            dct_math.animate.to_edge(LEFT * 2)
        )

        self.wait()

        brace_up = Brace(pixel_array_symbols, direction=UP)

        self.play(
            GrowFromCenter(brace_up)
        )
        x_vec = MathTex(r"\vec{x}").next_to(brace_up, UP)

        self.play(
            Write(x_vec)
        )
        self.wait()

        C_k_values = [r" \cos (\frac{\pi}{16} \cdot " + str(2 * i + 1) + r" \cdot k)" for i in [0, 1, 2, 7]]
        C_k_values.insert(-1, r"\vdots")
        C_k_vec_def = self.make_column_vector(C_k_values, v_buff=1.2)

        self.play(
            FadeOut(forward_arrow),
            FadeOut(dct_space_group)
        )
        self.wait()

        self.play(
            Circumscribe(dct_math[2])
        )
        self.wait()

        C_k = MathTex(r"\vec{C}_k = ")
        C_k_vec_def[0][3].shift(UP * SMALL_BUFF * 1.5)

        C_k_vec_group = VGroup(C_k, C_k_vec_def).arrange(RIGHT).move_to(RIGHT * 3.5 + UP * 1.3)
        self.play(
            FadeIn(C_k_vec_group)
        )
        self.wait()

        double_arrow = MathTex(r"\Longleftrightarrow").next_to(dct_math, RIGHT * 2)

        dot_product_def = MathTex(r"X_k = \vec{C}_k^T \vec{x}").next_to(double_arrow, RIGHT * 2)

        dct_equivalence = VGroup(dct_math, double_arrow, dot_product_def)
        self.play(
            Write(double_arrow)
        )
        self.wait()

        self.play(
            Write(dot_product_def)
        )

        self.wait()

        self.play(
            FadeOut(pixel_space_group),
            FadeOut(C_k_vec_group),
            FadeOut(brace_up),
            FadeOut(x_vec),
            dct_equivalence.animate.move_to(UP * 3.2),
            run_time=2

        )
        self.wait()

        forward_dct_matrix_eq = self.get_forward_transform_matrix_eq().scale(1.2)
        self.play(
            FadeIn(forward_dct_matrix_eq)
        )
        self.wait()

        orthoganility = MathTex(r"\vec{C}_i^T \vec{C}_j = 0 \quad \forall i \neq j")
        orthoganility.next_to(forward_dct_matrix_eq, DOWN)

        self.play(
            FadeIn(orthoganility)
        )
        self.wait()

        entire_group.shift(DOWN * 1)

        self.clear()

        self.add(entire_group)

        forward_dct_matrix_eq.scale(1 / 1.2)

        self.show_invertibility(pixel_space_group, forward_arrow, dct_space_group, forward_dct_matrix_eq, graph, dots)

    def show_invertibility(self, pixel_space_group, forward_arrow, dct_space_group, forward_dct_matrix_eq, graph, dots):
        original_graph, original_dots = graph.copy(), dots.copy()
        invertibility = Tex("Invertibility").move_to(UP * 3.5)

        forward_dct_group = VGroup(pixel_space_group, forward_arrow, dct_space_group)

        self.play(
            Write(invertibility)
        )
        self.wait()

        surround_rect_forward = SurroundingRectangle(forward_dct_group, color=REDUCIBLE_YELLOW, buff=MED_SMALL_BUFF)

        self.play(
            Create(surround_rect_forward)
        )
        self.wait()

        self.play(
            forward_dct_group.animate.scale(0.65).shift(UP * 1.5),
            surround_rect_forward.animate.scale(0.65).shift(UP * 1.5)
        )

        self.wait()
        shift_down = DOWN * 3.5
        new_idct_group_left = dct_space_group.copy().move_to(pixel_space_group.get_center()).shift(shift_down)
        new_idct_arrow = forward_arrow.copy().shift(shift_down)
        new_idct_group_right = pixel_space_group.copy().move_to(dct_space_group.get_center()).shift(shift_down)

        self.play(
            TransformFromCopy(dct_space_group, new_idct_group_left)
        )
        self.wait()

        self.play(
            TransformFromCopy(forward_arrow, new_idct_arrow)
        )
        self.wait()

        self.play(
            TransformFromCopy(pixel_space_group, new_idct_group_right)
        )
        self.wait()

        inverse_dct_group = VGroup(new_idct_group_left, new_idct_arrow, new_idct_group_right)

        surround_rect_inverse = SurroundingRectangle(inverse_dct_group, color=REDUCIBLE_PURPLE, buff=MED_SMALL_BUFF)

        self.play(
            Create(surround_rect_inverse)
        )
        self.wait()

        shift_left = LEFT * 2

        self.play(
            surround_rect_forward.animate.shift(shift_left),
            surround_rect_inverse.animate.shift(shift_left),
            forward_dct_group.animate.shift(shift_left),
            inverse_dct_group.animate.shift(shift_left),
        )
        self.wait()

        forward_transform = MathTex(r"\vec{X} = M \vec{x}")
        forward_transform.next_to(surround_rect_forward, RIGHT).shift(RIGHT * 1.3)
        inverse_transform = MathTex(r"\vec{x} = M^{-1} \vec{X}")
        inverse_transform.next_to(surround_rect_inverse, RIGHT).shift(RIGHT * 1)

        self.play(
            FadeIn(forward_transform),
            FadeIn(inverse_transform)
        )
        self.wait()

        forward_dct_text = Tex("DCT").scale(1.2)
        inverse_dct_text = Tex("Inverse DCT").scale(1.2)

        forward_dct_text.next_to(forward_transform, UP)
        inverse_dct_text.next_to(inverse_transform, UP)

        self.play(
            FadeIn(forward_dct_text),
            FadeIn(inverse_dct_text)
        )
        self.wait()


        inverse_dct_matrix_eq = self.get_inverse_tranform_matrix_eq()
        inverse_dct_matrix_eq.scale(0.8).move_to(inverse_dct_group.get_center())
        forward_dct_matrix_eq.scale(0.8).move_to(forward_dct_group.get_center())

        self.play(
            forward_dct_group.animate.fade(0.9),
            FadeIn(forward_dct_matrix_eq),
        )
        self.wait()

        self.play(
            inverse_dct_group.animate.fade(0.9),
            FadeIn(inverse_dct_matrix_eq)
        )
        self.wait()

        asterik = Tex("(*)").next_to(inverse_dct_matrix_eq, RIGHT)
        asterik[0][1].shift(DOWN * SMALL_BUFF)
        note = Tex("*with additional normalization").scale(0.6)

        note.next_to(inverse_transform, DOWN * 2)

        self.add(asterik, note)
        self.wait()

        self.play(
            FadeOut(invertibility),
            FadeOut(asterik),
            FadeOut(note),
            FadeOut(surround_rect_inverse),
            FadeOut(surround_rect_forward),
            inverse_dct_group.animate.fade(1),
            forward_dct_group.animate.fade(1),
            FadeOut(forward_dct_text),
            FadeOut(forward_transform),
            FadeOut(inverse_transform),
            FadeOut(forward_dct_matrix_eq),
            inverse_dct_text.animate.move_to(UP * 3.5),
            inverse_dct_matrix_eq.animate.scale(1 / 0.8).next_to(invertibility, DOWN),
            run_time=2
        )
        self.wait()

        column_split_sum = self.get_column_sum_matrix(inverse_dct_matrix_eq[0])
        column_split_sum.next_to(inverse_dct_matrix_eq, DOWN)

        self.play(
            FadeIn(column_split_sum)
        )

        self.wait()

        self.play(
            FadeOut(inverse_dct_matrix_eq),
            column_split_sum.animate.next_to(inverse_dct_text, DOWN)
        )
        self.wait()

        self.play(
            column_split_sum[0].animate.set_color(REDUCIBLE_YELLOW),
            column_split_sum[2][1].animate.set_color(REDUCIBLE_VIOLET),
            column_split_sum[4][1].animate.set_color(REDUCIBLE_VIOLET),
            column_split_sum[8][1].animate.set_color(REDUCIBLE_VIOLET),
        )
        self.wait()

        cosine_sum_visual_rep = self.show_summing_different_cosine_waves(original_graph, original_dots)

        cosine_sum_visual_rep.scale(0.75).next_to(column_split_sum, DOWN)

        self.play(
            FadeIn(cosine_sum_visual_rep)
        )
        self.wait()

    def show_summing_different_cosine_waves(self, graph, original_dots):
        first_freq, first_axes = self.get_cosine_wave(lambda x: 0.5)
        second_freq, second_axes = self.get_cosine_wave(lambda x: np.cos(x))
        last_freq, last_axes = self.get_cosine_wave(lambda x: np.cos(7 * x))

        first_freq_dots = self.get_dots(first_axes, first_freq, 0).move_to(first_freq.get_center())
        second_freq_dots = self.get_dots(second_axes, second_freq, 1)
        last_freq_dots = self.get_dots(last_axes, last_freq, 7)

        first_surround_rect = second_freq[1].copy().move_to(first_axes.get_center())
        right_shift = first_freq[1].get_center()[0] - first_surround_rect.get_center()[0]
        first_surround_rect.shift(RIGHT * right_shift)

        first_cosine_graph = VGroup(first_freq[0], first_surround_rect, first_freq_dots)
        second_cosine_graph = VGroup(second_freq, second_freq_dots)
        last_cosine_graph = VGroup(last_freq, last_freq_dots)
        
        X_0, X_1, X_7 = MathTex("X_0"), MathTex("X_1"), MathTex("X_7")

        first_cosine_comp = VGroup(X_0,  first_cosine_graph).arrange(RIGHT, buff=SMALL_BUFF)
        second_cosine_comp = VGroup(X_1,second_cosine_graph).arrange(RIGHT, buff=SMALL_BUFF)
        last_cosine_comp = VGroup(X_7, last_cosine_graph).arrange(RIGHT, buff=SMALL_BUFF)

        plus = MathTex("+")
        ellipses = MathTex(r"\cdots")
        equals = MathTex("=")

        original_graph = VGroup(graph, original_dots)
        surround_rect_original = SurroundingRectangle(original_graph, color=REDUCIBLE_YELLOW)
        original_graph = VGroup(original_graph, surround_rect_original)

        group = VGroup(original_graph, equals, first_cosine_comp, plus, second_cosine_comp, plus.copy(), ellipses, plus.copy(), last_cosine_comp).arrange(RIGHT)

        return group

    def get_column_sum_matrix(self, pixel_vec):
        X_0, X_1, X_7 = MathTex("X_0"), MathTex("X_1"), MathTex("X_7")
        c_0 = self.get_individual_column_vec(0)
        c_1 = self.get_individual_column_vec(1)
        c_7 = self.get_individual_column_vec(7)

        first_elem = VGroup(X_0, c_0).arrange(RIGHT, buff=SMALL_BUFF)
        second_elem = VGroup(X_1, c_1).arrange(RIGHT, buff=SMALL_BUFF)
        last_elem = VGroup(X_7, c_7).arrange(RIGHT, buff=SMALL_BUFF)

        plus = MathTex("+")
        equals = MathTex("=")
        cdots = MathTex(r"\cdots")

        column_split_equation = VGroup(
            pixel_vec.copy(),
            equals,
            first_elem,
            plus,
            second_elem,
            plus.copy(),
            cdots,
            plus.copy(),
            last_elem,
        ).arrange(RIGHT)

        return column_split_equation

    def get_matrix_m(self):
        row0 = self.get_cosine_row_tex(0)
        row1 = self.get_cosine_row_tex(1)
        vdots = MathTex(r"\vdots")
        row7 = self.get_cosine_row_tex(7)

        rows = VGroup(row0, row1, vdots, row7).arrange(DOWN).move_to(DOWN * 2)
        bracket_pair = MathTex("[", "]")
        bracket_pair.scale(2)
        bracket_v_buff = MED_SMALL_BUFF
        bracket_h_buff = MED_SMALL_BUFF
        bracket_pair.stretch_to_fit_height(rows.height + 2 * bracket_v_buff)
        l_bracket, r_bracket = bracket_pair.split()
        l_bracket.next_to(rows, LEFT, bracket_h_buff)
        r_bracket.next_to(rows, RIGHT, bracket_h_buff)
        brackets = VGroup(l_bracket, r_bracket)

        return VGroup(brackets, rows)

    def get_inverse_matrix_m(self):
        col0 = self.get_cosine_col_tex(0)
        col1 = self.get_cosine_col_tex(1)
        cdots = MathTex(r"\cdots")
        col7 = self.get_cosine_col_tex(7)

        cols = VGroup(col0, col1, cdots, col7).arrange(RIGHT, buff=0.7).move_to(DOWN * 2)
        bracket_pair = MathTex("[", "]")
        bracket_pair.scale(2)
        bracket_v_buff = MED_SMALL_BUFF
        bracket_h_buff = MED_SMALL_BUFF
        bracket_pair.stretch_to_fit_height(cols.height + 2 * bracket_v_buff)
        l_bracket, r_bracket = bracket_pair.split()
        l_bracket.next_to(cols, LEFT, bracket_h_buff)
        r_bracket.next_to(cols, RIGHT, bracket_h_buff)
        brackets = VGroup(l_bracket, r_bracket)

        return VGroup(brackets, cols)

    def make_column_vector(self, values, v_buff=0.6, scale=0.6):
        integer_values = []
        for value in values:
            if isinstance(value, str):
                integer_values.append(value)
            else:
                integer_values.append(int(value))
        vector = Matrix([[value] for value in integer_values], v_buff=v_buff, element_alignment_corner=DOWN)
        return vector.scale(scale)

    def get_individual_column_vec(self, index):
        cosine_col_text = self.get_cosine_col_tex(index)
        bracket_pair = MathTex("[", "]")
        bracket_pair.scale(2)
        bracket_v_buff = MED_SMALL_BUFF
        bracket_h_buff = MED_SMALL_BUFF
        bracket_pair.stretch_to_fit_height(cosine_col_text.height + 2 * bracket_v_buff)
        l_bracket, r_bracket = bracket_pair.split()
        l_bracket.next_to(cosine_col_text, LEFT, bracket_h_buff)
        r_bracket.next_to(cosine_col_text, RIGHT, bracket_h_buff)
        brackets = VGroup(l_bracket, r_bracket)

        return VGroup(brackets, cosine_col_text)

    def get_cosine_row_tex(self, index):
        text = MathTex(f"C_{index}^T").scale(0.8)
        left_arrow = Arrow(RIGHT * 2, ORIGIN, stroke_width=3, max_tip_length_to_length_ratio=0.15).next_to(text, LEFT).set_color(WHITE)
        right_arrow = Arrow(ORIGIN, RIGHT * 2, stroke_width=3, max_tip_length_to_length_ratio=0.15).next_to(text, RIGHT).set_color(WHITE)
        return VGroup(left_arrow, text, right_arrow)

    def get_cosine_col_tex(self, index):
        text = MathTex(f"C_{index}").scale(0.8)
        up_arrow = Arrow(ORIGIN, UP * 1.4, stroke_width=3, max_tip_length_to_length_ratio=0.15).next_to(text, UP).set_color(WHITE)
        down_arrow = Arrow(ORIGIN, DOWN * 1.4, stroke_width=3, max_tip_length_to_length_ratio=0.15).next_to(text, DOWN).set_color(WHITE)
        return VGroup(up_arrow, text, down_arrow)

    def get_forward_transform_matrix_eq(self):
        coeff_vector_values = [f"X_{i}" for i in range(8)]
        X_coeff_vec = self.make_column_vector(coeff_vector_values)
        equals = MathTex("=")
        dct_matrix = self.get_matrix_m()
        pixel_vector_values = [f"x_{i}" for i in range(8)]
        pixel_vec = self.make_column_vector(pixel_vector_values)

        matrix_equation = VGroup(X_coeff_vec, equals, dct_matrix, pixel_vec).arrange(RIGHT)

        return matrix_equation

    def get_inverse_tranform_matrix_eq(self):
        coeff_vector_values = [f"X_{i}" for i in range(8)]
        X_coeff_vec = self.make_column_vector(coeff_vector_values)
        equals = MathTex("=")
        idct_matrix = self.get_inverse_matrix_m()
        pixel_vector_values = [f"x_{i}" for i in range(8)]
        pixel_vec = self.make_column_vector(pixel_vector_values)

        matrix_equation = VGroup(pixel_vec, equals, idct_matrix, X_coeff_vec).arrange(RIGHT)

        return matrix_equation

    def get_broad_dct_components(self, ax, graph, dots, pixel_row_mob, dct_row_pixels):
        group = VGroup(ax, graph, dots, pixel_row_mob).move_to(LEFT * 3.5 + DOWN * 0.5)

        general_vals = [f"x_{i}" for i in range(len(pixel_row_mob))]
        array_mob_symbols = self.get_gen_array_obj(general_vals, length=pixel_row_mob.width, height=pixel_row_mob.height)
        array_mob_symbols.next_to(pixel_row_mob, UP)

        forward_arrow = MathTex(r"\Rightarrow").scale(1.5).shift(RIGHT * SMALL_BUFF * 3)

        pixel_space_group = VGroup(group, array_mob_symbols)

        dct_ax = self.get_dct_axis(dct_row_pixels, -80, 80)

        dct_graph, dct_points = self.plot_row_values(dct_ax, dct_row_pixels, color=REDUCIBLE_PURPLE)

        dct_graph_components = VGroup(dct_ax, dct_graph, dct_points).move_to(RIGHT * 3.5 + DOWN * 0.5)

        vertical_lines = self.get_vertical_lines_from_points(dct_ax, dct_points)

        general_dct_vals = [f"X_{i}" for i in range(len(pixel_row_mob))]

        array_mob_dct_symbols = self.get_gen_array_obj(general_dct_vals, length=pixel_row_mob.width + 0.5, height=pixel_row_mob.height + SMALL_BUFF, color=REDUCIBLE_VIOLET)
        array_mob_dct_symbols.next_to(pixel_row_mob, UP)
        shift_amount = dct_graph_components.get_center()[0] - array_mob_dct_symbols.get_center()[0]
        array_mob_dct_symbols.shift(RIGHT * (shift_amount + 0.3))

        dct_space_group = VGroup(dct_graph_components, vertical_lines, array_mob_dct_symbols)

        return VGroup(pixel_space_group, forward_arrow, dct_space_group)

class Introduce2DDCT(DCTExperiments):
    def construct(self):
        image_mob = ImageMobject("dog").move_to(UP * 2)
        self.play(
            FadeIn(image_mob)
        )
        self.wait()

        print("Image size:", image_mob.get_pixel_array().shape)

        pixel_grid = self.add_grid(image_mob)

        block_image, block_pixel_grid, block, pixel_array_mob_2d = self.highlight_pixel_block(image_mob, 125, 125, pixel_grid)

        self.play(
            *[integer.animate.set_value(integer.get_value() - 128).move_to(integer.get_center()) for integer in pixel_array_mob_2d[1]]
        )
        self.wait()

        self.play(
            block_pixel_grid.animate.scale(0.65).shift(UP * 2.7 + RIGHT * 0.5),
            block_image.animate.scale(0.65).shift(UP * 2.7 + RIGHT * 0.5),
            pixel_array_mob_2d.animate.scale(0.65).shift(UP * 2.7 + LEFT * 0.5),
        )

        block_image, block_pixel_grid, pixel_array_mob_2d, dct_array_2d_row_col_group = self.show_data_flow_dct_2d(pixel_array_mob_2d, block[:, :, 0], block_image, block_pixel_grid)

        self.show_dct_components(dct_array_2d_row_col_group[0])

    def show_data_flow_dct_2d(self, pixel_array_mob_2d, block, block_image, block_pixel_grid):
        pixel_array_mob_2d_copy_input = pixel_array_mob_2d.copy().move_to(LEFT * 4.5)
        self.play(
            TransformFromCopy(pixel_array_mob_2d, pixel_array_mob_2d_copy_input)
        )
        self.wait()

        row_rep, col_rep = self.get_2d_dct_mob_reps(pixel_array_mob_2d)

        arrow_top_left = Arrow(pixel_array_mob_2d_copy_input.get_right(), row_rep.get_left()).set_color(REDUCIBLE_YELLOW)

        arrow_top_left_label_top = Tex(r"8 $\cross$ DCT 1D").scale(0.7).next_to(arrow_top_left, UP, buff=SMALL_BUFF)
        arrow_top_left_label_bottom = Tex("Rows").scale(0.8).next_to(arrow_top_left, DOWN, buff=SMALL_BUFF)
        self.play(
            Write(arrow_top_left),
            Write(arrow_top_left_label_top),
            Write(arrow_top_left_label_bottom),
        )
        self.wait()

        self.play(
            FadeIn(row_rep[0]),
            *[GrowFromCenter(arrow) for arrow in row_rep[1]]
        )
        self.wait()
        dct_array_2d_row_group = self.get_dct_array_2d_mob(pixel_array_mob_2d, [row_rep[0]], dct_rows(block - 128), RIGHT * 4.5)

        dct_mob_array_2d, overlay = dct_array_2d_row_group

        arrow_top_right = Arrow(row_rep.get_right(), dct_mob_array_2d.get_left()).set_color(REDUCIBLE_YELLOW)

        self.play(
            Write(arrow_top_right)
        )
        self.wait()

        self.play(
            FadeIn(dct_mob_array_2d),
            TransformFromCopy(row_rep[0], overlay[0])
        )
        self.wait()

        dct_array_2d_row_group_input = dct_array_2d_row_group.copy().move_to(LEFT * 4.5 + DOWN * 2.7)

        self.play(
            TransformFromCopy(dct_array_2d_row_group, dct_array_2d_row_group_input)
        )
        self.wait()

        col_rep.move_to(DOWN * 2.7)

        arrow_bottom_left = Arrow(dct_array_2d_row_group_input.get_right(), col_rep.get_left()).set_color(REDUCIBLE_VIOLET)

        arrow_bottom_left_label_top = Tex(r"8 $\cross$ DCT 1D").scale(0.7).next_to(arrow_bottom_left, UP, buff=SMALL_BUFF)
        arrow_bottom_left_label_bottom = Tex("Columns").scale(0.8).next_to(arrow_bottom_left, DOWN, buff=SMALL_BUFF)

        self.play(
            Write(arrow_bottom_left_label_top),
            Write(arrow_bottom_left_label_bottom),
            Write(arrow_bottom_left),
        )
        self.wait()

        self.play(
            FadeIn(col_rep[0]),
            *[GrowFromCenter(arrow) for arrow in col_rep[1]]
        )
        self.wait()

        dct_array_2d_row_col_group = self.get_dct_array_2d_mob(pixel_array_mob_2d, VGroup(row_rep[0], col_rep[0]), dct_2d(block - 128), RIGHT * 4.5 + DOWN * 2.7)

        dct_mob_array_2d_row_col, overlay_row_col = dct_array_2d_row_col_group

        arrow_bottom_right = Arrow(col_rep.get_right(), dct_array_2d_row_col_group.get_left()).set_color(REDUCIBLE_VIOLET)

        self.play(
            Write(arrow_bottom_right)
        )

        self.wait()

        self.play(
            FadeIn(dct_mob_array_2d_row_col, overlay_row_col[0]),
            TransformFromCopy(col_rep[0], overlay_row_col[1])
        )
        self.wait()
        original_dct_array_2d_row_col_group = dct_array_2d_row_col_group.copy().move_to(LEFT * 4.5)
        dct_2d_group = VGroup(
            pixel_array_mob_2d_copy_input,
            arrow_top_left,
            row_rep,
            arrow_top_right,
            dct_array_2d_row_group,
            dct_array_2d_row_group_input,
            arrow_bottom_left,
            col_rep,
            arrow_bottom_right,
            dct_array_2d_row_col_group,
            arrow_top_left_label_top,
            arrow_top_left_label_bottom,
            arrow_bottom_left_label_top,
            arrow_bottom_left_label_bottom,
        )

        surround_rect_dct_2d = SurroundingRectangle(dct_2d_group)

        dct_2d_text = Tex("DCT 2D").scale(4)

        dct_2d_text.move_to(surround_rect_dct_2d.get_center())

        self.play(
            Create(surround_rect_dct_2d)
        )

        self.play(
            dct_2d_group.animate.fade(0.8),
            Write(dct_2d_text)
        )
        self.wait()

        self.play(
            *[elem.animate.fade(1) for elem in dct_2d_group if not (elem is dct_array_2d_row_col_group)],
            FadeOut(dct_2d_text),
            FadeOut(surround_rect_dct_2d),
            Transform(dct_array_2d_row_col_group, original_dct_array_2d_row_col_group),
            FadeOut(pixel_array_mob_2d),
            FadeOut(block_image),
            FadeOut(block_pixel_grid)
        )
        self.wait()

        return block_image, block_pixel_grid, pixel_array_mob_2d, dct_array_2d_row_col_group

    def show_dct_components(self, dct_array_2d, rows=8, cols=8):
        all_dct_components = []
        for row in range(rows):
            for col in range(cols):
                pixel_grid_component = self.get_dct_component(row, col, height=0.8)
                all_dct_components.append(pixel_grid_component)
        
        all_dct_components_group = VGroup(*all_dct_components).arrange_in_grid(rows=rows, buff=SMALL_BUFF)
        all_dct_components_group.shift(RIGHT * 2)

        lines = []
        for rect, component in zip(dct_array_2d[0], all_dct_components_group):
            line = Line(rect.get_center(), component.get_center()).set_stroke(width=1)
            lines.append(line)



        self.play(
            *[Create(line) for line in lines],
            FadeIn(all_dct_components_group)
        )
        self.wait()

    def get_dct_component(self, row, col, height=2):
        dct_matrix = np.zeros((8, 8))
        if row == 0 and col == 0:
            dct_matrix[row][col] = 1016
        else:
            dct_matrix[row][col] = 500
        pixel_array = idct_2d(dct_matrix) + 128
        all_in_range = (pixel_array >= 0) & (pixel_array <= 255)
        if not all(all_in_range.flatten()):
            print("Bad array\n", pixel_array)
            raise ValueError("All elements in pixel_array must be in range [0, 255]")
        image_mob = self.get_image_vector_mob(pixel_array, height=height)
        return image_mob

    def get_image_vector_mob(self, pixel_array, height=2, num_pixels=8):
        side_length = height / num_pixels
        adjusted_row_values = np.zeros(pixel_array.shape)
        for i in range(pixel_array.shape[0]):
            for j in range(pixel_array.shape[1]):
                adjusted_row_values[i][j] = int(pixel_array[i][j])
        pixel_grid_mob = VGroup(
            *[
                Square(side_length=side_length)
                .set_stroke(color=REDUCIBLE_GREEN_LIGHTER, width=0.5)
                .set_fill(color=gray_scale_value_to_hex(value), opacity=1)
                for value in adjusted_row_values.flatten()
            ]
        ).arrange_in_grid(rows=num_pixels, buff=0)
        return pixel_grid_mob


    def get_2d_dct_mob_reps(self, pixel_array_mob_2d):
        row_representation = self.make_dct_grid(pixel_array_mob_2d.width, pixel_array_mob_2d.height, rows=True)
        col_representation = self.make_dct_grid(pixel_array_mob_2d.width, pixel_array_mob_2d.height, rows=False, color=REDUCIBLE_VIOLET)
        return row_representation, col_representation

    def get_dct_array_2d_mob(self, pixel_array_mob_2d, overlays, dct_values, position, color=REDUCIBLE_YELLOW):
        dct_mob_array_2d = self.get_2d_pixel_array_mob(dct_values, height=pixel_array_mob_2d.height)
        dct_mob_array_2d.move_to(position)
        final_overlays = VGroup()
        for overlay in overlays:
            new_overlay = overlay.copy()
            for rect in new_overlay:
                rect.set_stroke(opacity=0.4)
            new_overlay.move_to(position)
            final_overlays.add(new_overlay)
        
        group = VGroup(dct_mob_array_2d, final_overlays)

        return group

    def make_dct_grid(self, width, height, block_size=8, rows=True, color=REDUCIBLE_YELLOW):
        if rows:
            rect_width = width
            rect_height = height / block_size
        else:
            rect_width = width / block_size
            rect_height = height

        grid = VGroup(*[Rectangle(height=rect_height, width=rect_width) for _ in range(block_size)])
        if rows:
            grid.arrange(DOWN, buff=0)
            arrows = VGroup(*[DoubleArrow(start=rect.get_left(), end=rect.get_right(), stroke_width=3, tip_length=SMALL_BUFF, buff=SMALL_BUFF) for rect in grid])
        else:
            grid.arrange(RIGHT, buff=0)
            arrows = VGroup(*[DoubleArrow(start=rect.get_top(), end=rect.get_bottom(), stroke_width=3, tip_length=SMALL_BUFF, buff=SMALL_BUFF) for rect in grid])

        return VGroup(grid, arrows).set_color(color)

    def get_pixel_grid(self, image, num_pixels_in_dimension, color=WHITE):
        height_pixels = image.get_pixel_array().shape[0]
        width_pixels = image.get_pixel_array().shape[1]
        aspect_ratio = width_pixels / height_pixels
        height_single_cell = image.height / num_pixels_in_dimension
        width_single_cell = height_single_cell * aspect_ratio
        pixel_grid = VGroup(
            *[
                Rectangle(height=height_single_cell, width=width_single_cell).set_stroke(
                    color=color, width=1, opacity=0.5
                )
                for _ in range(num_pixels_in_dimension ** 2)
            ]
        )
        pixel_grid.arrange_in_grid(rows=num_pixels_in_dimension, buff=0)
        return pixel_grid
    
    def add_grid(self, image_mob):
        pixel_grid = self.get_pixel_grid(image_mob, 32)
        pixel_grid.move_to(image_mob.get_center())
        self.play(
            Create(pixel_grid)
        )
        self.wait()

        return pixel_grid

    def get_2d_pixel_array_mob(self, block, height=3, block_size=8, color=REDUCIBLE_GREEN_LIGHTER):
        array_mob_2d = VGroup(
            *[Square(side_length=height/block_size) for _ in range(block_size ** 2)]
        ).arrange_in_grid(rows=block_size, buff=0).set_color(color)

        array_text = VGroup()
        for i in range(block_size):
            for j in range(block_size):
                val = block[i][j]
                # For space constraints of rendering negative 3 digit numbers
                if val < -99:
                    val = -val
                integer = Integer(val).scale(0.35 * height / 3).move_to(array_mob_2d[i * block_size + j].get_center())
                array_text.add(integer)

        return VGroup(array_mob_2d, array_text)

    def highlight_pixel_block(self, image_mob, start_row, start_col, pixel_grid, block_size=8):
        pixel_array = image_mob.get_pixel_array()
        height_pixels = pixel_array.shape[0]
        width_pixels = pixel_array.shape[1]
        block = pixel_array[
            start_row : start_row + block_size, start_col : start_col + block_size
        ]
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
        pixel_grid_prop = (start_col * width_pixels + start_col) / (height_pixels * width_pixels)
        tiny_square_highlight = pixel_grid[int(pixel_grid_prop * len(pixel_grid))].copy().set_color(REDUCIBLE_GREEN_LIGHTER)
        tiny_square_highlight.set_stroke(width=4)
        self.play(Create(tiny_square_highlight))
        self.wait()

        block_position = LEFT * 2 + DOWN * 2
        block_image = self.get_image_mob(block, height=3.5).move_to(block_position)
        block_pixel_grid = self.get_pixel_grid(block_image, block_size).move_to(
            block_position
        )
        surround_rect = SurroundingRectangle(block_pixel_grid, buff=0).set_color(
            REDUCIBLE_GREEN_LIGHTER
        )

        pixel_array_mob_2d = self.get_2d_pixel_array_mob(block[:, :, 1], height=block_image.height)
        pixel_array_mob_2d.move_to(RIGHT * 2 + DOWN * 2)
        surround_rect_integer_grid = SurroundingRectangle(pixel_array_mob_2d, buff=0, color=REDUCIBLE_GREEN_LIGHTER)
        self.play(
            FadeIn(block_image),
            FadeIn(block_pixel_grid),
            FadeIn(pixel_array_mob_2d),
            TransformFromCopy(tiny_square_highlight, surround_rect),
            TransformFromCopy(tiny_square_highlight, surround_rect_integer_grid)
        )
        self.wait()

        shift_up = UP * 2

        self.play(
            FadeOut(surround_rect),
            FadeOut(tiny_square_highlight),
            FadeOut(surround_rect_integer_grid),
            FadeOut(image_mob),
            pixel_array_mob_2d.animate.shift(shift_up),
            block_image.animate.shift(shift_up),
            block_pixel_grid.animate.shift(shift_up),
            FadeOut(pixel_grid)
        )
        self.wait()

        return block_image, block_pixel_grid, block, pixel_array_mob_2d

class DemoJPEGWithDCT2D(ThreeDScene, ImageUtils):
    """
    TODO: Implement https://www.mathworks.com/help/vision/ref/2didct.html
    """
    def construct(self):
        image_mob = ImageMobject("dog").move_to(LEFT * 2)

        axes = ThreeDAxes(
            x_range=[0, 7], y_range=[0, 7], z_range=[0, 255], 
            x_length=5, y_length=5, z_length=5,
            tips=False,
            axis_config={"include_ticks": False},
        ).shift(IN * 1.5 + LEFT * 2)

        axes.set_color(BLACK)

        block_image_2d, pixel_grid_2d, block_2d = self.get_pixel_block(image_mob, 125, 125, height=3)
        print("Before 2D\n", block_2d[:, :, 1])
        block_image_2d = self.get_image_vector_mob(block_2d[:, :, 1], height=3)
        self.add_fixed_in_frame_mobjects(block_image_2d)
        block_image_2d.move_to(LEFT * 3.5)

        self.play(
            FadeIn(block_image_2d),
        )
        self.wait()

        block_image, block = self.get_pixel_block_for_3d(image_mob, 125, 125, height=axes.x_length)

        print('Before\n', block[:, :, 1])
        block_centered = format_block(block)
        print('After centering\n', block_centered)

        dct_block = dct_2d(block_centered)
        np.set_printoptions(suppress=True)
        print('DCT block (rounded)\n', np.round(dct_block, decimals=1))
        expected = invert_format_block(idct_2d(dct_block))
        actual = self.get_original_matrix_from_func(dct_block)
        print('Expected\n', expected)
        print('Actual\n', actual)
        assert(np.allclose(expected, actual))



        surface = Surface(
            lambda u, v: axes.c2p(*self.func(u, v, dct_block)),
            u_range=[0, 7],
            v_range=[0, 7],
            checkerboard_colors=[REDUCIBLE_PURPLE],
            fill_opacity=0.5,
            resolution=8,
            stroke_color=REDUCIBLE_YELLOW,
            stroke_width=2,
        )
        
        self.position_image_on_axes(axes, block_image)

        # self.add_fixed_in_frame_mobjects(block_image)
        # lines_to_z, dots_z = self.get_lines_and_dots(axes, block[:, :, 1], block_image)
        self.set_camera_orientation(theta=70 * DEGREES, phi=80 * DEGREES)
        self.add(axes, block_image, surface)
        self.wait()

        number_line = self.initialize_slider(block_image, block[:, :, 1], surface)
        block_image = self.animate_slider(number_line, axes, block_image, dct_block, block[:, :, 1], surface, block_image_2d)

        # self.begin_ambient_camera_rotation(rate=0.1)
        # self.wait(5)

    def initialize_slider(self, block_image, block, surface):
        number_line = NumberLine(
            x_range=[0, 64, 8],
            length=10,
            color=REDUCIBLE_VIOLET,
            include_numbers=True,
            label_direction=UP,
        )
        self.add_fixed_in_frame_mobjects(number_line)
        number_line.move_to(DOWN * 3)

        self.play(
            FadeIn(number_line)
        )
        self.wait()
        return number_line

    def get_image_vector_mob(self, pixel_array, height=2, num_pixels=8, color=WHITE):
        side_length = height / num_pixels
        adjusted_row_values = np.zeros(pixel_array.shape)
        for i in range(pixel_array.shape[0]):
            for j in range(pixel_array.shape[1]):
                adjusted_row_values[i][j] = int(pixel_array[i][j])
        pixel_grid_mob = VGroup(
            *[
                Square(side_length=side_length)
                .set_stroke(color=color, width=0.5)
                .set_fill(color=gray_scale_value_to_hex(value), opacity=1)
                for value in adjusted_row_values.flatten()
            ]
        ).arrange_in_grid(rows=num_pixels, buff=0)
        return pixel_grid_mob

    def animate_slider(self, number_line, axes, block_image, dct_block, original_block, surface, block_image_2d):
        tick = Triangle().scale(0.2).set_color(REDUCIBLE_YELLOW)
        tick.set_fill(color=REDUCIBLE_YELLOW, opacity=1)
        self.add_fixed_in_frame_mobjects(tick)

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
        # surface_pos = RIGHT *
        def get_new_block():
            new_partial_block = self.get_partial_block(dct_block, tracker.get_value())
            print(f'Partial block - {tracker.get_value()} components')
            print('MSE', np.mean((new_partial_block - original_block) ** 2), '\n')

            new_partial_block_image = self.get_block_image_for_3d(new_partial_block, height=axes.x_length)            
            self.position_image_on_axes(axes, new_partial_block_image)
            return new_partial_block_image

        def get_new_surface():
            new_partial_block_dct = self.get_partial_dct_block(dct_block, tracker.get_value())
            # print('Generating surface of block:\n', new_partial_block_dct)
            new_surface = Surface(
                lambda u, v: axes.c2p(*self.func(u, v, new_partial_block_dct)),
                u_range=[0, 7],
                v_range=[0, 7],
                checkerboard_colors=[REDUCIBLE_PURPLE],
                fill_opacity=0.5,
                resolution=8,
                stroke_color=REDUCIBLE_YELLOW,
                stroke_width=2,
            )
            return new_surface
        
        def get_new_block_2d():
            new_partial_block = self.get_partial_block(dct_block, tracker.get_value())

            new_partial_block_image = self.get_image_vector_mob(new_partial_block, height=3)
            
            # TODO comment out this line for the smooth transition
            self.add_fixed_in_frame_mobjects(new_partial_block_image)
            new_partial_block_image.move_to(block_image_2d.get_center())
            return new_partial_block_image

        partial_block_image_2d = always_redraw(get_new_block_2d)
        # partial_pixel_grid_2d = self.get_pixel_grid(
        #     partial_block_image_2d, partial_block_2d.shape[0]
        # )
        # partial_pixel_grid_2d.move_to(par)

        partial_block = self.get_partial_block(dct_block, tracker.get_value())
        partial_block_image = always_redraw(get_new_block)
        partial_block_surface = always_redraw(get_new_surface)
        self.play(
            ReplacementTransform(block_image, partial_block_image),
            ReplacementTransform(surface, partial_block_surface),
            ReplacementTransform(block_image_2d, partial_block_image_2d),
        )
        self.wait()
        
       
        self.play(
            tracker.animate.set_value(64),
            run_time=10,
            rate_func=linear,
        )

        self.wait()
        return partial_block_image


    # 2D IDCT Function
    def func(self, x, y, dct_matrix):
        M, N = 8, 8
        def C(m):
            if m == 0:
                return 1 / np.sqrt(2)
            return 1

        result = 0
        norm_factor = 2 / (np.sqrt(M * N))
        for m in range(M):
            for n in range(N):
                cos_mx = np.cos((2 * x + 1) * m * np.pi / (2 * M))
                cos_ny = np.cos((2 * y + 1) * n * np.pi / (2 * N))
                result += C(m) * C(n) * dct_matrix[m][n] * cos_mx * cos_ny
        return np.array([x, y, norm_factor * result + 128])

    def get_original_matrix_from_func(self, dct_matrix):
        result = np.zeros((8, 8))
        for x in range(result.shape[0]):
            for y in range(result.shape[1]):
                result[x][y] = self.func(x, y, dct_matrix)[2]
        return result

    def position_image_on_axes(self, axes, block_image):
        block_image.move_to(axes.c2p(*[np.mean(axes.x_range) + 0.75, np.mean(axes.y_range) + 0.75, 0]))
        block_image.flip(RIGHT)

    def get_pixel_block_for_3d(self, image_mob, start_row, start_col, block_size=8, height=2):
        pixel_array = image_mob.get_pixel_array()
        block = pixel_array[start_row:start_row+block_size, start_col:start_col+block_size]

        block_image = self.get_block_image_for_3d(block[:, :, 1], block_size=block_size, height=height)

        return block_image, block

    def get_block_image_for_3d(self, block, block_size=8, height=2):
        # this block_image seems to break in 3D scenes, so using the pixel_grid itself
        # as a proxy for the image
        block_image = self.get_image_mob(block, height=height)
        pixel_grid = self.get_pixel_grid(block_image, block_size)

        for i in range(block.shape[0]):
            for j in range(block.shape[1]):
                index = two_d_to_1d_index(i, j)
                pixel_grid[index].set_fill(color=gray_scale_value_to_hex(block[i][j]), opacity=1)

        return pixel_grid

    def get_partial_block(self, dct_block, num_components):
        """
        @param: dct_block - dct coefficients of the block
        @param: num_components - float - number of dct components to
        include in partial block
        @return: pixel_array of partial block with num_components of DCT included
        """
        dct_matrix = self.get_partial_dct_block(dct_block, num_components)

        pixel_array = idct_2d(dct_matrix)
        return invert_format_block(pixel_array)

    def get_partial_dct_block(self, dct_block, num_components):
        """
        @param: dct_block - dct coefficients of the block
        @param: num_components - float - number of dct components to
        include in partial block
        @return: partial accumulated dct block containing a combination of num_components
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

        return dct_matrix

    def get_lines_and_dots(self, axes, block, block_image, color=REDUCIBLE_GREEN_DARKER):
        lines = VGroup()
        dots = VGroup()
        for i in range(block.shape[0]):
            for j in range(block.shape[1]):
                index = two_d_to_1d_index(i, j)
                start = block_image[index].get_center()
                end = axes.c2p(*[i, j, block[i][j]])
                dot = Dot().move_to(end).set_color(color=color)
                dots.add(dot)
                line = Line(start, end).set_stroke(color=color, width=1)
                lines.add(line)

        return lines, dots

    def get_pixel_block(self, image_mob, start_row, start_col, block_size=8, height=2):
        pixel_array = image_mob.get_pixel_array()
        block = pixel_array[
            start_row : start_row + block_size, start_col : start_col + block_size
        ]

        block_image = self.get_image_mob(block, height=height)
        pixel_grid = self.get_pixel_grid(block_image, block_size)

        return block_image, pixel_grid, block

class HeatMapExperiments(Introduce2DDCT):
    def construct(self):
        image_mob = ImageMobject("dog").move_to(UP * 2.5)
        self.play(
            FadeIn(image_mob)
        )
        self.wait()

        print("Image size:", image_mob.get_pixel_array().shape)

        pixel_grid = self.add_grid(image_mob)

        block_image, block_pixel_grid, block, pixel_array_mob_2d, tiny_square_highlight = self.highlight_pixel_block(image_mob, 125, 125, pixel_grid)

        block_single_channel = block[:, :, 0]

        self.show_changing_blocks_with_dct(image_mob, block_image, block_pixel_grid, pixel_array_mob_2d, block_single_channel, tiny_square_highlight, pixel_grid)

    def show_changing_blocks_with_dct(self, image_mob, block_image, block_pixel_grid, pixel_array_mob_2d, block_single_channel, tiny_square_highlight, pixel_grid):
        dct_array_2d_mob, dct_heat_map = self.get_array_and_heat_map(block_image, pixel_array_mob_2d, block_single_channel)
        pixel_array = image_mob.get_pixel_array()
        height_pixels = pixel_array.shape[0]
        width_pixels = pixel_array.shape[1]

        self.play(
            FadeIn(dct_array_2d_mob),
            FadeIn(dct_heat_map)
        )
        self.wait()
        for _ in range(20):
            start_row, start_col = np.random.randint(0, height_pixels), np.random.randint(0, width_pixels)
            print(start_row, start_col)
            new_block_image, new_block_pixel_grid, new_block = self.get_pixel_block(image_mob, start_row, start_col, height=block_image.height)
            new_block_image.move_to(block_image.get_center())
            new_block_pixel_grid.move_to(block_pixel_grid.get_center())
            
            new_tiny_square_highlight = self.get_tiny_square(start_row, start_col, image_mob, pixel_array, pixel_grid)
            tiny_square_highlight.become(new_tiny_square_highlight)

            new_pixel_array_mob_2d = self.get_2d_pixel_array_mob(new_block[:, :, 0], height=pixel_array_mob_2d.height)
            
            new_pixel_array_mob_2d.move_to(pixel_array_mob_2d.get_center())
            
            block_image.become(new_block_image)
            block_pixel_grid.become(new_block_pixel_grid)
            pixel_array_mob_2d.become(new_pixel_array_mob_2d)

            new_block_single_channel = new_block[:, :, 0]
            new_dct_array_2d_mob, new_dct_heat_map = self.get_array_and_heat_map(new_block_image, new_pixel_array_mob_2d, new_block_single_channel)

            dct_array_2d_mob.become(new_dct_array_2d_mob)
            dct_heat_map.become(new_dct_heat_map)
            self.wait()
        # self.remove(block_image, block_pixel_grid, pixel_array_mob_2d)
        # self.add(new_block_image, new_pixel_grid, new_pixel_array_mob_2d)
        # self.wait()

    def get_pixel_block(self, image_mob, start_row, start_col, block_size=8, height=2):
        pixel_array = image_mob.get_pixel_array()
        block = pixel_array[
            start_row : start_row + block_size, start_col : start_col + block_size
        ]

        block_image = self.get_image_mob(block, height=height)
        pixel_grid = self.get_pixel_grid(block_image, block_size)

        return block_image, pixel_grid, block
    
    def get_array_and_heat_map(self, block_image, pixel_array_mob_2d, block_single_channel):
        formatted_block = format_block(block_single_channel)
        dct_values = dct_2d(formatted_block)
        dct_array_2d_mob = self.get_2d_pixel_array_mob(dct_values, height=pixel_array_mob_2d.height, color=REDUCIBLE_VIOLET)
        dct_array_2d_mob.move_to(RIGHT * 3.5 + UP * 2)

        dct_heat_map = self.get_heat_map(block_image, dct_values)
        dct_heat_map.next_to(dct_array_2d_mob, DOWN, aligned_edge=LEFT)
        shift_down = (-2 - dct_heat_map.get_center()[1])
        dct_heat_map.shift(UP * shift_down)
        
        return dct_array_2d_mob, dct_heat_map

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
            square.set_fill(
                color=interpolate_color(min_color, max_color, alpha), opacity=1
            )

        scale = Line(pixel_grid_dct.get_top(), pixel_grid_dct.get_bottom())
        scale.set_stroke(width=10).set_color(color=[min_color, max_color])
        integer_scale = 0.5
        top_value = Integer(round(max_dct_coeff)).scale(integer_scale)
        top_value.next_to(scale, RIGHT, aligned_edge=UP)
        bottom_value = Integer(0).scale(integer_scale)
        bottom_value.next_to(scale, RIGHT, aligned_edge=DOWN)

        heat_map_scale = VGroup(scale, top_value, bottom_value)

        return VGroup(pixel_grid_dct, heat_map_scale).arrange(RIGHT)

    def get_tiny_square(self, start_row, start_col, image_mob, pixel_array, pixel_grid, block_size=8):
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
        tiny_square_highlight.set_color(REDUCIBLE_GREEN_LIGHTER).move_to(highlight_position)
        return tiny_square_highlight
        # height_pixels = pixel_array.shape[0]
        # width_pixels = pixel_array.shape[1]
        # print(width_pixels, height_pixels)
        # pixel_grid_prop = (start_row * width_pixels + start_col) / (height_pixels * width_pixels)
        # tiny_square_highlight = pixel_grid[int(pixel_grid_prop * len(pixel_grid))].copy().set_color(REDUCIBLE_GREEN_LIGHTER)
        # tiny_square_highlight.set_stroke(width=4)
        # return tiny_square_highlight

    def highlight_pixel_block(self, image_mob, start_row, start_col, pixel_grid, block_size=8):
        pixel_array = image_mob.get_pixel_array()
        height_pixels = pixel_array.shape[0]
        width_pixels = pixel_array.shape[1]
        block = pixel_array[
            start_row : start_row + block_size, start_col : start_col + block_size
        ]
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
        tiny_square_highlight.set_color(REDUCIBLE_GREEN_LIGHTER).move_to(highlight_position)
        self.play(Create(tiny_square_highlight))
        self.wait()

        block_position = LEFT * 2 + DOWN * 1
        block_image = self.get_image_mob(block, height=3.5).move_to(block_position)
        block_pixel_grid = self.get_pixel_grid(block_image, block_size).move_to(
            block_position
        )
        surround_rect = SurroundingRectangle(block_pixel_grid, buff=0).set_color(
            REDUCIBLE_GREEN_LIGHTER
        )

        pixel_array_mob_2d = self.get_2d_pixel_array_mob(block[:, :, 1], height=block_image.height)
        pixel_array_mob_2d.move_to(RIGHT * 2 + DOWN * 1)
        surround_rect_integer_grid = SurroundingRectangle(pixel_array_mob_2d, buff=0, color=REDUCIBLE_GREEN_LIGHTER)
        self.play(
            FadeIn(block_image),
            FadeIn(block_pixel_grid),
            FadeIn(pixel_array_mob_2d),
            TransformFromCopy(tiny_square_highlight, surround_rect),
            TransformFromCopy(tiny_square_highlight, surround_rect_integer_grid)
        )
        self.wait()

        self.play(
            pixel_array_mob_2d.animate.scale(0.8).move_to(LEFT * 3.5 + UP * 2),
            block_image.animate.scale(0.8).move_to(LEFT * 3.5 + DOWN * 2),
            block_pixel_grid.animate.scale(0.8).move_to(LEFT * 3.5 + DOWN * 2),
            surround_rect.animate.scale(0.8).move_to(LEFT * 3.5 + DOWN * 2),
            surround_rect_integer_grid.animate.scale(0.8).move_to(LEFT * 3.5 + UP * 2),
            image_mob.animate.move_to(ORIGIN),
            pixel_grid.animate.move_to(ORIGIN),
            tiny_square_highlight.animate.shift(DOWN * 2.5)
        )
        self.wait()

        self.add_foreground_mobject(surround_rect)

        return block_image, block_pixel_grid, block, pixel_array_mob_2d, tiny_square_highlight
# Quick test of gray_scale_value_to_hex
class TestHexToGrayScale(Scene):
    def construct(self):
        for i in range(256):
            dot = Dot().set_color(gray_scale_value_to_hex(i))
            self.add(dot)
            self.wait()
            self.remove(dot)

def two_d_to_1d_index(i, j, block_size=8):
    return j * block_size + i

def gray_scale_value_to_hex(value):
    assert value >= 0 and value <= 255, f'Invalid value {value}'
    integer_value = int(round(value))
    hex_string = hex(integer_value).split("x")[-1]
    if integer_value < 16:
        hex_string = "0" + hex_string
    return "#" + hex_string * 3


def make_lut_u():
    return np.array([[[i, 255 - i, 0] for i in range(256)]], dtype=np.uint8)


def make_lut_v():
    return np.array([[[0, 255 - i, i] for i in range(256)]], dtype=np.uint8)


def dct1D_manual(f, N):
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

    x = np.arange(0, N + 0.001, 0.001)
    y = f(x)
    print(np.sum(y))
    print([f(i) for i in range(8)])

    plt.plot(x, y)
    plt.show()

def compute_dot_product(f, g, sample_points):
    a = np.array([f(p) for p in sample_points])
    b = np.array([g(p) for p in sample_points])
    return np.dot(a, b)


def g(i):
    return np.cos((2 * i + 1) * 5 * np.pi / 16)


def h(i):
    return np.cos((2 * i + 1) * 5 * np.pi / 16) * np.cos((2 * i + 1) * 3 * np.pi / 16)

def func(j, k):
    def dot_product_func(n):
        return np.cos((2 * n + 1) * j * np.pi / 16) * np.cos((2 * n + 1) * k * np.pi / 16)
    return dot_product_func

def get_dct_elem(i, j, N):
    return np.cos(j * (2 * i + 1) * np.pi / (2 * N))


def get_dct_matrix(N):
    matrix = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            matrix[j][i] = get_dct_elem(i, j, N)

    return matrix


def get_dot_product_matrix(N):
    dct_matrix = get_dct_matrix(N)
    dot_product_matrix = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            value = np.dot(dct_matrix[i], dct_matrix[j])
            if np.isclose(value, 0):
                dot_product_matrix[i][j] = 0
            else:
                dot_product_matrix[i][j] = value

    return dot_product_matrix


def format_block(block):
    if len(block.shape) < 3:
        return block.astype(float) - 128
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

def dct_cols(block):
    return fftpack.dct(block.T, norm="ortho").T

def dct_rows(block):
    return fftpack.dct(block, norm="ortho")

def dct_1d(row):
    return fftpack.dct(row, norm="ortho")


def idct_1d(row):
    return fftpack.idct(row, norm="ortho")


def dct_2d(block):
    return fftpack.dct(fftpack.dct(block.T, norm="ortho").T, norm="ortho")


def idct_2d(block):
    return fftpack.idct(fftpack.idct(block.T, norm="ortho").T, norm="ortho")


def quantize(block):
    quant_table = get_quantization_table()
    return (block / quant_table).round().astype(np.int32)


def get_quantization_table():
    quant_table = np.array(
        [
            [16, 11, 10, 16, 24, 40, 51, 61],
            [12, 12, 14, 19, 26, 58, 60, 55],
            [14, 13, 16, 24, 40, 57, 69, 56],
            [14, 17, 22, 29, 51, 87, 80, 62],
            [18, 22, 37, 56, 68, 109, 103, 77],
            [24, 35, 55, 64, 81, 104, 113, 92],
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
    """zigzag rows"""

    def compare(xy):
        x, y = xy
        return (x + y, -y if (x + y) % 2 else y)

    xs = range(n)
    return {
        n: index
        for n, index in enumerate(sorted(((x, y) for x in xs for y in xs), key=compare))
    }
