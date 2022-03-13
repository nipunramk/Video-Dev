from manim import *

np.random.seed(1)
config["assets_dir"] = "assets"


class TestBarChart(Scene):
    def construct(self):
        encode_ms = SVGMobject("encode_ms.svg").scale(3)
        decode_ms = SVGMobject("decode_ms.svg").scale(3)
        comp_rate = SVGMobject("comp_rate.svg").scale(3)

        self.play(Write(encode_ms))

        self.wait()

        self.play(FadeOut(encode_ms))
        self.play(Write(decode_ms))

        self.wait()

        self.play(FadeOut(decode_ms))
        self.play(Write(comp_rate))

        self.wait()
