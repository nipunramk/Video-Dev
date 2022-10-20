import sys

### THIS IS A WORKAROUND FOR NOW
### IT REQUIRES RUNNING MANIM FROM INSIDE DIRECTORY VIDEO-DEV
sys.path.insert(1, "common/")

from manim import *
from dft_utils import *
from reducible_colors import *


class BeginIntroSampling_002(Scene):
    def construct(self):
        x_min = 0
        x_max = 2 * PI
        frequency = 7
        num_points = 7

        cosine_signal = get_cosine_func(freq=frequency)
        axes, signal_mob = plot_time_domain(cosine_signal, t_max=x_max - PI / 4)
        sampled_dots = get_sampled_dots(
            signal_mob, axes, x_max=x_max, num_points=num_points
        ).set_color(REDUCIBLE_VIOLET)

        freq_txt = (
            Text("ƒ = " + str(frequency), font=REDUCIBLE_MONO)
            .next_to(signal_mob, DOWN, aligned_edge=LEFT, buff=1)
            .scale(0.6)
        )
        point_n_txt = (
            Text("N = " + str(num_points), font=REDUCIBLE_MONO)
            .next_to(signal_mob, DOWN, aligned_edge=RIGHT, buff=1)
            .scale(0.6)
        )

        self.play(Write(signal_mob), FadeIn(freq_txt))
        self.play(Write(sampled_dots))

        self.wait()

        # aliasing
        # show how any multiple of our original freq is an alias
        multiples = 5
        original_signal_mob = signal_mob.copy()
        original_freq_txt = freq_txt.copy()
        aliasing_txt = (
            Text("Aliasing", font=REDUCIBLE_FONT, weight=BOLD)
            .set_stroke(width=8, background=True)
            .to_corner(UR)
        )
        self.play(FadeIn(aliasing_txt, shift=LEFT))
        for m in range(2, multiples):
            cosine_signal_aa = get_cosine_func(freq=frequency * m)
            axes_aa, signal_mob_aa = plot_time_domain(
                cosine_signal_aa, t_max=x_max - PI / 4
            )
            new_freq_txt = (
                Text(
                    "ƒ = " + str(frequency * m),
                    font=REDUCIBLE_MONO,
                )
                .scale(0.6)
                .move_to(freq_txt)
            )

            self.play(
                Transform(signal_mob, signal_mob_aa),
                FadeTransform(freq_txt, new_freq_txt),
                run_time=2,
            )
            freq_txt = new_freq_txt

        self.play(FadeOut(aliasing_txt, shift=RIGHT))
        self.wait()
        # reset
        self.play(
            Transform(signal_mob, original_signal_mob),
            FadeTransform(freq_txt, original_freq_txt),
        )
        freq_txt = original_freq_txt
        self.wait()

        # double sampling
        double_sampling = get_sampled_dots(
            signal_mob, axes, x_max=x_max, num_points=frequency * 2
        ).set_color(REDUCIBLE_VIOLET)

        point_n_txt = (
            Text("N = " + str(num_points), font=REDUCIBLE_MONO)
            .next_to(signal_mob, DOWN, aligned_edge=RIGHT, buff=1)
            .scale(0.6)
        )
        point_2n_txt = (
            Text("N = " + str(num_points * 2), font=REDUCIBLE_MONO)
            .next_to(signal_mob, DOWN, aligned_edge=RIGHT, buff=1)
            .scale(0.6)
        )

        self.play(FadeIn(point_n_txt, shift=LEFT))
        self.wait()
        self.play(
            Transform(sampled_dots, double_sampling[:-1]),
            FadeTransform(point_n_txt, point_2n_txt),
        )
        point_n_txt = point_2n_txt
        self.wait()

        # shannon sampling
        shannon_sampling = get_sampled_dots(
            signal_mob, axes, x_max=x_max, num_points=frequency * 2 + 1
        )
        point_shannon_txt = (
            Text("N = " + str(num_points * 2 + 1), font=REDUCIBLE_MONO)
            .next_to(signal_mob, DOWN, aligned_edge=RIGHT, buff=1)
            .scale(0.6)
        )
        self.play(
            Transform(sampled_dots, shannon_sampling[:-1]),
            FadeTransform(point_n_txt, point_shannon_txt),
        )
