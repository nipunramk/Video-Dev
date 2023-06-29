import chess
from manim import *
from chess_utils import *

config["assets_dir"] = "assets/Chess"


class BoardRender(Scene):
    def construct(self):
        board_gui = create_board_gui()
        self.play(FadeIn(board_gui))
        self.wait()

        board = chess.Board()
        square_to_icons = get_board_from_position(board, board_gui)
        self.play(*[FadeIn(icon) for icon in square_to_icons.values()])
        self.wait()
