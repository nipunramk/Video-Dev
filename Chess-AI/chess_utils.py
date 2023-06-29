from manim import *
import chess

DEFAULT_DARK_SQUARE_COLOR = "#8B4513"
DEFAULT_LIGHT_SQUARE_COLOR = "#F0D9B5"

config["assets_dir"] = "assets/Chess"

color_to_name = {chess.WHITE: "white", chess.BLACK: "black"}


def create_board_gui(
    white_color=DEFAULT_LIGHT_SQUARE_COLOR,
    black_color=DEFAULT_DARK_SQUARE_COLOR,
    board_side_length=5,
):
    square_side_length = board_side_length / 8
    board = VGroup(
        *[Square(side_length=square_side_length) for _ in range(64)]
    ).arrange_in_grid(rows=8, buff=0)
    even_color = black_color
    odd_color = white_color
    for i, square in enumerate(board):
        if i > 0 and i % 8 == 0:
            even_color, odd_color = odd_color, even_color
        if i % 2 == 0:
            square.set_fill(color=even_color, opacity=1).set_color(even_color)
        else:
            square.set_fill(color=odd_color, opacity=1).set_color(odd_color)
    return board.rotate_about_origin(PI).flip(UP)


def get_board_from_position(board, board_gui):
    squares_to_icons = {}
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            print(square, piece.symbol(), piece.color)
            piece_icon = get_piece_icon(piece)
            piece_icon.move_to(board_gui[square]).scale_to_fit_height(
                board_gui[square].height - SMALL_BUFF * 1
            )
            squares_to_icons[square] = piece_icon
    return squares_to_icons


def get_piece_icon(piece):
    piece_name = chess.piece_name(piece.piece_type)
    filename = color_to_name[piece.color] + "_" + piece_name
    return ImageMobject(filename)
