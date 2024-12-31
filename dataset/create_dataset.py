import chess
import chess.pgn
import csv
import random

def generate_random_chess_position():
    board = chess.Board()
    board.clear()  # Start with an empty board

    def place_pieces(num_pieces, color, exclude_squares, is_black=False):
        positions = set()
        while len(positions) < num_pieces:
            square = random.choice(chess.SQUARES)
            if square not in exclude_squares and board.piece_at(square) is None:
                if is_black:
                    liste_piece=[chess.PAWN, chess.PAWN, chess.KNIGHT, chess.KNIGHT, chess.BISHOP, chess.BISHOP, chess.ROOK, chess.ROOK, chess.QUEEN]
                else:
                    liste_piece=[chess.PAWN, chess.PAWN, chess.PAWN, chess.KNIGHT, chess.KNIGHT, chess.BISHOP, chess.BISHOP, chess.ROOK, chess.ROOK, chess.QUEEN]
                piece_type = random.choice(liste_piece)
                piece = chess.Piece(piece_type, color)
                board.set_piece_at(square, piece)
                positions.add(square)
        return positions

    # Place kings first to ensure they exist
    white_king_square = random.choice(chess.SQUARES)
    board.set_piece_at(white_king_square, chess.Piece(chess.KING, chess.WHITE))

    black_king_square = random.choice([sq for sq in chess.SQUARES if sq != white_king_square])
    board.set_piece_at(black_king_square, chess.Piece(chess.KING, chess.BLACK))

    # Ensure other pieces do not overwrite kings
    exclude_squares = {white_king_square, black_king_square}

    # Place white pieces (1-8)
    white_pieces = random.randint(1, 8)
    place_pieces(white_pieces, chess.WHITE, exclude_squares)

    # Place black pieces (1-8)
    black_pieces = random.randint(1, 8)
    place_pieces(black_pieces, chess.BLACK, exclude_squares, is_black=True)

    return board

def can_capture_king(board, color):
    king_square = board.king(not color)  # Opponent's king
    if king_square is None:
        return False

    attackers = board.attackers(color, king_square)
    return len(attackers) > 0

def generate_dataset(filename, num_samples):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["position", "evaluation"])

        num_it=0
        while num_it<num_samples:    
            board = generate_random_chess_position()

            # Determine the evaluation based on the player to move
            if board.turn == chess.WHITE:
                evaluation = 0
                if can_capture_king(board, chess.WHITE):
                    evaluation = +32000
                # elif board.is_checkmate():
                #     evaluation = -32000
            else:
                evaluation = 0
                if can_capture_king(board, chess.BLACK):
                    evaluation = -32000
                # elif board.is_checkmate():
                #     evaluation = +32000
            if evaluation>=0: 
                if True: #random.random()>.85:
                    writer.writerow([board.fen(), evaluation])
                    num_it+=1
            else:
                writer.writerow([board.fen(), evaluation])
                num_it+=1
                    

# Example usage:
generate_dataset("capture_king_viewable.csv", 10000)
print("toto")
generate_dataset("capture_king.csv", 1000000)
