#!/usr/bin/env python3
import sys
import chess
import math
import signal
import random

from chess.polyglot import zobrist_hash  # fast TT key?


INF = 10_000


# Forced-capture move generation

def generate_forced_capture_moves(board: chess.Board):
    moves = list(board.legal_moves)
    captures = [m for m in moves if board.is_capture(m)]
    return captures if captures else moves


# Evaluation

PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}

# Piece-Square Tables, mostly for midgame eval rightnow
# Table from https://github.com/dimdano/numbfish/blob/main/numbfish.py
PST = {
    chess.PAWN: [
        0,   0,   0,   0,   0,   0,   0,   0,
        78,  83,  86,  73, 102,  82,  85,  90,
        7,  29,  21,  44,  40,  31,  44,   7,
       -17,  16,  -2,  15,  14,   0,  15, -13,
       -26,   3,  10,   9,   6,   1,   0, -23,
       -22,   9,   5, -11, -10,  -2,   3, -19,
       -31,   8,  -7, -37, -36, -14,   3, -31,
         0,   0,   0,   0,   0,   0,   0,   0
    ],
    chess.KNIGHT: [
       -66, -53, -75, -75, -10, -55, -58, -70,
        -3,  -6, 100, -36,   4,  62,  -4, -14,
        10,  67,   1,  74,  73,  27,  62,  -2,
        24,  24,  45,  37,  33,  41,  25,  17,
        -1,   5,  31,  21,  22,  35,   2,   0,
       -18,  10,  13,  22,  18,  15,  11, -14,
       -23, -15,   2,   0,   2,   0, -23, -20,
       -74, -23, -26, -24, -19, -35, -22, -69
    ],
    chess.BISHOP: [
        59, -78, -82, -76, -23,-107, -37, -50,
       -11,  20,  35, -42, -39,  31,   2, -22,
        -9,  39, -32,  41,  52, -10,  28, -14,
        25,  17,  20,  34,  26,  25,  15,  10,
        13,  10,  17,  23,  17,  16,   0,   7,
        14,  25,  24,  15,   8,  25,  20,  15,
        19,  20,  11,   6,   7,   6,  20,  16,
        -7,   2, -15, -12, -14, -15, -10, -10
    ],
    chess.ROOK: [
        35,  29,  33,   4,  37,  33,  56,  50,
        55,  29,  56,  67,  55,  62,  34,  60,
        19,  35,  28,  33,  45,  27,  25,  15,
         0,   5,  16,  13,  18,  -4,  -9,  -6,
       -28, -35, -16, -21, -13, -29, -46, -30,
       -42, -28, -42, -25, -25, -35, -26, -46,
       -53, -38, -31, -26, -29, -43, -44, -53,
       -30, -24, -18,   5,  -2, -18, -31, -32
    ],
    chess.QUEEN: [
         6,   1,  -8,-104,  69,  24,  88,  26,
        14,  32,  60, -10,  20,  76,  57,  24,
        -2,  43,  32,  60,  72,  63,  43,   2,
         1, -16,  22,  17,  25,  20, -13,  -6,
       -14, -15,  -2,  -5,  -1, -10, -20, -22,
       -30,  -6, -13, -11, -16, -11, -16, -27,
       -36, -18,   0, -19, -15, -15, -21, -38,
       -39, -30, -31, -13, -31, -36, -34, -42
    ],
    chess.KING: [
         4,  54,  47, -99, -99,  60,  83, -62,
       -32,  10,  55,  56,  56,  55,  10,   3,
       -62,  12, -57,  44, -67,  28,  37, -31,
       -55,  50,  11,  -4, -19,  13,   0, -49,
       -55, -43, -52, -28, -51, -47,  -8, -50,
       -47, -42, -43, -79, -64, -32, -29, -32,
        -4,   3, -14, -50, -57, -18,  13,   4,
        17,  30,  -3, -14,   6,  -1,  40,  18
    ],
}


def evaluate(board: chess.Board) -> int:
    """
    Evaluation:
    - Material
    - Piece-square tables
    - Pawn structure
    - King safety / activity (very basic)
    - Bishop pair
    - Endgame with extra passed-pawn bonuses
    """
    if board.is_game_over():
        result = board.result(claim_draw=True)
        if result == "1-0":
            return INF if board.turn == chess.WHITE else -INF
        elif result == "0-1":
            return INF if board.turn == chess.BLACK else -INF
        return 0

    score = 0

    # Material
    for piece_type, val in PIECE_VALUES.items():
        score += val * len(board.pieces(piece_type, chess.WHITE))
        score -= val * len(board.pieces(piece_type, chess.BLACK))

    # Detect endgame (simplified)
    total_material = sum(
        PIECE_VALUES[p] * (len(board.pieces(p, chess.WHITE)) + len(board.pieces(p, chess.BLACK)))
        for p in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]
    )
    endgame = total_material <= 1600

    # Piece-Square Tables (not very effective right now though)
    for piece_type in PST:
        for sq in board.pieces(piece_type, chess.WHITE):
            score += PST[piece_type][sq]
        for sq in board.pieces(piece_type, chess.BLACK):
            score -= PST[piece_type][chess.square_mirror(sq)]


    # Pawn structure
    for color in [chess.WHITE, chess.BLACK]:
        pawns = board.pieces(chess.PAWN, color)
        files = [chess.square_file(p) for p in pawns]

        # doubled pawns
        doubled = len(pawns) - len(set(files))
        if doubled > 0:
            score += (10 if color == chess.BLACK else -10) * doubled

        # isolated pawns
        for p in pawns:
            f = chess.square_file(p)
            neighbors = [ff for ff in [f - 1, f + 1] if 0 <= ff <= 7]
            if not any(file in neighbors for file in files):
                score += 15 if color == chess.BLACK else -15

        # passed pawns bonus
        for p in pawns:
            f = chess.square_file(p)
            r = chess.square_rank(p)
            is_passed = not any(
                board.pieces(chess.PAWN, 1 - color) & chess.BB_SQUARES[sq]
                for sq in chess.SQUARES
                if chess.square_file(sq) in [f - 1, f, f + 1]
                and (
                    (color == chess.WHITE and chess.square_rank(sq) > r)
                    or (color == chess.BLACK and chess.square_rank(sq) < r)
                )
            )
            if is_passed:
                bonus = 20 + (r if color == chess.WHITE else (7 - r)) * 10
                score += bonus if color == chess.WHITE else -bonus

    # King approach
    if endgame:
        wk_sq = list(board.pieces(chess.KING, chess.WHITE))[0]
        bk_sq = list(board.pieces(chess.KING, chess.BLACK))[0]
        wk_rank, wk_file = chess.square_rank(wk_sq), chess.square_file(wk_sq)
        bk_rank, bk_file = chess.square_rank(bk_sq), chess.square_file(bk_sq)

        dist = abs(wk_rank - bk_rank) + abs(wk_file - bk_file)
        material_adv = sum(
            PIECE_VALUES[p] * (len(board.pieces(p, chess.WHITE)) - len(board.pieces(p, chess.BLACK)))
            for p in PIECE_VALUES
        )
        scale = min(1.0, max(0.2, material_adv / 1000))
        approach_bonus = ((8 - dist) ** 2) * 15 * scale
        score += approach_bonus

        def king_freedom_space(board: chess.Board, color: chess.Color) -> int:
            # Return number of squares opponent king can eventually reach
            from collections import deque

            king_sq = list(board.pieces(chess.KING, color))[0]
            visited = set()
            queue = deque([king_sq])
            own_color = not color

            while queue:
                sq = queue.popleft()
                if sq in visited:
                    continue
                visited.add(sq)

                for move_sq in chess.SquareSet(chess.BB_KING_ATTACKS[sq]):
                    piece = board.piece_at(move_sq)
                    if piece and piece.color == own_color:
                        continue
                    if move_sq not in visited:
                        queue.append(move_sq)
            return len(visited)

        opp_space = king_freedom_space(board, chess.BLACK)
        mobility_bonus = max(0, 64 - opp_space) * 10 * scale
        score += mobility_bonus

        edge_bonus = 0
        for coord in [bk_rank, bk_file]:
            if coord == 0 or coord == 7:
                edge_bonus += 20
        king_proximity = max(0, 8 - dist)
        edge_bonus *= (1 + (king_proximity / 4) ** 2)
        score += edge_bonus

        # Aggressive pawn push
        for p in board.pieces(chess.PAWN, chess.WHITE):
            rank = chess.square_rank(p)
            file = chess.square_file(p)

            advancement = rank
            score += (advancement ** 2) * 10

            manhattan = abs(rank - wk_rank) + abs(file - wk_file)
            score += max(0, 8 - manhattan) ** 2

            if bk_file in [file - 1, file, file + 1] and bk_rank >= rank:
                score -= 30

        for p in board.pieces(chess.PAWN, chess.BLACK):
            rank = chess.square_rank(p)
            file = chess.square_file(p)

            advancement = 7 - rank
            score -= (advancement ** 2) * 10

            manhattan = abs(rank - bk_rank) + abs(file - bk_file)
            score -= max(0, 8 - manhattan) ** 2

            if wk_file in [file - 1, file, file + 1] and wk_rank <= rank:
                score += 30

    else:
        if board.has_kingside_castling_rights(chess.WHITE):
            score += 10
        if board.has_queenside_castling_rights(chess.WHITE):
            score += 5
        if board.has_kingside_castling_rights(chess.BLACK):
            score -= 10
        if board.has_queenside_castling_rights(chess.BLACK):
            score -= 5

    if len(board.pieces(chess.BISHOP, chess.WHITE)) >= 2:
        score += 15
    if len(board.pieces(chess.BISHOP, chess.BLACK)) >= 2:
        score -= 15

    return score if board.turn == chess.WHITE else -score


# SEARCH

TT = {}          # (zobrist_hash, depth) -> int score
TT_MOVE = {}     # zobrist_hash -> best move for ordering


def order_moves(board: chess.Board, moves, tt_move=None):
    moves = list(moves)

    if tt_move is not None:
        try:
            i = moves.index(tt_move)
            if i != 0:
                moves[0], moves[i] = moves[i], moves[0]
        except ValueError:
            pass

    scored = []
    for m in moves:
        s = 0
        if board.is_capture(m):
            victim = board.piece_type_at(m.to_square)
            attacker = board.piece_type_at(m.from_square)
            if victim is not None and attacker is not None:
                s += 10 * PIECE_VALUES.get(victim, 0) - PIECE_VALUES.get(attacker, 0)
            s += 10_000  # all captures before quiets
        if m.promotion:
            s += 1_000 + PIECE_VALUES.get(m.promotion, 0)
        scored.append((s, m))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [m for _, m in scored]

# experimental, to improve on lter
def quiesce(board: chess.Board, alpha: int, beta: int, depth=0, max_q_depth=8) -> int:
    stand_pat = evaluate(board)

    if stand_pat >= beta:
        return beta
    if stand_pat > alpha:
        alpha = stand_pat

    if depth >= max_q_depth:
        return alpha

    h = zobrist_hash(board)
    tt_m = TT_MOVE.get(h)

    capture_moves = [m for m in board.legal_moves if board.is_capture(m)]
    if not capture_moves:
        return alpha

    capture_moves = order_moves(board, capture_moves, tt_move=tt_m)

    for move in capture_moves:
        board.push(move)
        score = -quiesce(board, -beta, -alpha, depth=depth + 1, max_q_depth=max_q_depth)
        board.pop()

        if score >= beta:
            return beta
        if score > alpha:
            alpha = score

    return alpha


def search(board: chess.Board, depth: int, alpha: int, beta: int) -> int:
    if board.is_game_over():
        return evaluate(board)

    h = zobrist_hash(board)
    key = (h, depth)

    entry = TT.get(key)
    if entry is not None:
        return entry

    if depth <= 0:
        val = quiesce(board, alpha, beta)
        TT[key] = val
        return val

    moves = generate_forced_capture_moves(board)
    if not moves:
        return evaluate(board)

    captures_exist = False
    for m in moves:
        if board.is_capture(m):
            captures_exist = True
            break
    next_depth = depth if captures_exist else (depth - 1)

    tt_m = TT_MOVE.get(h)
    moves = order_moves(board, moves, tt_move=tt_m)

    best = -INF
    best_move = None

    for move in moves:
        board.push(move)
        score = -search(board, next_depth, -beta, -alpha)
        board.pop()

        if score > best:
            best = score
            best_move = move

        if score > alpha:
            alpha = score
        if alpha >= beta:
            break

    TT[key] = best
    if best_move is not None:
        TT_MOVE[h] = best_move

    return best


def find_best_move(board: chess.Board, max_depth: int, epsilon: float = 0.0):
    moves = generate_forced_capture_moves(board)
    if not moves:
        return None

    for move in moves:
        board.push(move)
        if board.is_checkmate():
            board.pop()
            return move
        board.pop()

    if epsilon > 0.0 and random.random() < epsilon:
        return random.choice(moves)

    best_move = None

    for depth in range(1, max_depth + 1):
        local_best = None
        local_best_score = -INF

        h = zobrist_hash(board)
        ordered_moves = order_moves(board, moves, tt_move=TT_MOVE.get(h))

        for move in ordered_moves:
            board.push(move)
            score = -search(board, depth - 1, -INF, INF)
            board.pop()

            if score > local_best_score or local_best is None:
                local_best_score = score
                local_best = move

        if local_best is not None:
            best_move = local_best

    return best_move

def has_any_capture(board: chess.Board) -> bool:
    for m in board.legal_moves:
        if board.is_capture(m):
            return True
    return False

class XBoardEngine:
    def __init__(self):
        self.board = chess.Board()
        self.fixed_depth = 4
        self.fixed_epsilon = 0

        self.force_mode = True
        self.just_moved_since_last_usermove = False

        self.variant_capture = False

        self.time_left_ms = None
        self.opp_time_left_ms = None

    def log(self, msg: str):
        print(msg, file=sys.stderr, flush=True)

    def _do_engine_move(self):
        if self.board.is_game_over():
            print("resign", flush=True)
            return

        quiet_depth = 3

        capture_depth = 4


        if has_any_capture(self.board):
            depth = capture_depth
        else:
            depth = quiet_depth

        best_move = find_best_move(self.board, depth, self.fixed_epsilon)
        if best_move is None or best_move not in self.board.legal_moves:
            print("resign", flush=True)
            return

        self.board.push(best_move)
        self.just_moved_since_last_usermove = True
        print(f"move {best_move.uci()}", flush=True)


    def handle_new(self):
        self.board = chess.Board()
        self.just_moved_since_last_usermove = False

    def handle_force(self):
        self.force_mode = True
        self.just_moved_since_last_usermove = False

    def handle_go(self):
        self.force_mode = False
        if not self.just_moved_since_last_usermove:
            self._do_engine_move()

    def handle_setboard(self, fen: str):
        try:
            self.board = chess.Board(fen)
        except ValueError:
            self.log(f"Invalid FEN in setboard: {fen}")
        self.just_moved_since_last_usermove = False

    def handle_usermove(self, token: str):
        try:
            move = self.board.parse_uci(token)
        except ValueError:
            try:
                move = self.board.parse_san(token)
            except ValueError:
                self.log(f"Illegal move received: {token}")
                return

        if move not in self.board.legal_moves:
            self.log(f"Illegal move (not in legal_moves): {token}")
            return

        self.board.push(move)
        self.just_moved_since_last_usermove = False

        if not self.force_mode and not self.board.is_game_over():
            self._do_engine_move()

    def is_coord_move(self, line: str) -> bool:
        if len(line) not in (4, 5):
            return False
        f1, r1, f2, r2 = line[0], line[1], line[2], line[3]
        if f1 not in "abcdefgh" or f2 not in "abcdefgh":
            return False
        if r1 not in "12345678" or r2 not in "12345678":
            return False
        return True

    def loop(self):
        for raw_line in sys.stdin:
            line = raw_line.strip()
            if not line:
                continue

            if line == "xboard":
                continue

            if line.startswith("protover"):
                print("feature done=0", flush=True)
                print("feature san=0", flush=True)
                print("feature usermove=1", flush=True)
                print("feature setboard=1", flush=True)
                print("feature setup=1", flush=True)
                print('feature variants="normal,capture"', flush=True)
                print("feature ping=1", flush=True)
                print("feature done=1", flush=True)
                continue

            if line == "new":
                self.handle_new()
                continue

            if line.startswith("variant"):
                parts = line.split()
                if len(parts) >= 2:
                    v = parts[1].strip().lower()
                    self.variant_capture = (v == "capture")
                    self.log(f"Variant set to {v}, capture_variant={self.variant_capture}")
                continue

            if line == "force":
                self.handle_force()
                continue

            if line == "go" or line == "playother":
                self.handle_go()
                continue

            if line == "quit":
                break

            if line.startswith("time"):
                try:
                    t_cs = int(line.split()[1])
                    # do something with time
                    self.time_left_ms = t_cs * 10
                except Exception:
                    pass
                continue

            if line.startswith("otim"):
                try:
                    t_cs = int(line.split()[1])
                    self.opp_time_left_ms = t_cs * 10
                except Exception:
                    pass
                continue

            if line in ("white", "black", "computer"):
                continue

            if line.startswith("ping"):
                parts = line.split()
                if len(parts) >= 2:
                    print(f"pong {parts[1]}", flush=True)
                else:
                    print("pong", flush=True)
                continue

            if line.startswith("setboard"):
                fen = line[len("setboard"):].strip()
                self.handle_setboard(fen)
                continue

            if line.startswith("result"):
                self.force_mode = True
                self.just_moved_since_last_usermove = False
                continue

            if line.startswith("usermove"):
                parts = line.split()
                if len(parts) >= 2:
                    self.handle_usermove(parts[1])
                continue

            if self.is_coord_move(line):
                self.handle_usermove(line)
                continue

            continue


def main():
    # have to do this, why does xboard just kill us otherwise???
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    engine = XBoardEngine()
    engine.loop()


if __name__ == "__main__":
    main()
