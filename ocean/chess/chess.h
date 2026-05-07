#pragma once
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <time.h>
#include <math.h>
#include "raylib.h"
#include "stockfish_wrapper.h"

//Board dimensions
#define BOARD_SQ    64
#define MAX_MOVES   256
#define CHESS_OBS_AUX  7
#define CHESS_BOARD_OBS_SIZE (12 * BOARD_SQ + CHESS_OBS_AUX)
#define ACTION_SPACE_SIZE (BOARD_SQ * 73)
#define CHESS_ACTION_MASK_OFFSET CHESS_BOARD_OBS_SIZE
#define CHESS_OBS_SIZE (CHESS_BOARD_OBS_SIZE + ACTION_SPACE_SIZE)
#define CHESS_ACT_SIZE ACTION_SPACE_SIZE
#define POSITION_HISTORY_LEN 128
#define BOARD_SIZE  512
#define BOARD_PAD   24
#define STATUS_H    92
#define WIN_W       (BOARD_SIZE + BOARD_PAD * 2)
#define WIN_H       (BOARD_SIZE + BOARD_PAD * 2 + STATUS_H)
#define SQ_SIZE     (BOARD_SIZE / 8)
#define BOARD_LEFT  BOARD_PAD
#define BOARD_TOP   BOARD_PAD

//Piece encoding (0=empty, 1-6=white, 7-12=black)
#define EMPTY    0
#define W_PAWN   1
#define W_KNIGHT 2
#define W_BISHOP 3
#define W_ROOK   4
#define W_QUEEN  5
#define W_KING   6
#define B_PAWN   7
#define B_KNIGHT 8
#define B_BISHOP 9
#define B_ROOK   10
#define B_QUEEN  11
#define B_KING   12

// Castling rights bits
#define CASTLE_WK 1
#define CASTLE_WQ 2
#define CASTLE_BK 4
#define CASTLE_BQ 8

//Rewards
#define REWARD_WIN      1.0f
#define REWARD_LOSS    -1.0f
#define REWARD_DRAW    -0.1f
#define REWARD_ILLEGAL -0.1f

//Piece helpers
#define PIECE_COLOR(p)   ((p) >= 7 ? 1 : 0)
#define IS_WHITE(p)      ((p) >= 1 && (p) <= 6)
#define IS_BLACK(p)      ((p) >= 7 && (p) <= 12)
#define PIECE_TYPE(p)    ((p) > 6 ? (p) - 6 : (p))
#define OPP_PIECE(p)     ((p) <= 6 ? (p) + 6 : (p) - 6)
#define SAME_COLOR(a,b)  ((a) != EMPTY && (b) != EMPTY && PIECE_COLOR(a) == PIECE_COLOR(b))
#define RANK(sq)         ((sq) / 8)
#define FILE(sq)         ((sq) % 8)
#define SQ(r,f)          ((r)*8+(f))
#define IN_BOUNDS(r,f)   ((r)>=0&&(r)<8&&(f)>=0&&(f)<8)


typedef struct Log Log;
struct Log {
	float perf;
	float score;
	float episode_return;
	float episode_length;
	float illegal_moves;
	float draws;
	float stockfish_moves;
	float stockfish_fallbacks;
	float stockfish_cp;
	float n;
};


typedef struct Client Client;
struct Client {
	Font font;
	Texture2D pieces[13];
};


typedef struct {
	int8_t from, to;
	int8_t promo;     
	int8_t ep_capture; 
} Move;

typedef struct Chess Chess;
struct Chess {
	float*   observations;
	float*   actions;
	float*   rewards;
	float*   terminals;
	int      num_agents;
	Log      log;
	Client*  client;
	unsigned int rng;
	int8_t   board[64];
	int      side;       
	int8_t   ep_square;    
	uint8_t  castling;      
	int      halfmove_clock;
	int      fullmove;
	int8_t   last_from, last_to;

	int      selfplay;  
	int      human_side; // 0 none, 1 white, 2 black
	int      stockfish_enabled;
	int      stockfish_depth;
	int      stockfish_movetime_ms;
	int      ppo_side;
	int      randomize_ppo_side;
	int      max_ppo_turns;
	int      eval_shaping;
	float    last_stockfish_eval;
	int      has_stockfish_eval;
	StockfishEngine stockfish;

	int      tick;
	int      game_over;
	int8_t   selected_sq;
	int8_t   promotion_choice;
	int      pending_human_action;
	uint64_t position_history[POSITION_HISTORY_LEN];
	int      history_len;
};

void allocate_cchess(Chess* env) {
	env->observations = (float*)calloc(CHESS_OBS_SIZE, sizeof(float));
	env->actions      = (float*)calloc(1,   sizeof(float));
	env->rewards      = (float*)calloc(1,   sizeof(float));
	env->terminals    = (float*)calloc(1,   sizeof(float));
}
void free_allocated_cchess(Chess* env) {
	free(env->observations);
	free(env->actions);
	free(env->rewards);
	free(env->terminals);
}


static void set_start_position(Chess* env) {
	memset(env->board, EMPTY, sizeof(env->board));
	// White back rank
	env->board[0]=W_ROOK; env->board[1]=W_KNIGHT; env->board[2]=W_BISHOP;
	env->board[3]=W_QUEEN; env->board[4]=W_KING;
	env->board[5]=W_BISHOP; env->board[6]=W_KNIGHT; env->board[7]=W_ROOK;
	for (int f=0; f<8; f++) env->board[8+f]  = W_PAWN;
	// Black back rank
	env->board[56]=B_ROOK; env->board[57]=B_KNIGHT; env->board[58]=B_BISHOP;
	env->board[59]=B_QUEEN; env->board[60]=B_KING;
	env->board[61]=B_BISHOP; env->board[62]=B_KNIGHT; env->board[63]=B_ROOK;
	for (int f=0; f<8; f++) env->board[48+f] = B_PAWN;
	env->side            = 0;
	env->ep_square       = -1;
	env->castling        = CASTLE_WK|CASTLE_WQ|CASTLE_BK|CASTLE_BQ;
	env->halfmove_clock  = 0;
	env->fullmove        = 1;
	env->last_from       = -1;
	env->last_to         = -1;
	env->selected_sq     = -1;
	env->promotion_choice = 5;
	env->pending_human_action = -1;
	env->history_len     = 0;
}

static inline int square_color(int sq) {
	return (RANK(sq) + FILE(sq)) & 1;
}

static int8_t normalized_ep_square_board(const int8_t* board, int side, int8_t ep_square) {
	if (ep_square < 0) return -1;

	int r = RANK(ep_square);
	int f = FILE(ep_square);
	int pawn = side == 0 ? W_PAWN : B_PAWN;
	int from_r = side == 0 ? r - 1 : r + 1;
	if (from_r < 0 || from_r >= 8) return -1;

	if (f > 0 && board[SQ(from_r, f - 1)] == pawn) return ep_square;
	if (f < 7 && board[SQ(from_r, f + 1)] == pawn) return ep_square;
	return -1;
}

static uint64_t position_hash_board(const int8_t* board, int side, uint8_t castling, int8_t ep_square) {
	uint64_t hash = 1469598103934665603ULL;
	for (int sq = 0; sq < 64; sq++) {
		hash ^= (uint8_t)(board[sq] + 1);
		hash *= 1099511628211ULL;
	}

	hash ^= (uint8_t)side;
	hash *= 1099511628211ULL;
	hash ^= castling;
	hash *= 1099511628211ULL;
	hash ^= (uint8_t)(normalized_ep_square_board(board, side, ep_square) + 2);
	hash *= 1099511628211ULL;
	return hash;
}

static uint64_t current_position_hash(const Chess* env) {
	return position_hash_board(env->board, env->side, env->castling, env->ep_square);
}

static void reset_position_history(Chess* env) {
	env->history_len = 1;
	env->position_history[0] = current_position_hash(env);
}

static void push_position_history(Chess* env) {
	uint64_t hash = current_position_hash(env);
	if (env->history_len < POSITION_HISTORY_LEN) {
		env->position_history[env->history_len++] = hash;
		return;
	}

	memmove(env->position_history, env->position_history + 1,
		(POSITION_HISTORY_LEN - 1) * sizeof(uint64_t));
	env->position_history[POSITION_HISTORY_LEN - 1] = hash;
}

static int current_position_repetitions(const Chess* env) {
	uint64_t hash = current_position_hash(env);
	int count = 0;
	for (int i = 0; i < env->history_len; i++) {
		count += (env->position_history[i] == hash);
	}
	return count;
}


static bool sq_attacked(const int8_t* board, int sq, int by_color) {
	int r = RANK(sq), f = FILE(sq);

	// Pawns
	if (by_color == 0) {
		if (r>0 && f>0 && board[SQ(r-1,f-1)]==W_PAWN) return true;
		if (r>0 && f<7 && board[SQ(r-1,f+1)]==W_PAWN) return true;
	} else {
		if (r<7 && f>0 && board[SQ(r+1,f-1)]==B_PAWN) return true;
		if (r<7 && f<7 && board[SQ(r+1,f+1)]==B_PAWN) return true;
	}

	// Knights
	int8_t knight = by_color==0 ? W_KNIGHT : B_KNIGHT;
	static const int kdr[]={-2,-2,-1,-1,1,1,2,2};
	static const int kdf[]={-1,1,-2,2,-2,2,-1,1};
	for (int i=0;i<8;i++) {
		int nr=r+kdr[i], nf=f+kdf[i];
		if (IN_BOUNDS(nr,nf) && board[SQ(nr,nf)]==knight) return true;
	}

	// Bishops / Queens (diagonals)
	int8_t bishop = by_color==0?W_BISHOP:B_BISHOP;
	int8_t queen  = by_color==0?W_QUEEN :B_QUEEN;
	static const int ddr[]={1,1,-1,-1};
	static const int ddf[]={1,-1,1,-1};
	for (int d=0;d<4;d++) {
		for (int s=1;s<8;s++) {
			int nr=r+ddr[d]*s, nf=f+ddf[d]*s;
			if (!IN_BOUNDS(nr,nf)) break;
			int8_t p=board[SQ(nr,nf)];
			if (p==bishop||p==queen) return true;
			if (p!=EMPTY) break;
		}
	}

	// Rooks / Queens (orthogonals)
	int8_t rook = by_color==0?W_ROOK:B_ROOK;
	static const int odr[]={1,-1,0,0};
	static const int odf[]={0,0,1,-1};
	for (int d=0;d<4;d++) {
		for (int s=1;s<8;s++) {
			int nr=r+odr[d]*s, nf=f+odf[d]*s;
			if (!IN_BOUNDS(nr,nf)) break;
			int8_t p=board[SQ(nr,nf)];
			if (p==rook||p==queen) return true;
			if (p!=EMPTY) break;
		}
	}

	// King
	int8_t king = by_color==0?W_KING:B_KING;
	for (int dr=-1;dr<=1;dr++) for (int df=-1;df<=1;df++) {
		if (dr==0&&df==0) continue;
		int nr=r+dr, nf=f+df;
		if (IN_BOUNDS(nr,nf) && board[SQ(nr,nf)]==king) return true;
	}
	return false;
}

static int find_king(const int8_t* board, int color) {
	int8_t king = color==0 ? W_KING : B_KING;
	for (int sq=0;sq<64;sq++) if (board[sq]==king) return sq;
	return -1;
}

static bool in_check(const int8_t* board, int color) {
	int ksq = find_king(board, color);
	if (ksq<0) return false;
	return sq_attacked(board, ksq, 1-color);
}

// ─────────────────────────────────────────────────────────────────────────
//  Move generation
// ─────────────────────────────────────────────────────────────────────────
static int gen_moves(const Chess* env, Move* moves) {
	int n = 0;
	int side = env->side;
	int8_t enemy_king = side == 0 ? B_KING : W_KING;
	int8_t board[64];
	memcpy(board, env->board, 64);

	for (int fr=0; fr<64; fr++) {
		int8_t p = board[fr];
		if (p==EMPTY || PIECE_COLOR(p)!=side) continue;
		int r=RANK(fr), f=FILE(fr);
		int pt = PIECE_TYPE(p);

		if (pt==1) { // Pawn
			int dir = (side==0)?1:-1;
			int start_rank = (side==0)?1:6;
			int promo_rank = (side==0)?6:1;

			// Single push
			int nr=r+dir, nf=f;
			if (IN_BOUNDS(nr,nf) && board[SQ(nr,nf)]==EMPTY) {
				if (r==promo_rank) {
					for (int pp=2;pp<=5;pp++)
						moves[n++]=(Move){fr,SQ(nr,nf),pp,-1};
				} else moves[n++]=(Move){fr,SQ(nr,nf),0,-1};

				// Double push
				if (r==start_rank && board[SQ(r+2*dir,nf)]==EMPTY)
					moves[n++]=(Move){fr,SQ(r+2*dir,nf),0,-1};
			}
			// Captures
			for (int df=-1;df<=1;df+=2) {
				int ncf=f+df;
				if (!IN_BOUNDS(nr,ncf)) continue;
				int to=SQ(nr,ncf);
				bool ep = (to==env->ep_square);
				if (!ep && board[to] == enemy_king) continue;
				if ((board[to]!=EMPTY && !SAME_COLOR(p,board[to])) || ep) {
					int8_t epc = ep ? SQ(r,ncf) : -1;
					if (r==promo_rank) {
						for (int pp=2;pp<=5;pp++)
							moves[n++]=(Move){fr,to,pp,epc};
					} else moves[n++]=(Move){fr,to,0,epc};
				}
			}

		} else if (pt==2) { // Knight
			static const int kdr[]={-2,-2,-1,-1,1,1,2,2};
			static const int kdf[]={-1,1,-2,2,-2,2,-1,1};
			for (int i=0;i<8;i++) {
				int nr2=r+kdr[i], nf2=f+kdf[i];
				if (!IN_BOUNDS(nr2,nf2)) continue;
				int to=SQ(nr2,nf2);
				if (board[to] == enemy_king) continue;
				if (!SAME_COLOR(p,board[to]))
					moves[n++]=(Move){fr,to,0,-1};
			}

		} else if (pt==3||pt==5) { // Bishop or Queen
			static const int ddr[]={1,1,-1,-1};
			static const int ddf[]={1,-1,1,-1};
			for (int d=0;d<4;d++) {
				for (int s=1;s<8;s++) {
					int nr2=r+ddr[d]*s, nf2=f+ddf[d]*s;
					if (!IN_BOUNDS(nr2,nf2)) break;
					int to=SQ(nr2,nf2);
					if (board[to] == enemy_king) break;
					if (SAME_COLOR(p,board[to])) break;
					moves[n++]=(Move){fr,to,0,-1};
					if (board[to]!=EMPTY) break;
				}
			}
			if (pt==3) continue; // bishop done
			// Queen continues to rook directions below
		}

		if (pt==4||pt==5) { // Rook or Queen
			static const int odr[]={1,-1,0,0};
			static const int odf[]={0,0,1,-1};
			for (int d=0;d<4;d++) {
				for (int s=1;s<8;s++) {
					int nr2=r+odr[d]*s, nf2=f+odf[d]*s;
					if (!IN_BOUNDS(nr2,nf2)) break;
					int to=SQ(nr2,nf2);
					if (board[to] == enemy_king) break;
					if (SAME_COLOR(p,board[to])) break;
					moves[n++]=(Move){fr,to,0,-1};
					if (board[to]!=EMPTY) break;
				}
			}

		} else if (pt==6) { // King
			for (int dr=-1;dr<=1;dr++) for (int df2=-1;df2<=1;df2++) {
				if (dr==0&&df2==0) continue;
				int nr2=r+dr, nf2=f+df2;
				if (!IN_BOUNDS(nr2,nf2)) continue;
				int to=SQ(nr2,nf2);
				if (board[to] == enemy_king) continue;
				if (!SAME_COLOR(p,board[to]))
					moves[n++]=(Move){fr,to,0,-1};
			}
			// Castling
			if (side==0 && fr==4) {
				if ((env->castling&CASTLE_WK) &&
					board[7]==W_ROOK &&
					board[5]==EMPTY && board[6]==EMPTY &&
					!sq_attacked(board,4,1) && !sq_attacked(board,5,1) && !sq_attacked(board,6,1))
					moves[n++]=(Move){4,6,0,-1};
				if ((env->castling&CASTLE_WQ) &&
					board[0]==W_ROOK &&
					board[3]==EMPTY && board[2]==EMPTY && board[1]==EMPTY &&
					!sq_attacked(board,4,1) && !sq_attacked(board,3,1) && !sq_attacked(board,2,1))
					moves[n++]=(Move){4,2,0,-1};
			}
			if (side==1 && fr==60) {
				if ((env->castling&CASTLE_BK) &&
					board[63]==B_ROOK &&
					board[61]==EMPTY && board[62]==EMPTY &&
					!sq_attacked(board,60,0) && !sq_attacked(board,61,0) && !sq_attacked(board,62,0))
					moves[n++]=(Move){60,62,0,-1};
				if ((env->castling&CASTLE_BQ) &&
					board[56]==B_ROOK &&
					board[59]==EMPTY && board[58]==EMPTY && board[57]==EMPTY &&
					!sq_attacked(board,60,0) && !sq_attacked(board,59,0) && !sq_attacked(board,58,0))
					moves[n++]=(Move){60,58,0,-1};
			}
		}
	}
	return n;
}

//  Apply move to board (legality filter via temp copy)
static void apply_move_to_board(int8_t* board, uint8_t* castling,
								 int8_t* ep_out, int side,
								 const Move* m) {
	int8_t piece = board[m->from];
	int pt = PIECE_TYPE(piece);

	*ep_out = -1;

	// En passant capture
	if (m->ep_capture >= 0) board[m->ep_capture] = EMPTY;

	// Castling: move rook
	if (pt==6) {
		if (m->from==4 && m->to==6)  { board[7]=EMPTY; board[5]=W_ROOK; }
		if (m->from==4 && m->to==2)  { board[0]=EMPTY; board[3]=W_ROOK; }
		if (m->from==60 && m->to==62){ board[63]=EMPTY; board[61]=B_ROOK; }
		if (m->from==60 && m->to==58){ board[56]=EMPTY; board[59]=B_ROOK; }
		// Clear castling rights for this side
		if (side==0) *castling &= ~(CASTLE_WK|CASTLE_WQ);
		else         *castling &= ~(CASTLE_BK|CASTLE_BQ);
	}

	// Update castling rights when rook moves/is captured
	if (m->from==0  || m->to==0)  *castling &= ~CASTLE_WQ;
	if (m->from==7  || m->to==7)  *castling &= ~CASTLE_WK;
	if (m->from==56 || m->to==56) *castling &= ~CASTLE_BQ;
	if (m->from==63 || m->to==63) *castling &= ~CASTLE_BK;

	// Double pawn push: set ep square
	if (pt==1 && abs(m->to - m->from)==16) {
		*ep_out = (m->from + m->to) / 2;
	}

	// Make the move
	board[m->to]   = m->promo ? (int8_t)(m->promo + side*6) : piece;
	board[m->from] = EMPTY;
}

static void commit_move(Chess* env, const Move* m) {
	int side = env->side;
	int8_t moving_piece = env->board[m->from];
	int8_t captured_piece = env->board[m->to];
	uint8_t old_castling = env->castling;
	bool resets_halfmove = PIECE_TYPE(moving_piece) == 1 ||
		captured_piece != EMPTY || m->ep_capture >= 0;

	apply_move_to_board(env->board, &env->castling, &env->ep_square, side, m);
	env->last_from = m->from;
	env->last_to   = m->to;
	env->selected_sq = -1;

	if (side == 1) env->fullmove++;
	env->side ^= 1;
	env->halfmove_clock = resets_halfmove ? 0 : env->halfmove_clock + 1;

	if (resets_halfmove || env->castling != old_castling) {
		reset_position_history(env);
	} else {
		push_position_history(env);
	}
}

// Returns true if move keeps own king out of check
static bool is_legal(const Chess* env, const Move* m) {
	int8_t b[64];
	uint8_t castling = env->castling;
	int8_t ep_dummy;
	memcpy(b, env->board, 64);
	apply_move_to_board(b, &castling, &ep_dummy, env->side, m);
	return !in_check(b, env->side);
}

//  Legal move filtering
static int legal_moves(const Chess* env, Move* legal, Move* pseudo) {
	int np = gen_moves(env, pseudo);
	int nl = 0;
	for (int i=0;i<np;i++)
		if (is_legal(env, &pseudo[i]))
			legal[nl++] = pseudo[i];
	return nl;
}

static const int QUEEN_DIRECTIONS[8][2] = {
	{ 1,  0}, { 1,  1}, { 0,  1}, {-1,  1},
	{-1,  0}, {-1, -1}, { 0, -1}, { 1, -1},
};

static const int KNIGHT_DIRECTIONS[8][2] = {
	{ 2,  1}, { 1,  2}, {-1,  2}, {-2,  1},
	{-2, -1}, {-1, -2}, { 1, -2}, { 2, -1},
};

static const int UNDERPROMOTION_PIECES[3] = {2, 3, 4}; // knight, bishop, rook

static inline bool move_reaches_promotion_rank(int side, int to) {
	return RANK(to) == (side == 0 ? 7 : 0);
}

static inline Move invalid_decoded_move(void) {
	return (Move){-1, -1, 0, -1};
}

static inline int encode_action_index(const Move* m, int side) {
	if (m->from < 0 || m->from >= BOARD_SQ || m->to < 0 || m->to >= BOARD_SQ) {
		return -1;
	}

	int from_r = RANK(m->from);
	int from_f = FILE(m->from);
	int to_r = RANK(m->to);
	int to_f = FILE(m->to);
	int dr = to_r - from_r;
	int df = to_f - from_f;

	if (m->promo >= 2 && m->promo <= 4) {
		int forward = side == 0 ? 1 : -1;
		if (dr != forward || df < -1 || df > 1) return -1;
		int piece_idx = m->promo - 2;
		int dir_idx = df + 1;
		return m->from * 73 + 64 + piece_idx * 3 + dir_idx;
	}

	for (int d = 0; d < 8; d++) {
		for (int dist = 1; dist <= 7; dist++) {
			if (dr == QUEEN_DIRECTIONS[d][0] * dist &&
					df == QUEEN_DIRECTIONS[d][1] * dist) {
				return m->from * 73 + d * 7 + (dist - 1);
			}
		}
	}

	for (int d = 0; d < 8; d++) {
		if (dr == KNIGHT_DIRECTIONS[d][0] && df == KNIGHT_DIRECTIONS[d][1]) {
			return m->from * 73 + 56 + d;
		}
	}

	return -1;
}

static inline Move decode_action_index(const Chess* env, int action_idx) {
	if (action_idx < 0 || action_idx >= ACTION_SPACE_SIZE) return invalid_decoded_move();

	int from = action_idx / 73;
	int plane = action_idx % 73;
	int piece = env->board[from];
	if (piece == EMPTY || PIECE_COLOR(piece) != env->side) return invalid_decoded_move();

	int from_r = RANK(from);
	int from_f = FILE(from);
	int to_r = from_r;
	int to_f = from_f;
	Move m = {(int8_t)from, -1, 0, -1};

	if (plane < 56) {
		int dir = plane / 7;
		int dist = plane % 7 + 1;
		to_r += QUEEN_DIRECTIONS[dir][0] * dist;
		to_f += QUEEN_DIRECTIONS[dir][1] * dist;
	} else if (plane < 64) {
		int dir = plane - 56;
		to_r += KNIGHT_DIRECTIONS[dir][0];
		to_f += KNIGHT_DIRECTIONS[dir][1];
	} else {
		int rel = plane - 64;
		int piece_idx = rel / 3;
		int dir_idx = rel % 3;
		to_r += env->side == 0 ? 1 : -1;
		to_f += dir_idx - 1;
		m.promo = (int8_t)UNDERPROMOTION_PIECES[piece_idx];
	}

	if (!IN_BOUNDS(to_r, to_f)) return invalid_decoded_move();
	m.to = (int8_t)SQ(to_r, to_f);

	if (plane < 64 && PIECE_TYPE(piece) == 1 &&
			move_reaches_promotion_rank(env->side, m.to)) {
		m.promo = 5;
	}

	if (plane >= 64 && (PIECE_TYPE(piece) != 1 ||
			!move_reaches_promotion_rank(env->side, m.to))) {
		return invalid_decoded_move();
	}

	return m;
}

static void generate_action_mask(const Chess* env, float* mask_out) {
	memset(mask_out, 0, ACTION_SPACE_SIZE * sizeof(float));
	if (env->game_over) return;

	Move pseudo[MAX_MOVES], legal[MAX_MOVES];
	int nl = legal_moves(env, legal, pseudo);
	for (int i = 0; i < nl; i++) {
		int action_idx = encode_action_index(&legal[i], env->side);
		if (action_idx >= 0 && action_idx < ACTION_SPACE_SIZE) {
			mask_out[action_idx] = 1.0f;
		}
	}
}

static int find_legal_move_index(const Chess* env, int from, int to, int preferred_promo) {
	Move pseudo[MAX_MOVES], legal[MAX_MOVES];
	int nl = legal_moves(env, legal, pseudo);
	int fallback = -1;

	for (int i = 0; i < nl; i++) {
		if (legal[i].from != from || legal[i].to != to) continue;
		int action_idx = encode_action_index(&legal[i], env->side);
		if (fallback < 0) fallback = action_idx;
		if (legal[i].promo == preferred_promo) return action_idx;
		if (preferred_promo == 0 && legal[i].promo == 0) return action_idx;
	}

	return fallback;
}

static inline bool is_human_turn(const Chess* env) {
	return (env->human_side == 1 && env->side == 0) ||
		(env->human_side == 2 && env->side == 1);
}

static bool insufficient_material(const Chess* env) {
	int white_minors = 0;
	int black_minors = 0;
	int white_knights = 0;
	int black_knights = 0;
	int bishop_count = 0;
	int bishop_color_mask = 0;
	int total_non_king = 0;

	for (int sq = 0; sq < 64; sq++) {
		int8_t piece = env->board[sq];
		if (piece == EMPTY || PIECE_TYPE(piece) == 6) continue;

		total_non_king++;
		switch (PIECE_TYPE(piece)) {
		case 1:
		case 4:
		case 5:
			return false;
		case 2:
			if (IS_WHITE(piece)) {
				white_knights++;
				white_minors++;
			} else {
				black_knights++;
				black_minors++;
			}
			break;
		case 3:
			bishop_count++;
			bishop_color_mask |= 1 << square_color(sq);
			if (IS_WHITE(piece)) white_minors++;
			else                 black_minors++;
			break;
		default:
			return false;
		}
	}

	if (total_non_king == 0) return true;
	if (total_non_king == 1) return true;
	if (white_minors <= 1 && black_minors <= 1) return true;

	// Any number of bishops all living on the same color complex is dead.
	if (bishop_count == total_non_king && (bishop_color_mask == 1 || bishop_color_mask == 2)) {
		return true;
	}

	// Two knights alone are not a dead position under FIDE rules.
	if ((white_knights == 2 && white_minors == 2 && black_minors == 0) ||
			(black_knights == 2 && black_minors == 2 && white_minors == 0)) {
		return false;
	}

	return false;
}

static bool forced_draw(const Chess* env) {
	return env->halfmove_clock >= 100 ||
		current_position_repetitions(env) >= 3 ||
		insufficient_material(env);
}

//  Observation encoding: 775 board features followed by a 4672 action mask.
static void encode_obs(Chess* env) {
	float* obs = env->observations;
	memset(obs, 0, CHESS_OBS_SIZE * sizeof(float));

	// Piece planes: plane = piece-1 (0-11), square = rank*8+file
	// From white's POV always (rank 0 = a1)
	for (int sq=0;sq<64;sq++) {
		int8_t p = env->board[sq];
		if (p!=EMPTY) obs[(p-1)*64 + sq] = 1.0f;
	}

	// Auxiliary features at offset 768
	int a = 768;
	obs[a+0] = (env->castling & CASTLE_WK) ? 1.0f : 0.0f;
	obs[a+1] = (env->castling & CASTLE_WQ) ? 1.0f : 0.0f;
	obs[a+2] = (env->castling & CASTLE_BK) ? 1.0f : 0.0f;
	obs[a+3] = (env->castling & CASTLE_BQ) ? 1.0f : 0.0f;
	obs[a+4] = env->ep_square >= 0 ? FILE(env->ep_square) / 7.0f : -1.0f;
	obs[a+5] = env->halfmove_clock / 100.0f;
	obs[a+6] = (float)env->side;

	generate_action_mask(env, obs + CHESS_ACTION_MASK_OFFSET);
}

//  Logging
static void add_log(Chess* env, float reward) {
	env->log.perf           += (reward > 0.0f) ? 1.0f : 0.0f;
	env->log.score          += reward;
	env->log.episode_return += reward;
	env->log.episode_length += env->tick;
	env->log.draws          += (reward == REWARD_DRAW) ? 1.0f : 0.0f;
	if (env->has_stockfish_eval) env->log.stockfish_cp += env->last_stockfish_eval;
	env->log.n              += 1.0f;
}

//  Random opponent move
static void play_random_move(Chess* env) {
	Move pseudo[MAX_MOVES], legal[MAX_MOVES];
	int n = legal_moves(env, legal, pseudo);
	if (n == 0) return;
	Move* m = &legal[rand_r(&env->rng) % n];
	commit_move(env, m);
}

static inline int stockfish_active(const Chess* env) {
	return env->stockfish_enabled && !env->selfplay;
}

static inline int normalized_ppo_side(const Chess* env) {
	return env->ppo_side == 1 ? 1 : 0;
}

static inline int controlled_side(const Chess* env) {
	if (env->human_side == 1) return 0;
	if (env->human_side == 2) return 1;
	return normalized_ppo_side(env);
}

static inline int clamp_int(int value, int lo, int hi) {
	if (value < lo) return lo;
	if (value > hi) return hi;
	return value;
}

static char piece_to_fen(int8_t piece) {
	switch (piece) {
	case W_PAWN:   return 'P';
	case W_KNIGHT: return 'N';
	case W_BISHOP: return 'B';
	case W_ROOK:   return 'R';
	case W_QUEEN:  return 'Q';
	case W_KING:   return 'K';
	case B_PAWN:   return 'p';
	case B_KNIGHT: return 'n';
	case B_BISHOP: return 'b';
	case B_ROOK:   return 'r';
	case B_QUEEN:  return 'q';
	case B_KING:   return 'k';
	default:       return '\0';
	}
}

static void board_to_fen(const Chess* env, char* out, size_t out_sz) {
	char board_part[96];
	int n = 0;
	for (int r = 7; r >= 0; r--) {
		int empty = 0;
		for (int f = 0; f < 8; f++) {
			int8_t piece = env->board[SQ(r, f)];
			if (piece == EMPTY) {
				empty++;
				continue;
			}
			if (empty > 0) {
				board_part[n++] = (char)('0' + empty);
				empty = 0;
			}
			board_part[n++] = piece_to_fen(piece);
		}
		if (empty > 0) board_part[n++] = (char)('0' + empty);
		if (r > 0) board_part[n++] = '/';
	}
	board_part[n] = '\0';

	char castling[5];
	int c = 0;
	if (env->castling & CASTLE_WK) castling[c++] = 'K';
	if (env->castling & CASTLE_WQ) castling[c++] = 'Q';
	if (env->castling & CASTLE_BK) castling[c++] = 'k';
	if (env->castling & CASTLE_BQ) castling[c++] = 'q';
	if (c == 0) castling[c++] = '-';
	castling[c] = '\0';

	char ep[3] = {'-', '\0', '\0'};
	int8_t normalized_ep = normalized_ep_square_board(
		env->board, env->side, env->ep_square);
	if (normalized_ep >= 0) {
		ep[0] = (char)('a' + FILE(normalized_ep));
		ep[1] = (char)('1' + RANK(normalized_ep));
		ep[2] = '\0';
	}

	snprintf(out, out_sz, "%s %c %s %s %d %d",
		board_part,
		env->side == 0 ? 'w' : 'b',
		castling,
		ep,
		env->halfmove_clock,
		env->fullmove);
}

static int uci_square_to_index(const char* s) {
	if (s[0] < 'a' || s[0] > 'h' || s[1] < '1' || s[1] > '8') return -1;
	return SQ(s[1] - '1', s[0] - 'a');
}

static int promotion_from_uci(char c) {
	switch (c) {
	case 'n': return 2;
	case 'b': return 3;
	case 'r': return 4;
	case 'q': return 5;
	default:  return 0;
	}
}

static bool legal_move_from_uci(const Chess* env, const char* uci, Move* out) {
	if (uci == NULL || strlen(uci) < 4) return false;
	int from = uci_square_to_index(uci);
	int to = uci_square_to_index(uci + 2);
	int promo = strlen(uci) >= 5 ? promotion_from_uci(uci[4]) : 0;
	if (from < 0 || to < 0) return false;

	Move pseudo[MAX_MOVES], legal[MAX_MOVES];
	int n = legal_moves(env, legal, pseudo);
	for (int i = 0; i < n; i++) {
		if (legal[i].from != from || legal[i].to != to) continue;
		if (legal[i].promo == promo ||
				(promo == 0 && legal[i].promo == 0)) {
			*out = legal[i];
			return true;
		}
	}
	return false;
}

static int stockfish_score_for_ppo(const Chess* env, const StockfishResult* result) {
	int score = clamp_int(result->score_cp, -4000, 4000);
	return env->side == controlled_side(env) ? score : -score;
}

static float stockfish_eval_reward(Chess* env, int cp_for_ppo) {
	if (!env->eval_shaping) return 0.0f;

	float eval_cp = (float)clamp_int(cp_for_ppo, -2000, 2000);
	float reward = 0.01f * tanhf(eval_cp / 600.0f);
	if (env->has_stockfish_eval) {
		float delta = eval_cp - env->last_stockfish_eval;
		reward += 0.04f * tanhf(delta / 250.0f);
	}
	env->last_stockfish_eval = eval_cp;
	env->has_stockfish_eval = 1;
	return reward;
}

static bool play_stockfish_move(Chess* env, float* shaped_reward) {
	if (shaped_reward != NULL) *shaped_reward = 0.0f;

	char fen[128];
	board_to_fen(env, fen, sizeof(fen));

	StockfishResult result;
	if (stockfish_go(&env->stockfish, fen, env->stockfish_depth,
			env->stockfish_movetime_ms, &result)) {
		if (result.has_score && shaped_reward != NULL) {
			int cp_for_ppo = stockfish_score_for_ppo(env, &result);
			*shaped_reward = stockfish_eval_reward(env, cp_for_ppo);
		}

		Move chosen;
		if (legal_move_from_uci(env, result.bestmove, &chosen)) {
			commit_move(env, &chosen);
			env->log.stockfish_moves += 1.0f;
			return true;
		}
	}

	env->log.stockfish_fallbacks += 1.0f;
	play_random_move(env);
	return false;
}

static float no_legal_moves_reward(const Chess* env, bool side_in_check) {
	if (!side_in_check) return REWARD_DRAW;
	int winner = 1 - env->side;
	return winner == controlled_side(env) ? REWARD_WIN : REWARD_LOSS;
}

//  Reset
void c_reset(Chess* env) {
	set_start_position(env);
	if (env->randomize_ppo_side) {
		env->ppo_side = (int)(rand_r(&env->rng) & 1);
	}

	env->tick       = 0;
	env->game_over  = 0;
	env->rewards[0]   = 0.0f;
	env->terminals[0] = 0.0f;
	env->last_stockfish_eval = 0.0f;
	env->has_stockfish_eval = 0;
	reset_position_history(env);

	if (stockfish_active(env)) {
		if (!stockfish_new_game(&env->stockfish)) {
			env->log.stockfish_fallbacks += 1.0f;
		}
		if (env->side != controlled_side(env)) {
			play_stockfish_move(env, NULL);
		}
	}

	encode_obs(env);
}

void init(Chess* env) {
	stockfish_engine_init(&env->stockfish);
	env->tick = 0;
	env->selected_sq = -1;
	if (env->rng == 0) env->rng = (unsigned int)time(NULL);
	if (env->stockfish_depth <= 0) env->stockfish_depth = 8;
	if (env->max_ppo_turns <= 0) env->max_ppo_turns = 200;
	env->ppo_side = normalized_ppo_side(env);
}

//  Step
void c_step(Chess* env) {
	env->rewards[0]   = -0.002f;
	env->terminals[0] = 0.0f;

	if (env->game_over) {
		c_reset(env);
		return;
	}

	Move pseudo[MAX_MOVES], legal[MAX_MOVES];
	int nl = legal_moves(env, legal, pseudo);
	bool side_in_check = in_check(env->board, env->side);

	if (nl == 0) {
		float r = no_legal_moves_reward(env, side_in_check);
		env->rewards[0]   = r;
		env->terminals[0] = 1.0f;
		env->game_over    = 1;
		add_log(env, r);
		encode_obs(env);
		return;
	}

	if (forced_draw(env)) {
		env->rewards[0]   = REWARD_DRAW;
		env->terminals[0] = 1.0f;
		env->game_over    = 1;
		add_log(env, REWARD_DRAW);
		encode_obs(env);
		return;
	}

	if (stockfish_active(env) && env->side != controlled_side(env)) {
		play_stockfish_move(env, NULL);
		nl = legal_moves(env, legal, pseudo);
		side_in_check = in_check(env->board, env->side);
		if (nl == 0) {
			float r = no_legal_moves_reward(env, side_in_check);
			env->rewards[0]   = r;
			env->terminals[0] = 1.0f;
			env->game_over    = 1;
			add_log(env, r);
		} else if (forced_draw(env)) {
			env->rewards[0]   = REWARD_DRAW;
			env->terminals[0] = 1.0f;
			env->game_over    = 1;
			add_log(env, REWARD_DRAW);
		}
		encode_obs(env);
		return;
	}

	int action_idx;
	if (is_human_turn(env)) {
		if (env->pending_human_action < 0) {
			encode_obs(env);
			return;
		}
		action_idx = env->pending_human_action;
		env->pending_human_action = -1;
	} else {
		float action_value = env->actions[0];
		action_idx = isfinite(action_value) ? (int)action_value : -1;
	}

	Move decoded = decode_action_index(env, action_idx);
	Move chosen = decoded;
	bool is_valid = false;
	for (int i = 0; i < nl; i++) {
		if (legal[i].from == decoded.from &&
				legal[i].to == decoded.to &&
				legal[i].promo == decoded.promo) {
			chosen = legal[i];
			is_valid = true;
			break;
		}
	}

	if (!is_valid) {
		env->rewards[0] = REWARD_ILLEGAL;
		env->terminals[0] = 1.0f;
		env->game_over = 1;
		env->log.illegal_moves += 1.0f;
		add_log(env, REWARD_ILLEGAL);
		encode_obs(env);
		return;
	}

	env->tick++;

	commit_move(env, &chosen);

	int opp_moves = legal_moves(env, legal, pseudo);
	bool opp_in_check = in_check(env->board, env->side);

	if (opp_moves == 0) {
		float r = no_legal_moves_reward(env, opp_in_check);
		env->rewards[0]   = r;
		env->terminals[0] = 1.0f;
		env->game_over    = 1;
		add_log(env, r);
		encode_obs(env);
		return;
	}

	if (forced_draw(env)) {
		env->rewards[0]   = REWARD_DRAW;
		env->terminals[0] = 1.0f;
		env->game_over    = 1;
		add_log(env, REWARD_DRAW);
		encode_obs(env);
		return;
	}

	if (env->tick >= env->max_ppo_turns) {
		env->rewards[0]   = REWARD_DRAW;
		env->terminals[0] = 1.0f;
		env->game_over    = 1;
		add_log(env, REWARD_DRAW);
		encode_obs(env);
		return;
	}

	if (!env->selfplay) {
		float shaped_reward = 0.0f;
		if (env->stockfish_enabled) {
			play_stockfish_move(env, &shaped_reward);
		} else {
			play_random_move(env);
		}
		env->rewards[0] += shaped_reward;

		int my_moves = legal_moves(env, legal, pseudo);
		bool my_in_check = in_check(env->board, env->side);
		if (my_moves == 0) {
			float r = no_legal_moves_reward(env, my_in_check);
			env->rewards[0]   = r;
			env->terminals[0] = 1.0f;
			env->game_over    = 1;
			add_log(env, r);
		} else if (forced_draw(env)) {
			env->rewards[0]   = REWARD_DRAW;
			env->terminals[0] = 1.0f;
			env->game_over    = 1;
			add_log(env, REWARD_DRAW);
		}
	}

	encode_obs(env);
}

//  Rendering
static const Color LIGHT_SQ    = {239, 217, 190, 255};
static const Color DARK_SQ     = {164, 120,  90, 255};
static const Color MOVE_SQ     = {236, 205,  95, 180};
static const Color SELECT_SQ   = { 90, 191, 166, 210};
static const Color CHECK_SQ    = {213,  86,  76, 210};
static const Color BOARD_FRAME = { 28,  33,  41, 255};
static const Color BG_TOP      = { 16,  23,  31, 255};
static const Color BG_BOTTOM   = { 37,  51,  66, 255};
static const Color PANEL_BG    = { 18,  27,  36, 255};
static const Color TEXT_MAIN   = {238, 242, 247, 255};
static const Color TEXT_MUTED  = {164, 176, 190, 255};

static const char* PIECE_TEXTURE_PATHS[13] = {
	NULL,
	"resources/chess/white_pawn.png",
	"resources/chess/white_knight.png",
	"resources/chess/white_bishop.png",
	"resources/chess/white_rook.png",
	"resources/chess/white_queen.png",
	"resources/chess/white_king.png",
	"resources/chess/black_pawn.png",
	"resources/chess/black_knight.png",
	"resources/chess/black_bishop.png",
	"resources/chess/black_rook.png",
	"resources/chess/black_queen.png",
	"resources/chess/black_king.png",
};

static void draw_board_labels(Client* cl) {
	char label[2] = {0, 0};
	float font_size = 18.0f;

	for (int file = 0; file < 8; file++) {
		label[0] = (char)('a' + file);
		Vector2 size = MeasureTextEx(cl->font, label, font_size, 0.0f);
		float x = BOARD_LEFT + file * SQ_SIZE + 0.5f * (SQ_SIZE - size.x);
		DrawTextEx(cl->font, label, (Vector2){x, BOARD_TOP + BOARD_SIZE + 4}, font_size, 0.0f, TEXT_MUTED);
	}

	for (int rank = 0; rank < 8; rank++) {
		label[0] = (char)('1' + rank);
		Vector2 size = MeasureTextEx(cl->font, label, font_size, 0.0f);
		float y = BOARD_TOP + (7 - rank) * SQ_SIZE + 0.5f * (SQ_SIZE - size.y);
		DrawTextEx(cl->font, label, (Vector2){6.0f, y}, font_size, 0.0f, TEXT_MUTED);
	}
}

Client* make_client(void) {
	Client* c = (Client*)calloc(1, sizeof(Client));
	InitWindow(WIN_W, WIN_H, "PufferLib Chess");
	SetTargetFPS(60);
	c->font = LoadFontEx("resources/shared/JetBrainsMono-Bold.ttf", 28, NULL, 0);
	GenTextureMipmaps(&c->font.texture);
	SetTextureFilter(c->font.texture, TEXTURE_FILTER_BILINEAR);

	for (int piece = 1; piece <= 12; piece++) {
		c->pieces[piece] = LoadTexture(PIECE_TEXTURE_PATHS[piece]);
		SetTextureFilter(c->pieces[piece], TEXTURE_FILTER_BILINEAR);
	}

	return c;
}

void c_render(Chess* env) {
	if (IsKeyDown(KEY_ESCAPE)) exit(0);

	if (env->human_side != 0) {
		if (IsKeyPressed(KEY_F5)) {
			c_reset(env);
		}
		if (IsKeyPressed(KEY_Q)) env->promotion_choice = 5;
		if (IsKeyPressed(KEY_R)) env->promotion_choice = 4;
		if (IsKeyPressed(KEY_B)) env->promotion_choice = 3;
		if (IsKeyPressed(KEY_N)) env->promotion_choice = 2;

		if (is_human_turn(env) && IsMouseButtonPressed(MOUSE_LEFT_BUTTON)) {
			Vector2 mouse = GetMousePosition();
			int sq = -1;
			if (mouse.x >= BOARD_LEFT && mouse.x < BOARD_LEFT + BOARD_SIZE &&
					mouse.y >= BOARD_TOP && mouse.y < BOARD_TOP + BOARD_SIZE) {
				int file = (int)((mouse.x - BOARD_LEFT) / SQ_SIZE);
				int rank_from_top = (int)((mouse.y - BOARD_TOP) / SQ_SIZE);
				int rank = 7 - rank_from_top;
				sq = SQ(rank, file);
			}

			if (sq < 0) {
				env->selected_sq = -1;
			} else {
				int8_t piece = env->board[sq];
				if (env->selected_sq < 0) {
					if (piece != EMPTY && PIECE_COLOR(piece) == env->side) {
						env->selected_sq = sq;
					}
				} else if (sq == env->selected_sq) {
					env->selected_sq = -1;
				} else if (piece != EMPTY && PIECE_COLOR(piece) == env->side) {
					env->selected_sq = sq;
				} else {
					int action_idx = find_legal_move_index(env, env->selected_sq, sq, env->promotion_choice);
					if (action_idx >= 0) {
						env->pending_human_action = action_idx;
					}
					env->selected_sq = -1;
				}
			}
		}
	}

	if (env->client == NULL) env->client = make_client();
	Client* cl = env->client;

	BeginDrawing();
	ClearBackground(BG_TOP);
	DrawRectangleGradientV(0, 0, WIN_W, WIN_H, BG_TOP, BG_BOTTOM);
	DrawRectangle(BOARD_LEFT - 8, BOARD_TOP - 8, BOARD_SIZE + 16, BOARD_SIZE + 16, BOARD_FRAME);
	DrawRectangle(BOARD_LEFT - 1, BOARD_TOP - 1, BOARD_SIZE + 2, BOARD_SIZE + 2, Fade(WHITE, 0.08f));

	int checked_king_sq = -1;
	if (in_check(env->board, env->side)) {
		checked_king_sq = find_king(env->board, env->side);
	}

	Move pseudo[MAX_MOVES], legal[MAX_MOVES];
	int num_legal = 0;
	if (env->selected_sq >= 0) {
		num_legal = legal_moves(env, legal, pseudo);
	}

	for (int sq=0; sq<64; sq++) {
		int r = RANK(sq), f = FILE(sq);
		int px = BOARD_LEFT + f * SQ_SIZE;
		int py = BOARD_TOP + (7 - r) * SQ_SIZE;

		bool light = (r + f) % 2 == 0;
		bool hl    = (sq == env->last_from || sq == env->last_to);
		bool selected = (sq == env->selected_sq);

		Color bg = light ? LIGHT_SQ : DARK_SQ;
		if (hl) bg = ColorAlphaBlend(bg, MOVE_SQ, WHITE);
		if (sq == checked_king_sq) bg = ColorAlphaBlend(bg, CHECK_SQ, WHITE);
		DrawRectangle(px, py, SQ_SIZE, SQ_SIZE, bg);
		if (selected) {
			DrawRectangleLinesEx((Rectangle){(float)px + 1.0f, (float)py + 1.0f,
				(float)SQ_SIZE - 2.0f, (float)SQ_SIZE - 2.0f}, 3.0f, SELECT_SQ);
		}
	}

	if (env->selected_sq >= 0) {
		for (int i = 0; i < num_legal; i++) {
			if (legal[i].from != env->selected_sq) continue;

			int to = legal[i].to;
			int px = BOARD_LEFT + FILE(to) * SQ_SIZE;
			int py = BOARD_TOP + (7 - RANK(to)) * SQ_SIZE;
			bool is_capture = env->board[to] != EMPTY || legal[i].ep_capture >= 0;
			Color hint = is_capture ? Fade((Color){232, 84, 84, 255}, 0.80f)
				: Fade(WHITE, 0.65f);

			if (is_capture) {
				DrawCircleLines(px + SQ_SIZE / 2, py + SQ_SIZE / 2, SQ_SIZE * 0.30f, hint);
				DrawCircleLines(px + SQ_SIZE / 2, py + SQ_SIZE / 2, SQ_SIZE * 0.33f, Fade(hint, 0.45f));
			} else {
				DrawCircle(px + SQ_SIZE / 2, py + SQ_SIZE / 2, SQ_SIZE * 0.10f, hint);
			}
		}
	}

	for (int sq = 0; sq < 64; sq++) {
		int8_t piece = env->board[sq];
		if (piece == EMPTY) continue;

		int px = BOARD_LEFT + FILE(sq) * SQ_SIZE;
		int py = BOARD_TOP + (7 - RANK(sq)) * SQ_SIZE;
		Texture2D tex = cl->pieces[piece];
		Rectangle src = {0.0f, 0.0f, (float)tex.width, (float)tex.height};
		Rectangle shadow = {(float)px + 9.0f, (float)py + 10.0f, (float)SQ_SIZE - 18.0f, (float)SQ_SIZE - 18.0f};
		Rectangle dst = {(float)px + 6.0f, (float)py + 6.0f, (float)SQ_SIZE - 12.0f, (float)SQ_SIZE - 12.0f};

		DrawTexturePro(tex, src, shadow, (Vector2){0.0f, 0.0f}, 0.0f, Fade(BLACK, 0.14f));
		DrawTexturePro(tex, src, dst, (Vector2){0.0f, 0.0f}, 0.0f, WHITE);
	}

	draw_board_labels(cl);

	int status_top = BOARD_TOP + BOARD_SIZE + 28;
	DrawRectangle(0, BOARD_TOP + BOARD_SIZE + 20, WIN_W, STATUS_H, PANEL_BG);

	char status[160];
	const char* turn_str = env->side == 0 ? "White" : "Black";
	const char* mode_str = env->selfplay ? "self play" :
		(env->human_side != 0 && env->stockfish_enabled ? "human vs Stockfish" :
		(env->stockfish_enabled ? "PPO vs Stockfish" : "PPO vs random"));
	snprintf(status, sizeof(status), "Move %d   %s to move   mode: %s",
		env->fullmove, turn_str, mode_str);
	DrawTextEx(cl->font, status, (Vector2){16.0f, (float)status_top}, 18.0f, 0.0f, TEXT_MAIN);

	char meta[160];
	snprintf(meta, sizeof(meta), "illegal moves: %.0f   halfmove: %d   controls: click to move, F5 reset, Q/R/B/N promotion, TAB toggle mode",
		env->log.illegal_moves, env->halfmove_clock);
	DrawTextEx(cl->font, meta, (Vector2){16.0f, (float)status_top + 28.0f}, 14.0f, 0.0f, TEXT_MUTED);

	EndDrawing();
}

void c_close(Chess* env) {
	stockfish_stop(&env->stockfish);
	if (env->client) {
		for (int piece = 1; piece <= 12; piece++) {
			if (env->client->pieces[piece].id != 0) {
				UnloadTexture(env->client->pieces[piece]);
			}
		}
		UnloadFont(env->client->font);
		CloseWindow();
		free(env->client);
		env->client = NULL;
	}
}
