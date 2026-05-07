#include "chess.h"

static int mouse_to_square(Vector2 mouse) {
	if (mouse.x < BOARD_LEFT || mouse.x >= BOARD_LEFT + BOARD_SIZE ||
			mouse.y < BOARD_TOP || mouse.y >= BOARD_TOP + BOARD_SIZE) {
		return -1;
	}

	int file = (int)((mouse.x - BOARD_LEFT) / SQ_SIZE);
	int rank_from_top = (int)((mouse.y - BOARD_TOP) / SQ_SIZE);
	int rank = 7 - rank_from_top;
	return SQ(rank, file);
}

static void handle_click(Chess* env) {
	if (env->human_side != 0) return;
	if (!IsMouseButtonPressed(MOUSE_LEFT_BUTTON)) return;

	int sq = mouse_to_square(GetMousePosition());
	if (sq < 0) {
		env->selected_sq = -1;
		return;
	}

	int8_t piece = env->board[sq];
	if (env->selected_sq < 0) {
		if (piece != EMPTY && PIECE_COLOR(piece) == env->side) {
			env->selected_sq = sq;
		}
		return;
	}

	if (sq == env->selected_sq) {
		env->selected_sq = -1;
		return;
	}

	if (piece != EMPTY && PIECE_COLOR(piece) == env->side) {
		env->selected_sq = sq;
		return;
	}

	int action_idx = find_legal_move_index(env, env->selected_sq, sq, env->promotion_choice);
	if (action_idx < 0) {
		env->selected_sq = -1;
		return;
	}

	env->actions[0] = (float)action_idx;
	c_step(env);
}

static void handle_keyboard(Chess* env) {
	if (IsKeyPressed(KEY_F5)) {
		c_reset(env);
		return;
	}

	if (IsKeyPressed(KEY_Q)) env->promotion_choice = 5;
	if (IsKeyPressed(KEY_R)) env->promotion_choice = 4;
	if (IsKeyPressed(KEY_B)) env->promotion_choice = 3;
	if (IsKeyPressed(KEY_N)) env->promotion_choice = 2;

	if (IsKeyPressed(KEY_TAB)) {
		env->selfplay = !env->selfplay;
		env->selected_sq = -1;
	}
}

int main(void) {
	Chess env = {0};
	allocate_cchess(&env);
	env.num_agents = 1;
	env.selfplay = 0;
	env.human_side = 1;
	env.stockfish_enabled = 1;
	env.stockfish_depth = 8;
	env.max_ppo_turns = 300;
	init(&env);
	c_reset(&env);

	while (!WindowShouldClose()) {
		handle_keyboard(&env);
		handle_click(&env);
		c_render(&env);
		c_step(&env);
	}

	c_close(&env);
	free_allocated_cchess(&env);
	return 0;
}
