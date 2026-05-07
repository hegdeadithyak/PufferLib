#include "chess.h"

#define OBS_SIZE     CHESS_OBS_SIZE
#define NUM_ATNS     1
#define ACT_SIZES    {CHESS_ACT_SIZE}
#define OBS_TENSOR_T FloatTensor

#define Env Chess
#include "vecenv.h"

void my_init(Env* env, Dict* kwargs) {
    env->num_agents = 1;
    env->selfplay   = (int)dict_get(kwargs, "selfplay")->value;
    DictItem* human_side = dict_get_unsafe(kwargs, "human_side");
    env->human_side = human_side ? (int)human_side->value : 0;
    DictItem* stockfish_enabled = dict_get_unsafe(kwargs, "stockfish_enabled");
    DictItem* stockfish_depth = dict_get_unsafe(kwargs, "stockfish_depth");
    DictItem* stockfish_movetime_ms = dict_get_unsafe(kwargs, "stockfish_movetime_ms");
    DictItem* ppo_side = dict_get_unsafe(kwargs, "ppo_side");
    DictItem* randomize_ppo_side = dict_get_unsafe(kwargs, "randomize_ppo_side");
    DictItem* max_ppo_turns = dict_get_unsafe(kwargs, "max_ppo_turns");
    DictItem* eval_shaping = dict_get_unsafe(kwargs, "eval_shaping");
    env->stockfish_enabled = stockfish_enabled ? (int)stockfish_enabled->value : 0;
    env->stockfish_depth = stockfish_depth ? (int)stockfish_depth->value : 8;
    env->stockfish_movetime_ms = stockfish_movetime_ms ? (int)stockfish_movetime_ms->value : 0;
    env->ppo_side = ppo_side ? (int)ppo_side->value : 0;
    env->randomize_ppo_side = randomize_ppo_side ? (int)randomize_ppo_side->value : 0;
    env->max_ppo_turns = max_ppo_turns ? (int)max_ppo_turns->value : 200;
    env->eval_shaping = eval_shaping ? (int)eval_shaping->value : 1;
    init(env);
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "perf",           log->perf);
    dict_set(out, "score",          log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "illegal_moves",  log->illegal_moves);
    dict_set(out, "draws",          log->draws);
    dict_set(out, "stockfish_moves", log->stockfish_moves);
    dict_set(out, "stockfish_fallbacks", log->stockfish_fallbacks);
    dict_set(out, "stockfish_cp",    log->stockfish_cp);
    dict_set(out, "n",              log->n);
}
