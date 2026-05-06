#include "chess.h"

// Observation: 12 piece planes × 64 squares + 7 auxiliary scalars
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
    init(env);
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "perf",           log->perf);
    dict_set(out, "score",          log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "illegal_moves",  log->illegal_moves);
    dict_set(out, "draws",          log->draws);
    dict_set(out, "n",              log->n);
}
