#pragma once

#include <errno.h>
#include <signal.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/select.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

typedef struct StockfishEngine {
	int initialized;
	int available;
	int in_fd;
	int out_fd;
	pid_t pid;
} StockfishEngine;

typedef struct StockfishResult {
	int ok;
	int has_score;
	int score_cp;
	int mate;
	char bestmove[16];
} StockfishResult;

static void stockfish_engine_init(StockfishEngine* sf) {
	if (sf->initialized) return;
	memset(sf, 0, sizeof(*sf));
	sf->in_fd = -1;
	sf->out_fd = -1;
	sf->pid = -1;
	sf->initialized = 1;
}

static void stockfish_mark_dead(StockfishEngine* sf) {
	if (sf->in_fd >= 0) close(sf->in_fd);
	if (sf->out_fd >= 0) close(sf->out_fd);
	sf->in_fd = -1;
	sf->out_fd = -1;
	sf->available = 0;
	if (sf->pid > 0) {
		int status = 0;
		if (waitpid(sf->pid, &status, WNOHANG) == 0) {
			kill(sf->pid, SIGTERM);
			waitpid(sf->pid, &status, 0);
		}
	}
	sf->pid = -1;
}

static int stockfish_write_all(int fd, const char* data, size_t len) {
	size_t off = 0;
	while (off < len) {
		ssize_t wrote = write(fd, data + off, len - off);
		if (wrote < 0) {
			if (errno == EINTR) continue;
			return 0;
		}
		if (wrote == 0) return 0;
		off += (size_t)wrote;
	}
	return 1;
}

static int stockfish_cmd(StockfishEngine* sf, const char* fmt, ...) {
	if (!sf->available || sf->in_fd < 0) return 0;

	char line[1024];
	va_list args;
	va_start(args, fmt);
	int n = vsnprintf(line, sizeof(line) - 2, fmt, args);
	va_end(args);
	if (n < 0) return 0;
	if (n > (int)sizeof(line) - 2) n = (int)sizeof(line) - 2;
	line[n++] = '\n';
	line[n] = '\0';

	if (!stockfish_write_all(sf->in_fd, line, (size_t)n)) {
		stockfish_mark_dead(sf);
		return 0;
	}
	return 1;
}

static int stockfish_read_line(StockfishEngine* sf, char* out, size_t out_sz, int timeout_ms) {
	if (!sf->available || sf->out_fd < 0 || out_sz == 0) return 0;

	size_t n = 0;
	out[0] = '\0';
	while (n + 1 < out_sz) {
		fd_set set;
		FD_ZERO(&set);
		FD_SET(sf->out_fd, &set);

		struct timeval tv;
		tv.tv_sec = timeout_ms / 1000;
		tv.tv_usec = (timeout_ms % 1000) * 1000;

		int ready = select(sf->out_fd + 1, &set, NULL, NULL, &tv);
		if (ready < 0) {
			if (errno == EINTR) continue;
			stockfish_mark_dead(sf);
			return 0;
		}
		if (ready == 0) return 0;

		char c = '\0';
		ssize_t got = read(sf->out_fd, &c, 1);
		if (got < 0) {
			if (errno == EINTR) continue;
			stockfish_mark_dead(sf);
			return 0;
		}
		if (got == 0) {
			stockfish_mark_dead(sf);
			return 0;
		}
		if (c == '\r') continue;
		if (c == '\n') {
			out[n] = '\0';
			return 1;
		}
		out[n++] = c;
	}

	out[n] = '\0';
	return 1;
}

static int stockfish_wait_token(StockfishEngine* sf, const char* token, int timeout_ms) {
	char line[1024];
	int elapsed = 0;
	while (elapsed < timeout_ms) {
		int slice = timeout_ms - elapsed;
		if (slice > 250) slice = 250;
		if (stockfish_read_line(sf, line, sizeof(line), slice)) {
			if (strstr(line, token) != NULL) return 1;
		}
		elapsed += slice;
	}
	return 0;
}

static int stockfish_start(StockfishEngine* sf) {
	stockfish_engine_init(sf);
	if (sf->available) return 1;

	signal(SIGPIPE, SIG_IGN);

	int child_stdin[2];
	int child_stdout[2];
	if (pipe(child_stdin) != 0) return 0;
	if (pipe(child_stdout) != 0) {
		close(child_stdin[0]);
		close(child_stdin[1]);
		return 0;
	}

	pid_t pid = fork();
	if (pid < 0) {
		close(child_stdin[0]);
		close(child_stdin[1]);
		close(child_stdout[0]);
		close(child_stdout[1]);
		return 0;
	}

	if (pid == 0) {
		dup2(child_stdin[0], STDIN_FILENO);
		dup2(child_stdout[1], STDOUT_FILENO);
		dup2(child_stdout[1], STDERR_FILENO);
		close(child_stdin[0]);
		close(child_stdin[1]);
		close(child_stdout[0]);
		close(child_stdout[1]);

		execlp("stockfish", "stockfish", (char*)NULL);
		execl("./ocean/chess/stockfish", "stockfish", (char*)NULL);
		execl("ocean/chess/stockfish", "stockfish", (char*)NULL);
		execl("/usr/games/stockfish", "stockfish", (char*)NULL);
		execl("/usr/bin/stockfish", "stockfish", (char*)NULL);
		execl("/usr/local/bin/stockfish", "stockfish", (char*)NULL);
		_exit(127);
	}

	close(child_stdin[0]);
	close(child_stdout[1]);
	sf->in_fd = child_stdin[1];
	sf->out_fd = child_stdout[0];
	sf->pid = pid;
	sf->available = 1;

	if (!stockfish_cmd(sf, "uci") || !stockfish_wait_token(sf, "uciok", 30000)) {
		stockfish_mark_dead(sf);
		return 0;
	}

	stockfish_cmd(sf, "setoption name Threads value 1");
	stockfish_cmd(sf, "setoption name Hash value 16");
	if (!stockfish_cmd(sf, "isready") || !stockfish_wait_token(sf, "readyok", 3000)) {
		stockfish_mark_dead(sf);
		return 0;
	}

	stockfish_cmd(sf, "ucinewgame");
	return 1;
}

static void stockfish_stop(StockfishEngine* sf) {
	if (!sf->initialized) return;

	if (sf->available && sf->in_fd >= 0) {
		stockfish_cmd(sf, "quit");
	}
	if (sf->in_fd >= 0) close(sf->in_fd);
	if (sf->out_fd >= 0) close(sf->out_fd);
	sf->in_fd = -1;
	sf->out_fd = -1;
	sf->available = 0;

	if (sf->pid > 0) {
		int status = 0;
		for (int i = 0; i < 10; i++) {
			if (waitpid(sf->pid, &status, WNOHANG) == sf->pid) {
				sf->pid = -1;
				return;
			}
			usleep(10000);
		}
		kill(sf->pid, SIGTERM);
		waitpid(sf->pid, &status, 0);
		sf->pid = -1;
	}
}

static void stockfish_parse_score_line(const char* line, StockfishResult* result) {
	const char* score = strstr(line, " score ");
	if (score == NULL) return;
	score += 7;

	if (strncmp(score, "cp ", 3) == 0) {
		result->score_cp = atoi(score + 3);
		result->mate = 0;
		result->has_score = 1;
	} else if (strncmp(score, "mate ", 5) == 0) {
		int mate = atoi(score + 5);
		result->mate = mate;
		result->score_cp = mate > 0 ? 100000 - mate * 1000 : -100000 - mate * 1000;
		result->has_score = 1;
	}
}

static int stockfish_go(StockfishEngine* sf, const char* fen, int depth,
		int movetime_ms, StockfishResult* result) {
	memset(result, 0, sizeof(*result));
	if (!stockfish_start(sf)) return 0;

	if (!stockfish_cmd(sf, "position fen %s", fen)) return 0;
	if (movetime_ms > 0) {
		if (!stockfish_cmd(sf, "go movetime %d", movetime_ms)) return 0;
	} else {
		if (depth < 1) depth = 8;
		if (!stockfish_cmd(sf, "go depth %d", depth)) return 0;
	}

	char line[1024];
	int elapsed = 0;
	const int timeout_ms = movetime_ms > 0 ? movetime_ms + 3000 : 10000;
	while (elapsed < timeout_ms) {
		int slice = timeout_ms - elapsed;
		if (slice > 250) slice = 250;
		if (stockfish_read_line(sf, line, sizeof(line), slice)) {
			if (strncmp(line, "info ", 5) == 0) {
				stockfish_parse_score_line(line, result);
			} else if (strncmp(line, "bestmove ", 9) == 0) {
				sscanf(line, "bestmove %15s", result->bestmove);
				result->ok = strcmp(result->bestmove, "(none)") != 0;
				return result->ok;
			}
		}
		elapsed += slice;
	}

	stockfish_mark_dead(sf);
	return 0;
}

static int stockfish_new_game(StockfishEngine* sf) {
	if (!stockfish_start(sf)) return 0;
	if (!stockfish_cmd(sf, "ucinewgame")) return 0;
	if (!stockfish_cmd(sf, "isready")) return 0;
	return stockfish_wait_token(sf, "readyok", 3000);
}
