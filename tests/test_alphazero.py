import random

import numpy as np
import torch
import chess

from pufferlib.alphazero import (
    AlphaZeroMCTS,
    CHESS_BOARD_OBS_SIZE,
    ChessAlphaZeroNet,
    ReplaySample,
    encode_board,
    encode_move,
    legal_action_map,
    train_batch,
)


def test_chess_action_mask_matches_legal_moves():
    board = chess.Board()
    obs = encode_board(board)
    mask = obs[CHESS_BOARD_OBS_SIZE:]

    assert int(mask.sum()) == board.legal_moves.count()
    for move in board.legal_moves:
        action = encode_move(move, board.turn)
        assert action >= 0
        assert mask[action] == 1.0


def test_mcts_smoke_visits_legal_edges():
    device = torch.device("cpu")
    model = ChessAlphaZeroNet(channels=8, blocks=1).to(device)
    model.eval()

    board = chess.Board()
    mcts = AlphaZeroMCTS(
        model=model,
        device=device,
        simulations=2,
        c_puct=1.5,
        dirichlet_alpha=0.3,
        root_noise=0.0,
    )
    root = mcts.run(board, add_noise=False)

    assert set(root.edges) == set(legal_action_map(board))
    assert sum(edge.visit_count for edge in root.edges.values()) == 2


def test_train_batch_smoke():
    device = torch.device("cpu")
    model = ChessAlphaZeroNet(channels=8, blocks=1).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    board = chess.Board()
    legal_actions = np.asarray(list(legal_action_map(board))[:4], dtype=np.int16)
    probs = np.full(len(legal_actions), 1.0 / len(legal_actions), dtype=np.float32)
    replay = [
        ReplaySample(board.fen(), legal_actions, probs, 0.0),
        ReplaySample(board.fen(), legal_actions, probs, 0.0),
    ]

    metrics = train_batch(
        model=model,
        optimizer=optimizer,
        replay=replay,
        cfg={"batch_size": 2, "max_grad_norm": 5.0},
        device=device,
        rng=random.Random(0),
    )

    assert metrics["loss"] > 0
    assert metrics["policy_loss"] > 0
